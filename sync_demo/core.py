"""Fixed, state-driven 3D–2D registration. Not a robot-control calibration."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

DEFAULT_CAMERA = {
    'lensmodel': 'LENSMODEL_OPENCV8', 'image_size': [1920, 1080],
    'intrinsics': [1450.455178496827, 1429.339083433449, 866.9314576837802, 568.0929141757828,
                   -.0071200513187242446, -.02094696502821218, .0017541009755967043,
                   .007817347723294591, .04635027968924447, -.003534950111442823,
                   -.001274634264479936, -.0023892671778790037]
}

def camera_parameters(config, size, scale=True):
    if config.get('lensmodel') != 'LENSMODEL_OPENCV8':
        raise ValueError('Expected lensmodel LENSMODEL_OPENCV8 (fx, fy, cx, cy, then 8 distortion coefficients).')
    a = np.asarray(config.get('intrinsics'), dtype=float)
    wh = np.asarray(config.get('image_size'), dtype=float)
    if a.shape != (12,) or not np.isfinite(a).all() or np.any(a[:2] <= 0):
        raise ValueError('intrinsics must contain 12 finite numbers; fx and fy must be positive.')
    if wh.shape != (2,) or not np.isfinite(wh).all() or np.any(wh <= 0):
        raise ValueError('image_size must be [width, height], both positive.')
    if not scale and not np.array_equal(wh, size):
        raise ValueError('Camera image_size does not match video. Enable scaling or supply matching intrinsics.')
    sx, sy = np.asarray(size) / wh
    K = np.array([[a[0]*sx, 0, a[2]*sx], [0, a[1]*sy, a[3]*sy], [0, 0, 1.]])
    return K, a[4:].copy()

def discover(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError('Folder not found on the computer running this app.')
    found=[]
    for csvpath in sorted(folder.rglob('episode_*.csv')):
        name=csvpath.stem
        # Supports standard session/raw/chunk-*/ layout or flattened episode folders.
        candidates=list(folder.glob(f'**/observation.images.endoscope.left/{name}.mp4'))
        session = csvpath.parent.parent.parent if csvpath.parent.parent.name=='raw' else None
        if session:
            preferred=list(session.glob(f'videos/**/observation.images.endoscope.left/{name}.mp4'))
            if preferred:candidates=preferred
        if not candidates:
            candidates=[p for p in folder.rglob('*.mp4') if p.stem==name and ('left' in p.name.lower() or 'left' in str(p.parent).lower())]
        if not candidates:
            candidates=list(csvpath.parent.glob(f'{name}.mp4'))
        if len(candidates)!=1:
            continue
        found.append({'csv':str(csvpath),'video':str(candidates[0]),'label':str(csvpath.relative_to(folder)), 'episode':name})
    if not found:
        raise ValueError('No unambiguous CSV + left MP4 pair found. Select the session root containing raw/ and videos/, or an episode folder containing a matching CSV and MP4.')
    return found

def read_episode(pair):
    csvpath=Path(pair['csv']); video=Path(pair['video'])
    with csvpath.open(newline='') as f: rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Empty CSV.')
    if [int(r['frame_index']) for r in rows] != list(range(len(rows))):
        raise ValueError('frame_index must be contiguous and zero-based. Refusing to guess video–row alignment.')
    state=np.array([[float(r[f'state_{j}']) for j in range(16)] for r in rows])
    if not np.isfinite(state).all():raise ValueError('State contains NaN or infinite values.')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Cannot decode the selected MP4.')
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
    size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    cap.release()
    if n!=len(rows):raise ValueError(f'CSV has {len(rows)} rows but video reports {n} frames. Cannot safely pair by row.')
    if fps<=0 or min(size)<=0:raise ValueError('Invalid video dimensions/FPS.')
    if n<10:raise ValueError('At least 10 frames are needed for this workflow.')
    samples=np.unique(np.rint(np.linspace(0,n-1,min(12,n))).astype(int)).tolist()
    ranges=[float(np.linalg.norm(np.ptp(state[:,j:j+3],axis=0))*1000) for j in [0,8]]
    return {**pair,'rows':rows,'state':state,'n':n,'fps':fps,'size':size,'samples':samples,
            'fingerprint':hashlib.sha256(csvpath.read_bytes()).hexdigest(),
            'motion_range_mm':ranges,'suggested_arm':int(np.argmax(ranges))+1}

def frame_jpeg(episode, index):
    if not 0<=index<episode['n']:raise ValueError('Frame index out of range.')
    cap=cv2.VideoCapture(episode['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,index)
    ok,im=cap.read();cap.release()
    if not ok:raise ValueError('Frame could not be decoded.')
    return cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tobytes()

def select_action_frames(episode, arm=1, count=12):
    """Deterministic coverage of action grasp range, transitions and SE(3) diversity.

    Selection only: does not alter state, image pairing, or registration inputs.
    Quaternion matrices eliminate q versus -q sign artifacts.
    """
    from scipy.ndimage import median_filter
    if arm not in [1,2] or not 8<=count<=40:raise ValueError('Choose PSM1/PSM2 and 8–40 representative frames.')
    try:a=np.array([[float(r[f'action_{j}']) for j in range((arm-1)*8,arm*8)] for r in episode['rows']])
    except (KeyError,TypeError,ValueError):raise ValueError('This CSV does not contain readable action pose/gripper fields for the selected arm.')
    norms=np.linalg.norm(a[:,3:7],axis=1)
    valid=np.isfinite(a).all(axis=1)&(norms>.5)&(norms<1.5)
    indices=np.flatnonzero(valid)
    if len(indices)<8:raise ValueError(f'Only {len(indices)} valid action rows for PSM{arm}. All-zero placeholder actions are excluded. Choose the active arm or use manual/uniform sampling.')
    count=min(count,len(indices));s=a[indices];n=episode['n'];fps=episode['fps']
    grip=median_filter(s[:,7],size=min(5,len(s)//2*2+1),mode='nearest')
    p=s[:,:3];rotation=Rotation.from_quat(s[:,3:7]).as_matrix().reshape(-1,9)
    def robust_features(x,floor):
        lo,hi=np.percentile(x,[5,95],axis=0)
        spread=max(float(np.linalg.norm(hi-lo)),floor)
        return np.clip((x-np.median(x,axis=0))/spread,-3,3)
    posfeat=robust_features(p,.005)
    rotfeat=robust_features(rotation,.25)
    gspan=float(np.ptp(grip));gfeat=((grip-np.median(grip))/max(float(np.percentile(grip,95)-np.percentile(grip,5)),.1))[:,None]
    timefeat=(indices/max(n-1,1))[:,None]*.35
    feature=np.c_[posfeat,rotfeat,gfeat,timefeat]
    picked=[];reason={};gap=max(1,round(.3*fps))
    def add(k,label,enforce=True):
        k=int(k)
        if k in reason or len(picked)>=count:return False
        if enforce and any(abs(int(indices[k])-int(indices[j]))<gap for j in picked):return False
        picked.append(k);reason[k]=label;return True
    if gspan>.02:
        # Stable representatives nearest robust low/high scalar levels, not claims
        # about the physical meaning of "open" and "closed".
        for percentile,label in [(5,'Low action gripper'),(95,'High action gripper')]:
            delta=abs(grip-np.percentile(grip,percentile));near=np.flatnonzero(delta<=delta.min()+max(gspan*.015,.001))
            candidates=sorted(near,key=lambda k:abs(k-np.median(near)))
            for k in candidates:
                if add(k,label):break
        hop=max(1,round(.12*fps));changes=np.zeros(len(indices))
        for k in range(hop,len(indices)-hop):
            if indices[k+hop]-indices[k-hop]<=hop*3:changes[k]=abs(grip[k+hop]-grip[k-hop])
        centers=[]
        for k in np.argsort(-changes,kind='stable'):
            if changes[k]<max(gspan*.15,.03):break
            if any(abs(indices[k]-indices[j])<fps*.75 for j in centers):continue
            centers.append(int(k))
            add(max(0,k-hop),'Before action grasp change')
            add(min(len(indices)-1,k+hop),'After action grasp change',False)
            if len(centers)>=min(2,count//6):break
    # Max-min sampling covers large translations, rotations and repeated phases.
    if not picked:add(len(indices)//2,'Action trajectory coverage')
    while len(picked)<count:
        distances=np.min(np.sum((feature[:,None]-feature[picked][None])**2,axis=2),axis=1)
        distances[picked]=-1
        for j in picked:distances[np.abs(indices-indices[j])<gap]=-1
        if distances.max()<0:
            distances=np.min(np.sum((feature[:,None]-feature[picked][None])**2,axis=2),axis=1);distances[picked]=-1
        k=int(np.argmax(distances));add(k,'Position / orientation / time coverage',False)
    output=[]
    for j,k in enumerate(sorted(picked,key=lambda k:indices[k])):
        output.append({'frame':int(indices[k]),'role':'validation' if j%4==1 else 'fit','reason':reason[k],
                       'action_gripper':float(a[indices[k],7])})
    return {'frames':output,'valid_rows':len(indices),'excluded_rows':int(n-len(indices)),'arm':arm,
            'action_position_span_mm':float(np.linalg.norm(np.ptp(p,axis=0))*1000),'action_gripper_range':[float(s[:,7].min()),float(s[:,7].max())],
            'note':'Action is used only to select candidate image frames. Registration and skeleton motion still use observation/state. Grasp-change candidates are not confirmed clutch events; inspect image visibility and physical response.'}

AXES={'+X':np.array([1.,0.,0.]),'+Y':np.array([0.,1.,0.]),'+Z':np.array([0.,0.,1.])}

def geometry_parameters(config=None):
    """Validate the user-selected, physically constrained jaw geometry."""
    config=dict(config or {})
    center_axis=config.get('center_axis','+Y');opening_axis=config.get('opening_axis','+X')
    mode=config.get('offset_mode','zero')
    if center_axis not in AXES or opening_axis not in AXES:
        raise ValueError('Centerline and opening direction must be +X, +Y, or +Z.')
    if center_axis==opening_axis:
        raise ValueError('Opening direction must use a different tool axis from the centerline.')
    if mode not in ['zero','manual','optimize']:
        raise ValueError('Offset mode must be zero, manual, or optimize.')
    manual=np.asarray(config.get('manual_offset_mm',[0,0,0]),dtype=float)
    if manual.shape!=(3,) or not np.isfinite(manual).all() or np.any(np.abs(manual)>60):
        raise ValueError('Manual offset must contain finite X/Y/Z values between -60 and 60 mm.')
    if mode=='zero':manual=np.zeros(3)
    return {'center_axis':center_axis,'opening_axis':opening_axis,'optimize_axes':bool(config.get('optimize_axes',True)),'offset_mode':mode,
            'manual_offset_mm':manual.tolist()}

def project_geometry(q, state, center, K, D, indices, geometry=None):
    """Project Root/A/B using one geometry definition for fit, validation and export.

    state_7 is the full jaw opening angle in radians. The two jaw tips rotate by
    plus/minus half of that angle around the fixed centerline.
    """
    geometry=geometry_parameters(geometry)
    s=state[np.asarray(indices,dtype=int)]
    r=Rotation.from_quat(s[:,3:7]).as_matrix()
    optimized=geometry['offset_mode']=='optimize'
    expected=10 if optimized else 7
    if len(q)!=expected:raise ValueError(f'Expected {expected} fit parameters for offset mode {geometry["offset_mode"]}.')
    offset=np.asarray(q[6:9] if optimized else geometry['manual_offset_mm'],dtype=float)
    if not optimized:offset=offset/1000
    L=np.exp(q[9] if optimized else q[6]);angle=s[:,7]/2
    centerline=AXES[geometry['center_axis']];opening=AXES[geometry['opening_axis']]
    points=np.empty((len(s),3,3));points[:,0]=offset
    forward=L*np.cos(angle)[:,None]*centerline
    spread=L*np.sin(angle)[:,None]*opening
    # Tip A follows the selected positive opening axis; Tip B follows negative.
    points[:,1]=offset+forward+spread;points[:,2]=offset+forward-spread
    base=np.einsum('nij,nkj->nki',r,points)+s[:,None,:3]-center
    Rc=Rotation.from_rotvec(q[:3]).as_matrix()
    cam=base@Rc.T+q[3:6]
    uv=cv2.projectPoints(cam.reshape(-1,3),np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,3,2)
    return uv,cam

def _fit_one_geometry(episode, annotations, camera, arm=1, scale=True, geometry=None, progress=lambda x:None, starts=8):
    if arm not in [1,2]:raise ValueError('Choose PSM1 or PSM2.')
    state=episode['state'][:,(arm-1)*8:arm*8]
    norms=np.linalg.norm(state[:,3:7],axis=1)
    if np.any(norms<.5) or np.any(norms>1.5):raise ValueError('Selected arm contains invalid pose quaternions.')
    K,D=camera_parameters(camera,episode['size'],scale);geometry=geometry_parameters(geometry)
    records=sorted(annotations,key=lambda a:int(a['frame']))
    ids=np.array([int(a['frame']) for a in records],int)
    if len(set(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=episode['n']):raise ValueError('Duplicate or invalid annotation frame index.')
    uv=np.zeros((len(ids),3,2));weights=np.zeros((len(ids),3,1))
    for i,a in enumerate(records):
        if a.get('role') not in ['fit','validation']:raise ValueError('Frame role must be fit or validation.')
        points=a.get('points',[])
        if len(points)!=3:raise ValueError('Each frame requires [Root, A, B], using null for skipped tips.')
        if points[0] is None:raise ValueError(f'Frame {ids[i]} needs a Root point.')
        for j,p in enumerate(points):
            if p is None:continue
            p=np.asarray(p,dtype=float)
            if p.shape!=(2,) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>=episode['size']):raise ValueError('Point must be within the original image.')
            uv[i,j]=p;weights[i,j]=1
    train=np.array([i for i,a in enumerate(records) if a['role']=='fit'],int)
    held=np.array([i for i,a in enumerate(records) if a['role']=='validation'],int)
    if len(train)<6:raise ValueError('Annotate at least 6 fitting frames (8+ recommended), each with Root.')
    if np.sum(np.all(weights[train,:,0]>0,axis=1))<4:raise ValueError('At least 4 fitting frames need Root + A + B.')
    if len(held)<2:raise ValueError('Reserve at least 2 annotated validation frames that are not used for fitting.')
    if np.linalg.norm(np.ptp(state[ids[train],:3],axis=0))<.001:raise ValueError('Selected fitting frames have less than 1 mm position variation. Add frames from the large-movement section.')
    center=state[:,:3].mean(0)
    fixed_offset=np.asarray(geometry['manual_offset_mm'])/1000
    pnp_offset=np.zeros(3) if geometry['offset_mode']=='optimize' else fixed_offset
    root_base=np.einsum('nij,j->ni',Rotation.from_quat(state[ids[train],3:7]).as_matrix(),pnp_offset)+state[ids[train],:3]
    progress('Initializing camera pose from fitting frames only…')
    ok,rv,tv=cv2.solvePnP((root_base-center).astype(float),uv[train,0],K,D,flags=cv2.SOLVEPNP_SQPNP)
    if not ok:raise ValueError('PnP initialization failed. Use more spatially diverse frames.')
    def residual(q):
        pred,cam=project_geometry(q,state,center,K,D,ids[train],geometry)
        return np.r_[((pred-uv[train])*weights[train]).ravel(),np.minimum(cam[:,:,2]-.03,0).ravel()*1e4]
    if geometry['offset_mode']=='optimize':
        lo=np.r_[[-np.inf]*3,[-2,-2,.03],[-.06]*3,np.log(.003)]
        hi=np.r_[[np.inf]*3,[2,2,2],[.06]*3,np.log(.035)]
    else:
        lo=np.r_[[-np.inf]*3,[-2,-2,.03],np.log(.003)]
        hi=np.r_[[np.inf]*3,[2,2,2],np.log(.035)]
    best=None;length_guesses=np.geomspace(.006,.028,max(2,starts))
    for i in range(starts):
        progress(f'Robust fit: initialization {i+1} / {starts}')
        q=np.r_[rv.ravel(),tv.ravel(),np.zeros(3) if geometry['offset_mode']=='optimize' else [],np.log(length_guesses[i])]
        q=np.clip(q,lo+1e-7,hi-1e-7)
        result=least_squares(residual,q,bounds=(lo,hi),loss='soft_l1',f_scale=8,x_scale='jac',max_nfev=450)
        if np.isfinite(result.cost) and (best is None or result.cost<best.cost):best=result
    if best is None:raise ValueError('No finite fit found.')
    q=best.x;Rc=Rotation.from_rotvec(q[:3]).as_matrix();C=np.eye(4);C[:3,:3]=Rc;C[:3,3]=q[3:6]-Rc@center
    projected,cam=project_geometry(q,state,center,K,D,np.arange(len(state)),geometry)
    error=np.linalg.norm(projected[ids]-uv,axis=2);valid=weights[:,:,0]>0
    def rms(sel):return float(np.sqrt(np.mean(error[sel][valid[sel]]**2)))
    axeslocal=np.vstack([np.zeros(3),np.eye(3)*.002])
    base=np.einsum('nij,kj->nki',Rotation.from_quat(state[:,3:7]).as_matrix(),axeslocal)+state[:,None,:3]
    axcam=base@Rc.T+C[:3,3]
    axes=cv2.projectPoints(axcam.reshape(-1,3),np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,4,2)
    warnings=['Visual registration with fitted schematic geometry; NOT a validated transform for robot control.',
              'No temporal shift applied. Source row i is paired with video frame i; timestamp semantics remain unverified.']
    if rms(held)>30:warnings.append('Validation RMS exceeds 30 px. Inspect landmarks, point identity, timing and the skeleton model.')
    if np.ptp(state[ids[train],7])<.3:warnings.append('Limited gripper variation: the selected opening direction is weakly checked by these frames.')
    if not np.all(cam[:,:,2]>0):warnings.append('Some projected points are behind the camera and will be hidden.')
    length_index=9 if geometry['offset_mode']=='optimize' else 6
    if np.isclose(q[length_index],lo[length_index],atol=.01) or np.isclose(q[length_index],hi[length_index],atol=.01):
        warnings.append('Jaw length reached its bound; treat this fit cautiously.')
    if geometry['offset_mode']=='optimize' and (np.any(np.isclose(q[6:9],lo[6:9],atol=.001)) or np.any(np.isclose(q[6:9],hi[6:9],atol=.001))):
        warnings.append('Optimized tool-origin offset reached its bound; treat this fit cautiously.')
    offset=(q[6:9] if geometry['offset_mode']=='optimize' else fixed_offset)
    details=[]
    for i,a in enumerate(records):details.append({'frame':int(ids[i]),'role':a['role'],'rms_px':float(np.sqrt(np.mean(error[i][valid[i]]**2)))})
    return {'status':'VISUAL_REGISTRATION_NOT_CONTROL_CALIBRATION','arm':arm,'camera':camera,'K_used':K.tolist(),'D_used':D.tolist(),
            'parameter_vector':q.tolist(),'center_m':center.tolist(),'T_camera_PSMbase':C.tolist(),'geometry':geometry,
            'pivot_offset_tool_m':offset.tolist(),'jaw_length_mm':float(np.exp(q[length_index])*1000),'opening_definition':'state_7 radians, total jaw opening angle; each jaw uses ±state_7/2',
            'fit_rms_px':rms(train),'validation_rms_px':rms(held),'details':details,'warnings':warnings,'dataset_fingerprint':episode['fingerprint'],
            'annotations':records,'projected':projected.tolist(),'axes':axes.tolist(),'visible':np.all(cam[:,:,2]>0,axis=1).tolist(),
            'optimizer_success':bool(best.success),'optimizer_message':best.message,'source':{'csv':episode['csv'],'video':episode['video'],'fps':episode['fps'],'size':episode['size']}}

def fit(episode, annotations, camera, arm=1, scale=True, geometry=None, progress=lambda x:None, starts=8):
    """Fit one fixed axis pair or select the best pair from six discrete choices.

    Automatic axis selection enumerates only orthogonal positive tool axes. Each
    candidate is independently fitted, and the winner is selected using fitting
    RMS only. Held-out validation pixels never influence the selected pair or its
    optimized parameters.
    """
    geometry=geometry_parameters(geometry)
    if not geometry['optimize_axes']:
        return _fit_one_geometry(episode,annotations,camera,arm,scale,geometry,progress,starts)
    candidates=[];results=[];pairs=[(c,o) for c in AXES for o in AXES if c!=o]
    for i,(center_axis,opening_axis) in enumerate(pairs):
        candidate={**geometry,'center_axis':center_axis,'opening_axis':opening_axis,'optimize_axes':False}
        def report(message,i=i,center_axis=center_axis,opening_axis=opening_axis):
            progress(f'Axes {i+1} / {len(pairs)} ({center_axis} center, {opening_axis} opening): {message}')
        result=_fit_one_geometry(episode,annotations,camera,arm,scale,candidate,report,starts)
        results.append(result);candidates.append({'center_axis':center_axis,'opening_axis':opening_axis,
            'fit_rms_px':result['fit_rms_px'],'validation_rms_px':result['validation_rms_px'],
            'jaw_length_mm':result['jaw_length_mm'],'pivot_offset_tool_m':result['pivot_offset_tool_m']})
    selected_index=int(np.argmin([x['fit_rms_px'] for x in candidates]));selected=results[selected_index]
    selected['geometry']['optimize_axes']=True
    selected['axis_selection']={'mode':'discrete_positive_coordinate_axes','selection_metric':'fit_rms_px',
        'validation_used_for_selection':False,'selected_index':selected_index,'candidates':candidates}
    selected['warnings'].append('Centerline and opening directions were selected from six orthogonal positive-axis pairs using fitting frames only.')
    return selected

def export_video(episode,result,path,progress=lambda x:None):
    cap=cv2.VideoCapture(episode['video']);width=1280;height=round(episode['size'][1]*width/episode['size'][0]/2)*2
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'avc1'),episode['fps'],(width,height+80))
    if not writer.isOpened():raise ValueError('H.264 encoder unavailable in this OpenCV build. Use the documented OpenCV environment.')
    projected=np.asarray(result['projected']);axes=np.asarray(result['axes']);colors=[(70,70,245),(70,220,70),(245,135,65)]
    try:
        for i in range(episode['n']):
            ok,im=cap.read()
            if not ok:raise ValueError(f'Video decoding failed at frame {i}.')
            if result['visible'][i] and np.isfinite(projected[i]).all() and np.max(np.abs(projected[i]))<1e6 and np.max(np.abs(axes[i]))<1e6:
                pt=np.rint(projected[i]).astype(int);ap=np.rint(axes[i]).astype(int)
                cv2.line(im,tuple(ap[0]),tuple(pt[0]),(235,235,235),3,cv2.LINE_AA)
                for j in [1,2]:cv2.line(im,tuple(pt[0]),tuple(pt[j]),(50,218,255),3,cv2.LINE_AA)
                for p in pt:cv2.circle(im,tuple(p),4,(50,218,255),-1,cv2.LINE_AA)
                for j,col in enumerate(colors):cv2.arrowedLine(im,tuple(ap[0]),tuple(ap[j+1]),col,3,cv2.LINE_AA,tipLength=.2)
            canvas=np.full((height+80,width,3),22,np.uint8);canvas[40:height+40]=cv2.resize(im,(width,height))
            cv2.putText(canvas,f'PSM{result["arm"]} | STATE-driven schematic | Visual registration, NOT control calibration',(16,27),0,.64,(240,240,240),1,cv2.LINE_AA)
            g=episode['state'][i,(result['arm']-1)*8+7]
            cv2.putText(canvas,f'Frame {i} | Video {i/episode["fps"]:.2f}s | g={g:.3f} | XYZ axes: 2 mm | Source FPS, no lag correction',(16,height+67),0,.6,(240,240,240),1,cv2.LINE_AA)
            writer.write(canvas)
            if i%60==0:progress(f'Exporting video: {i+1} / {episode["n"]}')
    finally:cap.release();writer.release()
    cap=cv2.VideoCapture(str(path));n=0
    while True:
        ok,_=cap.read()
        if not ok:break
        n+=1
    cap.release()
    if n!=episode['n']:raise ValueError(f'Output verification failed: {n} / {episode["n"]} frames decoded.')
    return {'frames':n,'duration_s':n/episode['fps']}
