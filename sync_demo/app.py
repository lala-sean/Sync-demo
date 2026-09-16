"""Loopback-only local application. No external service receives source data."""
import json,os,secrets,threading,uuid
from pathlib import Path
from flask import Flask,request,jsonify,send_file,Response
import numpy as np
from .core import DEFAULT_CAMERA,discover,read_episode,frame_jpeg,fit,export_video,camera_parameters,select_action_frames

ROOT=Path(__file__).parent
OUTPUT=ROOT/'outputs';OUTPUT.mkdir(exist_ok=True)
app=Flask(__name__,static_folder='static');app.config['MAX_CONTENT_LENGTH']=2*1024**3
TOKEN=secrets.token_hex(24);datasets={};jobs={};pairsets={};compute_lock=threading.Lock()

@app.before_request
def protect():
    if request.host.split(':')[0] not in ['127.0.0.1','localhost','[::1]']:return jsonify(error='Use localhost.'),403
    if request.method=='POST' and request.headers.get('X-Local-Token')!=TOKEN:return jsonify(error='Refresh this local page before continuing.'),403

@app.errorhandler(Exception)
def error(e):
    from werkzeug.exceptions import HTTPException
    return jsonify(error=str(e)),e.code if isinstance(e,HTTPException) else 400

@app.get('/')
def index():
    return Response((ROOT/'static/index.html').read_text().replace('__LOCAL_TOKEN__',TOKEN),mimetype='text/html')

@app.get('/api/defaults')
def defaults():return jsonify(camera=DEFAULT_CAMERA)

@app.post('/api/discover')
def discovery():
    pairs=discover(request.json['folder']);key=uuid.uuid4().hex;pairsets[key]=pairs
    return jsonify(id=key,episodes=[{'index':i,'label':x['label']} for i,x in enumerate(pairs)])

@app.post('/api/upload')
def upload():
    folder=OUTPUT/('import_'+uuid.uuid4().hex);folder.mkdir()
    files=request.files.getlist('files')
    if not files:raise ValueError('No CSV/MP4 files selected.')
    for f in files:
        p=Path(f.filename)
        if p.is_absolute() or '..' in p.parts or p.suffix.lower() not in ['.csv','.mp4']:raise ValueError('Only relative CSV/MP4 paths are accepted.')
        target=folder/p;target.parent.mkdir(parents=True,exist_ok=True);f.save(target)
    pairs=discover(folder);key=uuid.uuid4().hex;pairsets[key]=pairs
    return jsonify(id=key,episodes=[{'index':i,'label':x['label']} for i,x in enumerate(pairs)])

@app.post('/api/load')
def load():
    body=request.json;ep=read_episode(pairsets[body['id']][int(body['index'])]);key=uuid.uuid4().hex;datasets[key]=ep
    meta={k:ep[k] for k in ['n','fps','size','samples','fingerprint','motion_range_mm','suggested_arm','episode']}
    meta.update(id=key,gripper=ep['state'][:,[7,15]].tolist(),trajectory=ep['state'][:,[0,1,2,8,9,10]].tolist())
    return jsonify(meta)

@app.get('/api/frame/<key>/<int:index>')
def frame(key,index):
    response=Response(frame_jpeg(datasets[key],index),mimetype='image/jpeg')
    response.headers['Cache-Control']='private, max-age=86400'
    return response

@app.post('/api/camera')
def camera():
    b=request.json;K,D=camera_parameters(b['camera'],datasets[b['dataset']]['size'],b.get('scale',True))
    return jsonify(K=K.tolist(),D=D.tolist())

@app.post('/api/select-action')
def action_selection():
    b=request.json
    return jsonify(select_action_frames(datasets[b['dataset']],int(b.get('arm',1)),int(b.get('count',12))))

def start_job(kind,worker):
    if not compute_lock.acquire(blocking=False):raise ValueError('Another fit/export is running. Wait for it to finish.')
    key=uuid.uuid4().hex;jobs[key]={'status':'running','kind':kind,'message':'Starting…'}
    def work():
        try:
            result=worker(key,lambda msg:jobs[key].update(message=msg));jobs[key].update(status='done',message='Complete',result=result)
        except Exception as e:jobs[key].update(status='error',message=str(e))
        finally:compute_lock.release()
    threading.Thread(target=work,daemon=True).start();return jsonify(id=key)

@app.post('/api/fit')
def run_fit():
    b=request.json;ep=datasets[b['dataset']]
    def work(key,progress):
        result=fit(ep,b['annotations'],b['camera'],int(b['arm']),b.get('scale',True),b.get('geometry'),progress)
        result['dataset_id']=b['dataset'];(OUTPUT/f'{key}.json').write_text(json.dumps(result,indent=2))
        return result
    return start_job('fit',work)

@app.post('/api/export')
def export():
    b=request.json;previous=jobs[b['fit']]
    if previous['status']!='done' or previous['kind']!='fit':raise ValueError('A completed fit is required.')
    result=previous['result'];ep=datasets[result['dataset_id']]
    def work(key,progress):return export_video(ep,result,OUTPUT/f'{key}.mp4',progress)
    return start_job('export',work)

@app.get('/api/jobs/<key>')
def job(key):return jsonify(jobs[key])

@app.get('/api/download/<key>')
def download(key):
    job=jobs[key]
    if job['status']!='done':raise ValueError('Job is not complete.')
    suffix='.mp4' if job['kind']=='export' else '.json'
    return send_file(OUTPUT/(key+suffix),as_attachment=True,download_name='registration_overlay.mp4' if suffix=='.mp4' else 'visual_registration.json')

def main():
    app.run(host='127.0.0.1',port=int(os.environ.get('REGISTRATION_PORT','8769')),threaded=True,debug=False)

if __name__=='__main__':main()
