import copy
import json
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from sync_demo.core import (
    DEFAULT_CAMERA,
    camera_parameters,
    fit,
    geometry_parameters,
    project_geometry,
    select_action_frames,
)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[900.,0,400.],[0,900.,300.],[0,0,1.]])
        self.D=np.zeros(8)

    def test_geometry_defaults_and_validation(self):
        self.assertEqual(geometry_parameters(),{
            'center_axis':'+Y','opening_axis':'+X','offset_mode':'zero','manual_offset_mm':[0.,0.,0.]})
        with self.assertRaises(ValueError):geometry_parameters({'center_axis':'+Y','opening_axis':'+Y'})
        with self.assertRaises(ValueError):geometry_parameters({'offset_mode':'manual','manual_offset_mm':[100,0,0]})

    def test_projection_zero_manual_and_optimized_offset(self):
        state=np.array([[0,0,0,0,0,0,1,0],[.01,0,0,0,0,0,1,1.]])
        fixed=np.r_[np.zeros(3),[0,0,.5],np.log(.01)]
        uv,cam=project_geometry(fixed,state,np.zeros(3),self.K,self.D,[0,1])
        np.testing.assert_allclose(uv[0,1],uv[0,2])
        self.assertGreater(np.linalg.norm(uv[1,1]-uv[1,2]),0)
        manual={'center_axis':'+Y','opening_axis':'+X','offset_mode':'manual','manual_offset_mm':[10,0,0]}
        muv,_=project_geometry(fixed,state,np.zeros(3),self.K,self.D,[0,1],manual)
        optimized={'center_axis':'+Y','opening_axis':'+X','offset_mode':'optimize'}
        ouv,_=project_geometry(np.r_[fixed[:6],[.01,0,0],fixed[6]],state,np.zeros(3),self.K,self.D,[0,1],optimized)
        np.testing.assert_allclose(muv,ouv,atol=1e-10)
        print('Projection evidence:',json.dumps({'state':state.tolist(),'zero_offset_uv':uv.tolist(),
              'manual_offset_uv':muv.tolist(),'depth_m':cam[:,:,2].tolist()}))

    def test_action_selection(self):
        n=120;a=np.zeros((n,8));a[5:,6]=1
        a[5:60,7]=np.tile([0,0,0,0,0,1,1,1,1,1],6)[:55]
        a[60:,0]=np.linspace(0,.06,60);a[60:,3:7]=Rotation.from_euler('z',np.linspace(0,1.3,60)).as_quat();a[60:,7]=1
        rows=[{f'action_{j}':str(x[j]) for j in range(8)} for x in a]
        ep={'rows':rows,'n':n,'fps':30,'state':np.zeros((n,16))}
        result=select_action_frames(ep);ids=[x['frame'] for x in result['frames']]
        self.assertEqual(len(ids),12);self.assertEqual(len(set(ids)),12);self.assertTrue(all(i>=5 for i in ids))
        self.assertEqual(result['excluded_rows'],5);self.assertTrue(any(i<60 for i in ids));self.assertTrue(any(i>90 for i in ids))
        self.assertEqual(min(a[ids,7]),0);self.assertEqual(max(a[ids,7]),1)
        for i,row in enumerate(rows):
            if i%2:
                for j in range(3,7):row[f'action_{j}']=str(-float(row[f'action_{j}']))
        self.assertEqual(ids,[x['frame'] for x in select_action_frames(ep)['frames']])
        ep['state']+=999
        self.assertEqual(ids,[x['frame'] for x in select_action_frames(ep)['frames']])

    def test_camera_scale(self):
        K,D=camera_parameters(DEFAULT_CAMERA,[960,540]);self.assertAlmostEqual(K[0,0],DEFAULT_CAMERA['intrinsics'][0]/2);self.assertEqual(len(D),8)
        with self.assertRaises(ValueError):camera_parameters(DEFAULT_CAMERA,[960,540],False)

    def synthetic_fit_fixture(self,offset_mode='zero'):
        n=14;t=np.linspace(0,1,n)
        pose=np.c_[.025*np.sin(t*2.1),.02*np.cos(t*1.7),.015*t,
                   Rotation.from_euler('xyz',np.c_[.25*t,-.18*t,.35*t]).as_quat(),.15+.75*t]
        state=np.c_[pose,np.tile([0,0,0,0,0,0,1,0],(n,1))]
        camera={**DEFAULT_CAMERA,'image_size':[800,600],'intrinsics':[900,900,400,300,0,0,0,0,0,0,0,0]}
        geometry={'center_axis':'+Y','opening_axis':'+X','offset_mode':offset_mode,
                  'manual_offset_mm':[3,-2,1] if offset_mode=='manual' else [0,0,0]}
        center=pose[:,:3].mean(0);rv=Rotation.from_euler('xyz',[2.8,.1,-.3]).as_rotvec();tv=[.01,-.005,.32]
        offset=[.003,-.002,.001] if offset_mode=='optimize' else []
        q=np.r_[rv,tv,offset,np.log(.017)]
        uv,_=project_geometry(q,pose,center,self.K,self.D,np.arange(n),geometry)
        annotations=[{'frame':i,'role':'validation' if i in [3,10] else 'fit','points':uv[i].tolist()} for i in range(n)]
        episode={'state':state,'size':[800,600],'n':n,'fingerprint':'synthetic','csv':'synthetic.csv','video':'synthetic.mp4','fps':30}
        return episode,camera,geometry,annotations,uv

    def test_fit_and_validation_share_geometry_without_leakage(self):
        episode,camera,geometry,annotations,target=self.synthetic_fit_fixture('zero')
        result=fit(episode,annotations,camera,geometry=geometry,starts=4)
        self.assertLess(result['fit_rms_px'],1e-3);self.assertLess(result['validation_rms_px'],1e-3)
        self.assertEqual(result['geometry'],geometry);self.assertEqual(len(result['parameter_vector']),7)
        perturbed=copy.deepcopy(annotations)
        for a in perturbed:
            if a['role']=='validation':
                for p in a['points']:p[0]+=40
        other=fit(episode,perturbed,camera,geometry=geometry,starts=4)
        np.testing.assert_allclose(result['parameter_vector'],other['parameter_vector'],atol=1e-9,rtol=0)
        self.assertGreater(other['validation_rms_px'],result['validation_rms_px']+20)
        out=Path(__file__).resolve().parents[1]/'outputs/test_alignment.png';canvas=np.full((600,800,3),32,np.uint8)
        for manual,pred in zip(target[10],np.asarray(result['projected'])[10]):
            cv2.circle(canvas,tuple(np.rint(manual).astype(int)),8,(255,255,255),2,cv2.LINE_AA)
            cv2.drawMarker(canvas,tuple(np.rint(pred).astype(int)),(0,220,255),cv2.MARKER_CROSS,18,2,cv2.LINE_AA)
        cv2.putText(canvas,'White circles: target | Yellow crosses: fitted projection',(20,35),0,.7,(240,240,240),2,cv2.LINE_AA)
        cv2.imwrite(str(out),canvas)
        evidence={'fit_frames':[a['frame'] for a in annotations if a['role']=='fit'],
                  'heldout_frames':[a['frame'] for a in annotations if a['role']=='validation'],
                  'frame_10_target_uv':target[10].tolist(),'frame_10_prediction_uv':result['projected'][10],
                  'fit_rms_px':result['fit_rms_px'],'validation_rms_px':result['validation_rms_px'],
                  'validation_perturbation_px':[40,0],'parameters_unchanged':True,'overlay':str(out)}
        (out.parent/'test_alignment.json').write_text(json.dumps(evidence,indent=2))
        print('Fit/validation alignment evidence:',json.dumps(evidence))

    def test_optimized_offset_adds_only_three_parameters(self):
        episode,camera,geometry,annotations,_=self.synthetic_fit_fixture('optimize')
        result=fit(episode,annotations,camera,geometry=geometry,starts=5)
        self.assertEqual(len(result['parameter_vector']),10)
        self.assertLess(result['fit_rms_px'],.1);self.assertLess(result['validation_rms_px'],.1)
        np.testing.assert_allclose(result['pivot_offset_tool_m'],[.003,-.002,.001],atol=2e-4)

    def test_http_security_and_packaged_assets(self):
        from sync_demo import app
        client=app.app.test_client()
        self.assertEqual(client.post('/api/discover',json={'folder':'/tmp'}).status_code,403)
        self.assertEqual(client.get('/',headers={'Host':'untrusted.example'}).status_code,403)
        page=client.get('/');self.assertEqual(page.status_code,200);self.assertIn(b'Jaw centerline',page.data)


if __name__=='__main__':unittest.main()
