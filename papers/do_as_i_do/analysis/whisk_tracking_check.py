"""레포에 포함된 whisking 리타겟팅 결과의 물체 추적 오차 계산 (CPU). 실행: python3 analysis/whisk_tracking_check.py"""
import sys
from pathlib import Path
import numpy as np
from numpy.linalg import norm
base = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "repo/retargeting/outputs/sharpa/right/whisking/0")
k=np.load(base+'/trajectory_kinematic.npz'); m=np.load(base+'/trajectory_mjwp.npz')
q=m['qpos'].reshape(-1,35); t=m['time'].reshape(-1)
ref=k['qpos']; f=1/0.0333; tr=np.arange(len(ref))/f
tt=t-3.0; sel=(tt>=0)&(tt<=tr[-1])
r=np.stack([np.interp(tt[sel],tr,ref[:,j]) for j in range(35)],1); qs=q[sel]
pe=norm(qs[:,28:31]-r[:,28:31],axis=1)
def qmul(a,b):
  w1,x1,y1,z1=a.T; w2,x2,y2,z2=b.T
  return np.stack([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2],1)
qa=qs[:,31:35]/norm(qs[:,31:35],axis=1,keepdims=True); qb=r[:,31:35]/norm(r[:,31:35],axis=1,keepdims=True)
d=qmul(qa,qb*np.array([1,-1,-1,-1])); ang=2*np.arccos(np.clip(np.abs(d[:,0]),0,1))
print('frames %d (%.2fs)'%(sel.sum(),tr[-1]))
print('obj pos err mean %.4f m max %.4f'%(pe.mean(),pe.max()))
print('obj rot err mean %.4f rad max %.4f'%(ang.mean(),ang.max()))
print('success(Epos<0.1 & Erot<0.5):', pe.mean()<0.1 and ang.mean()<0.5)
print('ref obj z range %.3f..%.3f, sim %.3f..%.3f'%(r[:,30].min(),r[:,30].max(),qs[:,30].min(),qs[:,30].max()))
print('opt iterations per ctrl step:',m['opt_steps'].ravel().tolist())
