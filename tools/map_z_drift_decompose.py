import sys, numpy as np
from pathlib import Path
sys.path.insert(0,'/tmp/pb'); sys.setrecursionlimit(10000)
import n3map_pb2
def R(q):
    return np.array([[1-2*(q.qy*q.qy+q.qz*q.qz),2*(q.qx*q.qy-q.qz*q.qw),2*(q.qx*q.qz+q.qy*q.qw)],
                     [2*(q.qx*q.qy+q.qz*q.qw),1-2*(q.qx*q.qx+q.qz*q.qz),2*(q.qy*q.qz-q.qx*q.qw)],
                     [2*(q.qx*q.qz-q.qy*q.qw),2*(q.qy*q.qz+q.qx*q.qw),1-2*(q.qx*q.qx+q.qy*q.qy)]])
def floors(path_pb, pose='odom'):
    m=n3map_pb2.N3Map(); m.ParseFromString(Path(path_pb).read_bytes())
    kfs=sorted(m.keyframes,key=lambda k:k.timestamp)
    out=[]
    P=np.array([[ (k.pose_odom if pose=='odom' else k.pose_optimized).tx,
                  (k.pose_odom if pose=='odom' else k.pose_optimized).ty,
                  (k.pose_odom if pose=='odom' else k.pose_optimized).tz] for k in kfs])
    pl=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))])
    for i,k in enumerate(kfs):
        q=k.pose_odom if pose=='odom' else k.pose_optimized
        if k.cloud.num_points<400: continue
        pts=np.asarray(k.cloud.points,dtype=np.float64).reshape(-1,4)[:,:3]
        W=pts@R(q).T
        r=np.hypot(W[:,0],W[:,1]); W=W[r<8.0]
        if len(W)<400: continue
        zlo=np.percentile(W[:,2],2.0)
        band=W[(W[:,2]>zlo-0.10)&(W[:,2]<zlo+0.35)]
        if len(band)<400: continue
        out.append((pl[i],q.tx,q.ty,float(np.median(band[:,2]))+q.tz))
    return np.array(out)
A=floors(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else 'odom')
pl,x,y,z=A[:,0],A[:,1],A[:,2],A[:,3]
M=np.column_stack([x,y,np.ones_like(x)])
coef,*_=np.linalg.lstsq(M,z,rcond=None)
res=z-M@coef
tilt=np.degrees(np.arctan(np.hypot(coef[0],coef[1])))
print('可用 %d 帧, 路程 %.1f m'%(len(A),pl[-1]))
print('固定平面拟合: 倾角 %.2f deg  (对重访一致性不可见)'%tilt)
print('总 z 散布 %.3f m  ->  去掉固定倾斜后残差散布 %.3f m'%(z.std(),res.std()))
print()
print('  残差(随时间变化的部分,才是重访误差的来源) 按路程:')
print('   路程段     帧数    中位      标准差     范围')
for lo,hi in [(0,10),(10,25),(25,50),(50,75),(75,100),(100,150),(150,200),(200,300),(300,1e9)]:
    s=res[(pl>=lo)&(pl<hi)]
    if len(s)<3: continue
    print('  %4d-%-5s %5d  %+8.3f  %8.3f  %8.3f'%(lo,('%d'%hi if hi<1e9 else '+'),
          len(s),np.median(s),s.std(),s.max()-s.min()))
