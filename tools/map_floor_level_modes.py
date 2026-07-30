import sys,numpy as np
from pathlib import Path
sys.path.insert(0,'/tmp/pb'); sys.setrecursionlimit(10000)
import n3map_pb2
def R(q):
    return np.array([[1-2*(q.qy*q.qy+q.qz*q.qz),2*(q.qx*q.qy-q.qz*q.qw),2*(q.qx*q.qz+q.qy*q.qw)],
                     [2*(q.qx*q.qy+q.qz*q.qw),1-2*(q.qx*q.qx+q.qz*q.qz),2*(q.qy*q.qz-q.qx*q.qw)],
                     [2*(q.qx*q.qz-q.qy*q.qw),2*(q.qy*q.qz+q.qx*q.qw),1-2*(q.qx*q.qx+q.qy*q.qy)]])
for tag,p in (('b22 S5d','x_b22_s5d'),('b22 S9','x_b22_s9')):
    m=n3map_pb2.N3Map(); m.ParseFromString(Path('/home/user/ros_ws/n3mapping_v1_closeout/%s/map/n3map.pbstream'%p).read_bytes())
    z=[]
    for k in sorted(m.keyframes,key=lambda k:k.timestamp):
        if k.cloud.num_points<400: continue
        q=k.pose_optimized
        pts=np.asarray(k.cloud.points,dtype=np.float64).reshape(-1,4)[:,:3]
        W=pts@R(q).T; W=W[np.hypot(W[:,0],W[:,1])<8.0]
        if len(W)<400: continue
        lo=np.percentile(W[:,2],2.0)
        b=W[(W[:,2]>lo-0.10)&(W[:,2]<lo+0.35)]
        if len(b)<400: continue
        z.append(float(np.median(b[:,2]))+q.tz)
    z=np.array(z)
    h,e=np.histogram(z,bins=40)
    peaks=[(h[i],(e[i]+e[i+1])/2) for i in range(len(h)) if h[i]>0.15*h.max()]
    lows=[p for c,p in peaks if p<np.median(z)]; his=[p for c,p in peaks if p>=np.median(z)]
    print('%-9s %d 帧  地板 z: 最低 %.2f 最高 %.2f  跨度 %.2f'%(tag,len(z),z.min(),z.max(),z.ptp()))
    if lows and his:
        print('          两簇中心 %.2f / %.2f  ->  层间距 %.2f m'%(np.median(lows),np.median(his),np.median(his)-np.median(lows)))
