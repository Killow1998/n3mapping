import sys, numpy as np
from pathlib import Path
sys.path.insert(0,'/tmp/pb'); sys.setrecursionlimit(10000)
import n3map_pb2
XY=1.0; TGAP=30.0
for tag,path in (('S5d','/home/user/ros_ws/n3mapping_v1_closeout/s5d_segfix_map/n3map.pbstream'),
                 ('S6a','/home/user/ros_ws/n3mapping_v1_closeout/s6a_sigma/map/n3map.pbstream')):
    m=n3map_pb2.N3Map(); m.ParseFromString(Path(path).read_bytes())
    kfs=sorted(m.keyframes,key=lambda k:k.timestamp)
    P=np.array([[k.pose_optimized.tx,k.pose_optimized.ty,k.pose_optimized.tz] for k in kfs])
    t=np.array([k.timestamp for k in kfs])
    path_len=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))])
    pairs=[]
    for i in range(len(P)):
        for j in range(i+1,len(P)):
            if t[j]-t[i]<TGAP: continue
            if np.linalg.norm(P[j,:2]-P[i,:2])>XY: continue
            pairs.append((abs(P[j,2]-P[i,2]),min(path_len[i],path_len[j]),max(path_len[i],path_len[j])))
    A=np.array(sorted(pairs,reverse=True))
    print('##### %s   %d 对  总路程 %.1f m'%(tag,len(A),path_len[-1]))
    print('  按「较早那一端的路程位置」分箱:')
    print('   路程段      对数    |dz| 中位   p90     最大')
    for lo in (0,25,50,75,100,150,200):
        hi={0:25,25:50,50:75,75:100,100:150,150:200,200:1e9}[lo]
        s=A[(A[:,1]>=lo)&(A[:,1]<hi)]
        if not len(s): continue
        print('  %4d-%-4s  %5d   %8.3f %8.3f %8.3f'%(lo,('%d'%hi if hi<1e9 else '+'),len(s),
              np.median(s[:,0]),np.percentile(s[:,0],90),s[:,0].max()))
    print('  最差 6 对: '+', '.join('%.2fm@%.0f/%.0fm'%(a,b,c) for a,b,c in A[:6]))
    print()
