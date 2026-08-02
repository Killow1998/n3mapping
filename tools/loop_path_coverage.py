import sys, numpy as np
from pathlib import Path
sys.path.insert(0,"/tmp/pb"); sys.setrecursionlimit(10000)
import n3map_pb2
m=n3map_pb2.N3Map(); m.ParseFromString(Path(sys.argv[1]).read_bytes())
kfs=sorted(m.keyframes,key=lambda k:k.timestamp)
idx={k.id:i for i,k in enumerate(kfs)}
P=np.array([[k.pose_optimized.tx,k.pose_optimized.ty,k.pose_optimized.tz] for k in kfs])
path=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))])
loops=[e for e in m.edges if e.type==n3map_pb2.EdgeProto.LOOP]
print("关键帧 %d  总路程 %.1f m  闭环 %d 条"%(len(kfs),path[-1],len(loops)))
spans=[]
for e in loops:
    if e.from_id in idx and e.to_id in idx:
        a,b=path[idx[e.from_id]],path[idx[e.to_id]]
        spans.append((min(a,b),max(a,b)))
spans.sort()
print()
print("  闭环覆盖(按较早端的路程分箱):")
for lo,hi in [(0,25),(25,50),(50,75),(75,100),(100,150),(150,200),(200,300)]:
    n=sum(1 for a,b in spans if lo<=a<hi)
    cross=sum(1 for a,b in spans if a<lo and b>=lo)   # 跨过该段起点的
    print("   %4d-%-4d  起于此段 %2d 条   跨越此段起点 %2d 条"%(lo,hi,n,cross))
print()
print("  路程 90-115 m 附近的闭环:")
near=[(a,b) for a,b in spans if (90<=a<=115) or (90<=b<=115)]
for a,b in near[:12]: print("    %.1f m <-> %.1f m   (跨 %.1f m)"%(a,b,b-a))
print("    共 %d 条"%len(near))
