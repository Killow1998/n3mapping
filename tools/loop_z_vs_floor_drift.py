import json,sys,numpy as np
from pathlib import Path
sys.path.insert(0,'/tmp/pb'); sys.setrecursionlimit(10000)
import n3map_pb2
def R(q):
    return np.array([[1-2*(q.qy*q.qy+q.qz*q.qz),2*(q.qx*q.qy-q.qz*q.qw),2*(q.qx*q.qz+q.qy*q.qw)],
                     [2*(q.qx*q.qy+q.qz*q.qw),1-2*(q.qx*q.qx+q.qz*q.qz),2*(q.qy*q.qz-q.qx*q.qw)],
                     [2*(q.qx*q.qz-q.qy*q.qw),2*(q.qy*q.qz+q.qx*q.qw),1-2*(q.qx*q.qx+q.qy*q.qy)]])
base='/home/user/ros_ws/n3mapping_v1_closeout/s9_fixdefault/map/'
m=n3map_pb2.N3Map(); m.ParseFromString(Path(base+'n3map.pbstream').read_bytes())
kfs=sorted(m.keyframes,key=lambda k:k.timestamp)
# 每帧: 用 pose_odom(前端) 拟合脚下地板高度 -> 前端 z 参考
floor={}
for k in kfs:
    if k.cloud.num_points<400: continue
    q=k.pose_odom
    pts=np.asarray(k.cloud.points,dtype=np.float64).reshape(-1,4)[:,:3]
    W=pts@R(q).T; W=W[np.hypot(W[:,0],W[:,1])<8.0]
    if len(W)<400: continue
    lo=np.percentile(W[:,2],2.0); b=W[(W[:,2]>lo-0.10)&(W[:,2]<lo+0.35)]
    if len(b)<400: continue
    floor[k.id]=float(np.median(b[:,2]))+q.tz
rows=[json.loads(l) for l in open(base+'loop_debug.jsonl') if l.strip()]
acc=[r for r in rows if r.get('reject_reason') in ('','None',None) and r.get('loop_information_diag')]
pairs=[]
for r in acc:
    a,b_=r.get('query_id'),r.get('match_id')
    if a in floor and b_ in floor and r.get('icp_correction_match_z') is not None:
        # 地板给出的两帧前端 z 参考之差 = 这条闭环若正确应当修正的量
        pairs.append((r['icp_correction_match_z'], floor[a]-floor[b_], a, b_))
A=np.array([(p[0],p[1]) for p in pairs])
print('可比对的闭环 %d / %d 条'%(len(A),len(acc)))
print()
print('  闭环 z 修正 vs 地板给出的 z 漂移:')
print('    闭环 z 修正   中位 %+7.3f  绝对值中位 %6.3f  p90 %6.3f'%(np.median(A[:,0]),np.median(abs(A[:,0])),np.percentile(abs(A[:,0]),90)))
print('    地板 z 差     中位 %+7.3f  绝对值中位 %6.3f  p90 %6.3f'%(np.median(A[:,1]),np.median(abs(A[:,1])),np.percentile(abs(A[:,1]),90)))
d=A[:,0]-A[:,1]
print('    两者之差      中位 %+7.3f  绝对值中位 %6.3f  p90 %6.3f  最大 %6.3f'%(np.median(d),np.median(abs(d)),np.percentile(abs(d),90),abs(d).max()))
c=np.corrcoef(A[:,0],A[:,1])[0,1]
print('    相关系数 %.3f'%c)
print()
print('  判读: 闭环若正确, z 修正应当追上地板给出的漂移 -> 相关系数接近 1, 差值小')
