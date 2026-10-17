"""Build direct-callback input from the explicitly cost-only saved-map replay.
Uses the historical cloud's 48-byte XYZINormal layout; unused fields are zero.
This sqlite fixture is not presented as an independently recorded ROS bag.
"""
import csv,sqlite3,sys
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore,Stores
manifest=Path(sys.argv[1]); output=Path(sys.argv[2])
assert not output.exists()
cloud=odom=None
with AnyReader([Path(sys.argv[3])]) as reader:
 for c,t,b in reader.messages():
  if c.topic=='/Odometry' and odom is None:odom=reader.deserialize(b,c.msgtype)
  if c.topic=='/cloud_registered_body' and cloud is None:cloud=reader.deserialize(b,c.msgtype)
  if cloud is not None and odom is not None:break
store=get_typestore(Stores.ROS2_HUMBLE)
db=sqlite3.connect(output)
db.executescript('CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT); CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER,timestamp INTEGER,data BLOB); CREATE INDEX timestamp_idx ON messages(timestamp);')
db.executemany('INSERT INTO topics VALUES (?,?)',[(1,'/Odometry'),(2,'/cloud_registered_body')])
for f in csv.DictReader(manifest.open()):
 stamp=int(f['stamp_ns']);sec,nsec=divmod(stamp,10**9)
 cloud.header.stamp.sec=odom.header.stamp.sec=sec
 cloud.header.stamp.nanosec=odom.header.stamp.nanosec=nsec
 pose=odom.pose.pose
 pose.position.x,pose.position.y,pose.position.z=[float(f[k]) for k in ('tx','ty','tz')]
 pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=[float(f[k]) for k in ('qx','qy','qz','qw')]
 content=(manifest.parent/f['pcd_path']).read_bytes().split(b'DATA binary\n',1)[1]
 xyzi=np.frombuffer(content,dtype='<f4').reshape(-1,4)
 points=np.zeros((len(xyzi),12),dtype='<f4');points[:,:3]=xyzi[:,:3];points[:,3]=1;points[:,8]=xyzi[:,3]
 cloud.width=len(points);cloud.height=1;cloud.row_step=cloud.point_step*cloud.width
 cloud.data=points.view(np.uint8).reshape(-1)
 db.execute('INSERT INTO messages(topic_id,timestamp,data) VALUES (?,?,?)',(1,stamp,store.serialize_cdr(odom,odom.__msgtype__)))
 db.execute('INSERT INTO messages(topic_id,timestamp,data) VALUES (?,?,?)',(2,stamp+500000,store.serialize_cdr(cloud,cloud.__msgtype__)))
db.commit();db.close()
print(output)
