uint8 SEARCHING=0
uint8 REGION_HYPOTHESIS=1
uint8 FULL_6DOF_LOCKED=2
uint8 DEGRADED_TRACKING=3

uint8 POSE_SOURCE_NONE=0
uint8 POSE_SOURCE_ODOM_PREDICTED=1
uint8 POSE_SOURCE_GEOMETRICALLY_CORRECTED=2

std_msgs/Header header
uint8 state
uint8 pose_source
geometry_msgs/Pose pose
uint64 lock_epoch
int64 seed_keyframe_id
int64 support_keyframe_id
string decision
