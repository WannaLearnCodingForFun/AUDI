# One processed detection (ml/semantic_memory.AssetEvent)
std_msgs/Header header
string session_id
string type              # new_asset | confirmed | change | duplicate | tracked | rejected | error
string asset_id
string asset_type
int32 track_id
int32[4] bbox            # x1 y1 x2 y2, image pixels
string condition_label
float32 condition_confidence
string identity_stage    # aruco | site_map | geometric | visual_only | new
float32 identity_confidence
float32 change_magnitude
float32 severity_score
string severity_level
bool priority
bool critical
float64 latitude         # ground-projected fix (NaN when unknown)
float64 longitude
float32 location_uncertainty_m
string guidance          # 2.4 operator hint, empty when none
string growth            # 2.3 summary, empty when none
string observation_id
string json              # full event as JSON for consumers that need every field
