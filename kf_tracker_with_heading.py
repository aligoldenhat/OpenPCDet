import json
import numpy as np
from scipy.optimize import linear_sum_assignment


def quaternion_to_yaw(q):
    """
    Convert nuScenes quaternion [w, x, y, z] to yaw angle (heading) in radians.
    Yaw = rotation around Z axis (the one that matters in BEV).
    """
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw


def angle_diff(a, b):
    """
    Shortest angular difference between two angles in radians.
    Result is in [-pi, pi].
    """
    diff = a - b
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff


# SINGLE OBJECT TRACK 
class KalmanTrack:
    count = 0  # global ID counter, reset per scene

    def __init__(self, detection):
        self.id    = KalmanTrack.count
        KalmanTrack.count += 1
        self.hits   = 1
        self.misses = 0
        self.det_class = detection['detection_name']
        self.score     = detection['detection_score']

        # INITIAL STATE from detection 
        px, py, pz = detection['translation']
        vx, vy     = detection['velocity']
        heading    = quaternion_to_yaw(detection['rotation'])

        # x = [px, py, pz, vx, vy, vz, heading]
        self.x = np.array([px, py, pz, vx, vy, 0.0, heading])

        # UNCERTAINTY P 
        # Higher uncertainty for velocity and heading at init
        self.P = np.diag([1., 1., 1.,       # position  — fairly certain
                          10., 10., 10.,     # velocity  — less certain
                          0.5])              # heading   — somewhat certain

        # dt: nuScenes keyframe interval 
        self.dt = 0.5  # seconds

        # MOTION MODEL F 
        # Constant velocity model:
        #   position  += velocity * dt
        #   velocity   = constant
        #   heading    = constant (no angular velocity in this model)
        dt = self.dt
        self.F = np.array([
        #   px  py  pz  vx  vy  vz  hdg
            [1,  0,  0,  dt, 0,  0,  0  ],  # px
            [0,  1,  0,  0,  dt, 0,  0  ],  # py
            [0,  0,  1,  0,  0,  dt, 0  ],  # pz
            [0,  0,  0,  1,  0,  0,  0  ],  # vx
            [0,  0,  0,  0,  1,  0,  0  ],  # vy
            [0,  0,  0,  0,  0,  1,  0  ],  # vz
            [0,  0,  0,  0,  0,  0,  1  ],  # heading
        ])

        # MEASUREMENT MATRIX H 
        # We observe: [px, py, pz, vx, vy, heading] — 6 of the 7 state dims
        # (vz not observed directly)
        self.H = np.array([
        #   px  py  pz  vx  vy  vz  hdg
            [1,  0,  0,  0,  0,  0,  0  ],  # px
            [0,  1,  0,  0,  0,  0,  0  ],  # py
            [0,  0,  1,  0,  0,  0,  0  ],  # pz
            [0,  0,  0,  1,  0,  0,  0  ],  # vx
            [0,  0,  0,  0,  1,  0,  0  ],  # vy
            [0,  0,  0,  0,  0,  0,  1  ],  # heading
        ])

        # PROCESS NOISE Q 
        # How much we expect the world to surprise us between frames
        self.Q = np.diag([0.5,  0.5,  0.5,   # position noise
                          1.0,  1.0,  0.5,   # velocity noise
                          0.1])              # heading noise (slow changes)

        # MEASUREMENT NOISE R 
        # How much we trust the detector output
        self.R = np.diag([0.5,  0.5,  0.5,   # position measurement noise
                          1.5,  1.5,          # velocity measurement noise
                          0.3])              # heading measurement noise

    def predict(self):
        """Predict state forward one timestep using motion model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.misses += 1
        return self.x.copy()

    def update(self, detection):
        """Correct predicted state with a new detection."""
        px, py, pz = detection['translation']
        vx, vy     = detection['velocity']
        heading    = quaternion_to_yaw(detection['rotation'])

        z = np.array([px, py, pz, vx, vy, heading])

        # Residual — note: heading residual needs angle wrapping
        y = z - self.H @ self.x
        y[5] = angle_diff(z[5], (self.H @ self.x)[5])  # wrap heading diff

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        # Keep heading in [-pi, pi]
        self.x[6] = np.arctan2(np.sin(self.x[6]), np.cos(self.x[6]))

        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P

        self.hits  += 1
        self.misses = 0
        self.score  = detection['detection_score']
        self.det_class = detection['detection_name']


# MATCHING
def build_cost_matrix(tracks, detections):
    """
    Cost matrix using Euclidean distance in BEV (x, y only).
    Ignores z — nuScenes is mostly flat and z matching causes issues.
    """
    n, m = len(tracks), len(detections)
    cost = np.zeros((n, m))
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            tx, ty = t.x[0], t.x[1]
            dx, dy = d['translation'][0], d['translation'][1]
            cost[i, j] = np.sqrt((tx - dx)**2 + (ty - dy)**2)
    return cost


def match_detections_to_tracks(tracks, detections, dist_threshold=4.0):
    """
    Hungarian algorithm matching.
    Returns matched pairs, unmatched det indices, unmatched trk indices.
    """
    if not tracks:
        return [], list(range(len(detections))), []
    if not detections:
        return [], [], list(range(len(tracks)))

    cost = build_cost_matrix(tracks, detections)
    row_ind, col_ind = linear_sum_assignment(cost)

    matched, unmatched_dets, unmatched_trks = [], set(), set()

    assigned_trks = set()
    assigned_dets = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= dist_threshold:
            matched.append((r, c))
            assigned_trks.add(r)
            assigned_dets.add(c)
        else:
            unmatched_trks.add(r)
            unmatched_dets.add(c)

    for i in range(len(tracks)):
        if i not in assigned_trks:
            unmatched_trks.add(i)
    for j in range(len(detections)):
        if j not in assigned_dets:
            unmatched_dets.add(j)

    return matched, list(unmatched_dets), list(unmatched_trks)


#  MULTI-OBJECT TRACKER 
class Tracker3D:
    def __init__(self, max_misses=2, min_hits=1):
        self.tracks    = []
        self.max_misses = max_misses
        self.min_hits   = min_hits

    def update(self, detections):
        # 1. Predict all tracks forward
        for t in self.tracks:
            t.predict()

        # 2. Match detections to predicted positions
        matched, unmatched_dets, unmatched_trks = \
            match_detections_to_tracks(self.tracks, detections)

        # 3. Update matched tracks
        for trk_idx, det_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])

        # 4. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self.tracks.append(KalmanTrack(detections[det_idx]))

        # 5. Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        # Return confirmed tracks only
        return [t for t in self.tracks if t.hits >= self.min_hits]


# SCENE LOADING
def get_scene_ordered_tokens(nusc_root):
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes('v1.0-mini', nusc_root)
    scenes = []
    for scene in nusc.scene:
        tokens = []
        token = scene['first_sample_token']
        while token:
            tokens.append(token)
            sample = nusc.get('sample', token)
            token = sample['next']
        scenes.append((scene['name'], tokens))
    return scenes


# MAIN TRACKING LOOP
def run_tracker(results_path, nusc_root, score_threshold=0.3):
    with open(results_path) as f:
        data = json.load(f)

    scenes = get_scene_ordered_tokens(nusc_root)
    all_results = {}

    for scene_name, scene_tokens in scenes:
        val_tokens = [t for t in scene_tokens if t in data['results']]
        if not val_tokens:
            continue

        tracker = Tracker3D(max_misses=2, min_hits=1)
        KalmanTrack.count = 0

        print(f"\n── {scene_name} ({len(val_tokens)} val frames) ──")

        for sample_token in val_tokens:
            detections = data['results'][sample_token]
            dets = [d for d in detections if d['detection_score'] > score_threshold]
            active_tracks = tracker.update(dets)

            all_results[sample_token] = {
                'scene': scene_name,
                'tracks': [
                    {
                        'track_id':  t.id,
                        'class':     t.det_class,
                        'position':  t.x[:3].tolist(),
                        'velocity':  t.x[3:5].tolist(),
                        'heading':   float(t.x[6]),       # ← NEW
                        'heading_deg': float(np.degrees(t.x[6])),  # human readable
                        'score':     t.score,
                        'hits':      t.hits,
                    }
                    for t in active_tracks
                ]
            }

            print(f"  {sample_token[:8]}... | "
                  f"Dets: {len(dets):3d} | Tracks: {len(active_tracks):3d}")

    return all_results


if __name__ == '__main__':
    RESULTS_PATH = (
        '/workspace/output/nuscenes_models/cbgs_voxel0075_res3d_centerpoint'
        '/default/eval/epoch_6648/val/default/final_result/data/results_nusc.json'
    )
    NUSC_ROOT = '/workspace/data/nuscenes/v1.0-mini'

    results = run_tracker(RESULTS_PATH, NUSC_ROOT)

    # Save for use by visualizer or further analysis
    out_path = '/workspace/tracking_results_with_heading.json'
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"\nSaved tracking results to {out_path}")

    # Print sample output
    first_frame = list(results.values())[0]
    print(f"\nFirst frame — {len(first_frame['tracks'])} tracks:")
    for t in first_frame['tracks'][:6]:
        print(f"  ID={t['track_id']:3d} {t['class']:25s} "
              f"pos=({t['position'][0]:.1f}, {t['position'][1]:.1f}) "
              f"vel=({t['velocity'][0]:.1f}, {t['velocity'][1]:.1f}) "
              f"heading={t['heading_deg']:.1f}°  "
              f"hits={t['hits']}")