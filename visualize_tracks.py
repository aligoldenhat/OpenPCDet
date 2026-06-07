"""
visualize_tracks.py

Bird's Eye View (BEV) trajectory visualizer for tracking results.

Reads the JSON output from kf_tracker_with_heading.py and produces:
  1. Full trajectory plot — all tracks + ego vehicle trajectory
  2. Per-class breakdown — separate subplot per object class
  3. Animated GIF — frame-by-frame tracking with ego arrow (optional)

Run:
    python3 visualize_tracks.py
    python3 visualize_tracks.py --scene scene-0103
    python3 visualize_tracks.py --animate
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict


# COLOR MAP
CLASS_COLORS = {
    'car':                  '#4FC3F7',
    'truck':                '#FF8A65',
    'bus':                  '#FFD54F',
    'pedestrian':           '#81C784',
    'motorcycle':           '#CE93D8',
    'bicycle':              '#F48FB1',
    'trailer':              '#80CBC4',
    'construction_vehicle': '#BCAAA4',
    'barrier':              '#EF9A9A',
    'traffic_cone':         '#FFF176',
}
DEFAULT_COLOR  = '#AAAAAA'
EGO_COLOR      = '#FFFFFF'      # ego vehicle: bright white
EGO_PATH_COLOR = '#FF4444'      # ego trajectory: red


def get_color(cls):
    return CLASS_COLORS.get(cls, DEFAULT_COLOR)


def load_results(path):
    with open(path) as f:
        return json.load(f)


def quaternion_to_yaw(q):
    """nuScenes quaternion [w, x, y, z] → yaw in radians."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# EGO POSE LOADING
def load_ego_poses(nusc_root, scene_name):
    """
    For a given scene, return ordered list of:
        {'x': float, 'y': float, 'yaw': float, 'token': str}
    One entry per annotated keyframe (sample), using the LIDAR_TOP pose.

    How nuScenes stores ego pose:
        scene → sample tokens (keyframes at 2 Hz)
            → sample_data for LIDAR_TOP channel
                → ego_pose_token
                    → translation [x, y, z] + rotation [w, x, y, z]
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes('v1.0-mini', nusc_root)

    # Find the scene
    scene = next((s for s in nusc.scene if s['name'] == scene_name), None)
    if scene is None:
        print(f"  Warning: scene {scene_name} not found in nuScenes")
        return []

    ego_poses = []
    token = scene['first_sample_token']

    while token:
        sample = nusc.get('sample', token)

        # Get LIDAR_TOP sample_data — this has the ego pose at keyframe time
        lidar_token = sample['data']['LIDAR_TOP']
        sd = nusc.get('sample_data', lidar_token)
        ep = nusc.get('ego_pose', sd['ego_pose_token'])

        x, y, _ = ep['translation']
        yaw = quaternion_to_yaw(ep['rotation'])

        ego_poses.append({
            'x':     x,
            'y':     y,
            'yaw':   yaw,
            'token': token,
        })

        token = sample['next']  # '' when at last frame

    return ego_poses


# TRACK HISTORY BUILDER
def build_track_histories(results, scene_filter=None):
    track_histories = defaultdict(lambda: {
        'class': None, 'frames': [], 'positions': [],
        'headings': [], 'velocities': [], 'hits': [],
    })
    frame_list = []

    for frame_idx, (token, frame_data) in enumerate(results.items()):
        scene = frame_data.get('scene', '')
        if scene_filter and scene != scene_filter:
            continue
        frame_list.append(token)
        for t in frame_data['tracks']:
            tid = t['track_id']
            track_histories[tid]['class']     = t['class']
            track_histories[tid]['frames'].append(frame_idx)
            track_histories[tid]['positions'].append(t['position'][:2])
            track_histories[tid]['headings'].append(t.get('heading', 0.0))
            track_histories[tid]['velocities'].append(t['velocity'])
            track_histories[tid]['hits'].append(t['hits'])

    return dict(track_histories), frame_list


# EGO DRAWING HELPERS
def draw_ego_trajectory(ax, ego_poses, final_only=False):
    """
    Draw the ego vehicle's path and a car-shaped arrow at its final position.

    final_only=True  → only draw the current frame arrow (used in animation)
    final_only=False → draw full path + final arrow (used in static plots)
    """
    if not ego_poses:
        return

    xs = [p['x'] for p in ego_poses]
    ys = [p['y'] for p in ego_poses]

    if not final_only:
        # Dashed red trajectory line
        ax.plot(xs, ys,
                color=EGO_PATH_COLOR, linewidth=2.0,
                linestyle='--', alpha=0.7,
                zorder=8, label='ego path')

        # Dot at each keyframe position
        ax.scatter(xs[:-1], ys[:-1],
                   c=EGO_PATH_COLOR, s=25, alpha=0.4, zorder=9)

        # Start marker
        ax.scatter(xs[0], ys[0],
                   c=EGO_PATH_COLOR, s=80, marker='s',
                   zorder=10, label='ego start')

    # Final (or current) position: car-shaped arrow
    ex, ey   = ego_poses[-1]['x'], ego_poses[-1]['y']
    yaw      = ego_poses[-1]['yaw']
    car_len  = 4.5   # metres — approximate car length
    car_w    = 2.0

    # Arrow body: filled rectangle rotated to heading
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    # Four corners of the car rectangle in local frame
    # (front-right, front-left, back-left, back-right)
    corners_local = np.array([
        [ car_len / 2,  car_w / 2],
        [ car_len / 2, -car_w / 2],
        [-car_len / 2, -car_w / 2],
        [-car_len / 2,  car_w / 2],
    ])

    # Rotate and translate to world frame
    R = np.array([[cos_y, -sin_y],
                  [sin_y,  cos_y]])
    corners_world = (R @ corners_local.T).T + np.array([ex, ey])
    car_patch = plt.Polygon(corners_world, closed=True,
                            facecolor=EGO_COLOR, edgecolor='#FF4444',
                            linewidth=1.5, alpha=0.9, zorder=11)
    ax.add_patch(car_patch)

    # Heading arrow extending from car front
    arrow_len = car_len * 0.8
    dx = arrow_len * cos_y
    dy = arrow_len * sin_y
    ax.annotate('',
                xy=(ex + dx, ey + dy),
                xytext=(ex, ey),
                arrowprops=dict(arrowstyle='->', color='#FF4444',
                                lw=2.5, mutation_scale=18),
                zorder=12)

    # Label
    ax.text(ex + 1.5, ey + 1.5, 'EGO',
            color=EGO_COLOR, fontsize=8, fontweight='bold',
            zorder=13, alpha=0.9)


# PLOT 1: FULL TRAJECTORY
def plot_trajectories(track_histories, ego_poses, scene_name, out_path):
    fig, ax = plt.subplots(figsize=(14, 14), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    ax.set_title(f'BEV Track Trajectories — {scene_name}',
                 color='white', fontsize=16, pad=15)
    ax.set_xlabel('X (m)', color='#888888')
    ax.set_ylabel('Y (m)', color='#888888')
    ax.tick_params(colors='#555555')
    for spine in ax.spines.values():
        spine.set_edgecolor('#222222')
    ax.grid(True, color='#1A2030', linewidth=0.5, alpha=0.7)

    min_track_len = 3

    for tid, history in track_histories.items():
        positions = np.array(history['positions'])
        cls       = history['class']
        color     = get_color(cls)
        if len(positions) < min_track_len:
            continue

        n = len(positions)
        segments = [[positions[i], positions[i + 1]] for i in range(n - 1)]
        lc = LineCollection(segments, colors=[color], linewidths=1.2, alpha=0.6)
        ax.add_collection(lc)

        ax.scatter(positions[0, 0], positions[0, 1],
                   c=color, s=20, alpha=0.4, zorder=3)
        ax.scatter(positions[-1, 0], positions[-1, 1],
                   c=color, s=50, alpha=0.9, zorder=4)

        heading   = history['headings'][-1]
        arrow_len = 2.0
        ax.annotate('',
                    xy=(positions[-1, 0] + arrow_len * np.cos(heading),
                        positions[-1, 1] + arrow_len * np.sin(heading)),
                    xytext=(positions[-1, 0], positions[-1, 1]),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.2, alpha=0.8))

    # Ego vehicle
    draw_ego_trajectory(ax, ego_poses, final_only=False)

    # Legend
    legend_handles = [
        mpatches.Patch(color=get_color(cls), label=cls)
        for cls in CLASS_COLORS
    ] + [
        mpatches.Patch(color=EGO_COLOR,      label='ego vehicle'),
        mpatches.Patch(color=EGO_PATH_COLOR, label='ego path'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              facecolor='#161B22', edgecolor='#30363D',
              labelcolor='white', fontsize=9)

    ax.set_aspect('equal')
    ax.autoscale()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print(f"  Saved: {out_path}")


# PLOT 2: PER-CLASS BREAKDOWN
def plot_per_class(track_histories, ego_poses, scene_name, out_path):
    classes_present = sorted(set(
        h['class'] for h in track_histories.values()
        if len(h['positions']) >= 3
    ))
    if not classes_present:
        print("  No classes with enough tracks to plot.")
        return

    n_cols = 3
    n_rows = (len(classes_present) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 6 * n_rows),
                             facecolor='#0D1117')
    fig.suptitle(f'Per-Class Trajectories — {scene_name}',
                 color='white', fontsize=16, y=1.01)

    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax_idx, cls in enumerate(classes_present):
        ax = axes_flat[ax_idx]
        ax.set_facecolor('#0D1117')
        ax.set_title(cls, color=get_color(cls), fontsize=12)
        ax.tick_params(colors='#555555')
        ax.grid(True, color='#1A2030', linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#222222')

        color = get_color(cls)
        track_count = 0

        for tid, history in track_histories.items():
            if history['class'] != cls:
                continue
            positions = np.array(history['positions'])
            if len(positions) < 3:
                continue
            track_count += 1
            ax.plot(positions[:, 0], positions[:, 1],
                    color=color, linewidth=1.0, alpha=0.6)
            ax.scatter(positions[-1, 0], positions[-1, 1],
                       c=color, s=40, zorder=4, alpha=0.9)
            heading   = history['headings'][-1]
            arrow_len = 1.5
            ax.annotate('',
                        xy=(positions[-1, 0] + arrow_len * np.cos(heading),
                            positions[-1, 1] + arrow_len * np.sin(heading)),
                        xytext=(positions[-1, 0], positions[-1, 1]),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.0, alpha=0.7))

        # Draw ego on every subplot for spatial reference
        draw_ego_trajectory(ax, ego_poses, final_only=False)

        ax.set_aspect('equal')
        ax.autoscale()
        ax.set_xlabel(f'{track_count} tracks', color='#555555', fontsize=9)

    for i in range(len(classes_present), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print(f"  Saved: {out_path}")


# PLOT 3: ANIMATED GIF
def plot_animated(results, scene_filter, ego_poses, out_path):
    try:
        import imageio
    except ImportError:
        print("  Skipping animation — run: pip3 install imageio")
        return

    import tempfile, os

    frames_data = [
        (token, fd) for token, fd in results.items()
        if not scene_filter or fd.get('scene') == scene_filter
    ]

    # Build a token → ego_pose lookup for per-frame ego position
    ego_by_token = {p['token']: p for p in ego_poses}

    # Global bounds (include ego path in bounds)
    all_x = [p['x'] for p in ego_poses]
    all_y = [p['y'] for p in ego_poses]
    for _, fd in frames_data:
        for t in fd['tracks']:
            all_x.append(t['position'][0])
            all_y.append(t['position'][1])

    if not all_x:
        return

    pad   = 15
    x_min, x_max = min(all_x) - pad, max(all_x) + pad
    y_min, y_max = min(all_y) - pad, max(all_y) + pad

    # Build ego trajectory up to each frame for the "trail" effect
    ego_tokens_ordered = [p['token'] for p in ego_poses]

    tmpdir = tempfile.mkdtemp()
    frame_paths = []

    for frame_idx, (token, fd) in enumerate(frames_data):
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0D1117')
        ax.set_facecolor('#0D1117')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(
            f'{scene_filter or "scene"}  —  frame {frame_idx + 1}/{len(frames_data)}',
            color='white', fontsize=13)
        ax.grid(True, color='#1A2030', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#222222')
        ax.tick_params(colors='#444444')

        # Object tracks
        for t in fd['tracks']:
            px, py  = t['position'][:2]
            heading = t.get('heading', 0.0)
            cls     = t['class']
            color   = get_color(cls)

            radius = 1.0 + min(t['hits'], 10) * 0.05
            circle = plt.Circle((px, py), radius, color=color,
                                 alpha=0.5, fill=True)
            ax.add_patch(circle)

            arrow_len = 2.5
            ax.annotate('',
                        xy=(px + arrow_len * np.cos(heading),
                            py + arrow_len * np.sin(heading)),
                        xytext=(px, py),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

            ax.text(px + 0.5, py + 0.5, str(t['track_id']),
                    color=color, fontsize=6, alpha=0.8)

        # Ego: show trail up to current frame
        # Find how far along the ego_poses list we are
        if token in ego_by_token:
            try:
                ego_idx = ego_tokens_ordered.index(token)
            except ValueError:
                ego_idx = len(ego_poses) - 1

            ego_so_far = ego_poses[:ego_idx + 1]
            draw_ego_trajectory(ax, ego_so_far, final_only=False)

        frame_path = os.path.join(tmpdir, f'frame_{frame_idx:04d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight',
                    facecolor='#0D1117')
        plt.close()
        frame_paths.append(frame_path)

    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(out_path, images, fps=4)
    print(f"  Saved: {out_path}")

    for p in frame_paths:
        os.remove(p)


# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results',   default='/workspace/tracking_results_with_heading.json')
    parser.add_argument('--nusc_root', default='/workspace/data/nuscenes/v1.0-mini')
    parser.add_argument('--scene',     default=None, help='e.g. scene-0103')
    parser.add_argument('--outdir',    default='/workspace/vizualization_output')
    parser.add_argument('--animate',   action='store_true')
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading results from {args.results}")
    results = load_results(args.results)

    scenes_present = sorted(set(
        fd.get('scene', 'unknown') for fd in results.values()
    ))
    print(f"Scenes in results: {scenes_present}")

    scenes_to_plot = [args.scene] if args.scene else scenes_present

    for scene in scenes_to_plot:
        print(f"\nPlotting {scene}...")

        # Load ego poses from nuScenes API
        print(f"  Loading ego poses...")
        ego_poses = load_ego_poses(args.nusc_root, scene)
        print(f"  {len(ego_poses)} ego keyframes loaded")

        track_histories, frame_list = build_track_histories(results, scene)
        print(f"  {len(track_histories)} unique tracks, {len(frame_list)} frames")

        plot_trajectories(
            track_histories, ego_poses, scene,
            out_path=f"{args.outdir}/{scene}_trajectories.png"
        )
        plot_per_class(
            track_histories, ego_poses, scene,
            out_path=f"{args.outdir}/{scene}_per_class.png"
        )
        if args.animate:
            plot_animated(
                results, scene, ego_poses,
                out_path=f"{args.outdir}/{scene}_animation.gif"
            )

    print(f"\nAll outputs saved to {args.outdir}/")


if __name__ == '__main__':
    main()