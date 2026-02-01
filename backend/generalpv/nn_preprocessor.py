"""
Neural Network Preprocessor for Expected Threat Model.
Extracts rich features from the chained StatsBomb dataset.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

# Pitch dimensions (StatsBomb coordinates)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
GOAL_Y = 40.0
GOAL_WIDTH = 8.0  # Approximate goal width in StatsBomb units

# Penalty box coordinates
BOX_X_MIN = 102.0  # 18 yards from goal line
BOX_Y_MIN = 18.0
BOX_Y_MAX = 62.0


def normalize_x(x: float) -> float:
    """Normalize x coordinate to [0, 1]."""
    return x / PITCH_LENGTH if pd.notna(x) else 0.0


def normalize_y(y: float) -> float:
    """Normalize y coordinate to [0, 1]."""
    return y / PITCH_WIDTH if pd.notna(y) else 0.0


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def angle_to_point(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """Calculate angle in radians from one point to another."""
    return math.atan2(to_y - from_y, to_x - from_x)


def is_in_box(x: float, y: float) -> bool:
    """Check if a point is in the penalty box."""
    return x >= BOX_X_MIN and BOX_Y_MIN <= y <= BOX_Y_MAX


def calculate_shooting_lane_blocked(
    ball_x: float, ball_y: float,
    opponents: List[Dict[str, float]]
) -> float:
    """
    Calculate what percentage of the goal mouth is blocked by defenders.
    Uses ray casting from ball to goal posts.
    """
    if not opponents or ball_x >= GOAL_X:
        return 0.0

    goal_top = GOAL_Y + GOAL_WIDTH / 2
    goal_bottom = GOAL_Y - GOAL_WIDTH / 2

    # Discretize goal mouth into segments
    n_segments = 20
    blocked_segments = 0

    for i in range(n_segments):
        target_y = goal_bottom + \
            (goal_top - goal_bottom) * (i + 0.5) / n_segments

        # Check if any opponent blocks this ray
        for opp in opponents:
            if opp['x'] <= ball_x:
                continue  # Opponent behind ball

            # Check if opponent is close to the line from ball to target
            # Using point-to-line distance
            dx = GOAL_X - ball_x
            dy = target_y - ball_y
            line_len = math.sqrt(dx**2 + dy**2)

            if line_len == 0:
                continue

            # Distance from opponent to the line
            t = max(0, min(1, ((opp['x'] - ball_x) * dx +
                    (opp['y'] - ball_y) * dy) / (line_len ** 2)))
            closest_x = ball_x + t * dx
            closest_y = ball_y + t * dy

            dist_to_line = euclidean_distance(
                opp['x'], opp['y'], closest_x, closest_y)

            # If opponent within ~1.5m of the line, consider it blocked
            if dist_to_line < 1.5:
                blocked_segments += 1
                break

    return blocked_segments / n_segments


def calculate_passing_options(
    ball_x: float, ball_y: float,
    teammates: List[Dict[str, float]],
    opponents: List[Dict[str, float]]
) -> Tuple[int, float]:
    """
    Calculate number of viable passing options and best passing lane quality.
    A pass is "viable" if there's a clear lane without defenders.
    """
    if not teammates:
        return 0, 0.0

    viable_passes = 0
    best_lane_quality = 0.0

    for tm in teammates:
        # Skip teammates behind the ball (backward passes less valuable for xT)
        if tm['x'] < ball_x - 10:
            continue

        # Calculate pass distance
        pass_dist = euclidean_distance(ball_x, ball_y, tm['x'], tm['y'])
        if pass_dist < 3 or pass_dist > 40:  # Too close or too far
            continue

        # Check if lane is clear
        lane_blocked = False
        min_defender_dist = float('inf')

        for opp in opponents:
            # Point-to-line distance from opponent to pass line
            dx = tm['x'] - ball_x
            dy = tm['y'] - ball_y
            line_len_sq = dx**2 + dy**2

            if line_len_sq == 0:
                continue

            t = max(
                0, min(1, ((opp['x'] - ball_x) * dx + (opp['y'] - ball_y) * dy) / line_len_sq))
            closest_x = ball_x + t * dx
            closest_y = ball_y + t * dy

            dist_to_line = euclidean_distance(
                opp['x'], opp['y'], closest_x, closest_y)
            min_defender_dist = min(min_defender_dist, dist_to_line)

            if dist_to_line < 2.0:  # Defender within 2m of pass line
                lane_blocked = True
                break

        if not lane_blocked:
            viable_passes += 1
            # Lane quality based on position gain and clearance
            position_gain = (tm['x'] - ball_x) / \
                PITCH_LENGTH  # Forward progress
            # How clear the lane is
            clearance = min(min_defender_dist / 10.0, 1.0)
            lane_quality = 0.5 * position_gain + 0.5 * clearance
            best_lane_quality = max(best_lane_quality, lane_quality)

    return viable_passes, best_lane_quality


def calculate_space_control(
    ball_x: float, ball_y: float,
    teammates: List[Dict[str, float]],
    opponents: List[Dict[str, float]]
) -> float:
    """
    Calculate attacking team's control of dangerous space near the ball.
    Uses a simplified Voronoi-like approach.
    """
    # Sample points in a grid around the ball (forward-biased)
    control_score = 0.0
    n_points = 0

    for dx in range(-10, 25, 5):  # -10 to +20 meters
        for dy in range(-15, 16, 5):  # -15 to +15 meters
            sample_x = ball_x + dx
            sample_y = ball_y + dy

            # Skip points outside pitch
            if sample_x < 0 or sample_x > PITCH_LENGTH or sample_y < 0 or sample_y > PITCH_WIDTH:
                continue

            n_points += 1

            # Find nearest teammate and opponent
            nearest_tm_dist = float('inf')
            nearest_opp_dist = float('inf')

            for tm in teammates:
                d = euclidean_distance(sample_x, sample_y, tm['x'], tm['y'])
                nearest_tm_dist = min(nearest_tm_dist, d)

            for opp in opponents:
                d = euclidean_distance(sample_x, sample_y, opp['x'], opp['y'])
                nearest_opp_dist = min(nearest_opp_dist, d)

            # Point controlled by team if nearest teammate is closer
            if nearest_tm_dist < nearest_opp_dist:
                # Weight by position value (closer to goal = more valuable)
                position_value = sample_x / PITCH_LENGTH
                control_score += position_value

    return control_score / max(n_points, 1)


def calculate_defensive_line(opponents: List[Dict[str, float]]) -> float:
    """Calculate the x-position of the defensive line (2nd-to-last defender)."""
    if len(opponents) < 2:
        return GOAL_X

    x_positions = sorted([opp['x'] for opp in opponents], reverse=True)
    # 2nd highest x is the defensive line (ignore keeper)
    return x_positions[1] if len(x_positions) > 1 else x_positions[0]


def extract_ball_features(row: pd.Series) -> np.ndarray:
    """
    Extract features for the ball position.
    Returns: [8] dimensional vector
    """
    ball_x = row['start_x'] if pd.notna(row['start_x']) else 60.0
    ball_y = row['start_y'] if pd.notna(row['start_y']) else 40.0

    dist_to_goal = euclidean_distance(ball_x, ball_y, GOAL_X, GOAL_Y)
    angle_to_goal = angle_to_point(ball_x, ball_y, GOAL_X, GOAL_Y)

    features = np.array([
        normalize_x(ball_x),           # Normalized x
        normalize_y(ball_y),           # Normalized y
        dist_to_goal / PITCH_LENGTH,   # Normalized distance to goal
        angle_to_goal / math.pi,       # Normalized angle [-1, 1]
        1.0 if is_in_box(ball_x, ball_y) else 0.0,  # In penalty box
        # Progress (same as norm_x but semantically different)
        ball_x / PITCH_LENGTH,
        # Centrality (0 = center, 1 = edge)
        abs(ball_y - GOAL_Y) / (PITCH_WIDTH / 2),
        1.0 if ball_x > 102 else 0.0,  # In final third danger zone
    ], dtype=np.float32)

    return features


def extract_player_features(
    player_x: float,
    player_y: float,
    ball_x: float,
    ball_y: float,
    is_teammate: bool,
    is_keeper: bool
) -> np.ndarray:
    """
    Extract features for a single player.
    Returns: [10] dimensional vector
    """
    # Relative position to ball
    rel_x = (player_x - ball_x) / PITCH_LENGTH
    rel_y = (player_y - ball_y) / PITCH_WIDTH

    dist_to_ball = euclidean_distance(player_x, player_y, ball_x, ball_y)
    dist_to_goal = euclidean_distance(player_x, player_y, GOAL_X, GOAL_Y)
    angle_from_ball = angle_to_point(ball_x, ball_y, player_x, player_y)

    # Is player between ball and goal (potential blocker)?
    blocks_shot = 0.0
    if player_x > ball_x:  # Player ahead of ball
        # Check if player is in the "cone" from ball to goal
        angle_to_player = angle_to_point(ball_x, ball_y, player_x, player_y)
        angle_to_goal_center = angle_to_point(ball_x, ball_y, GOAL_X, GOAL_Y)
        angle_diff = abs(angle_to_player - angle_to_goal_center)
        if angle_diff < 0.3:  # Within ~17 degrees
            blocks_shot = 1.0

    features = np.array([
        rel_x,                              # Relative x to ball
        rel_y,                              # Relative y to ball
        dist_to_ball / PITCH_LENGTH,        # Normalized distance to ball
        dist_to_goal / PITCH_LENGTH,        # Normalized distance to goal
        angle_from_ball / math.pi,          # Angle from ball [-1, 1]
        1.0 if is_teammate else 0.0,        # Is teammate
        1.0 if is_keeper else 0.0,          # Is goalkeeper
        blocks_shot,                        # Blocks shooting lane
        1.0 if is_in_box(player_x, player_y) else 0.0,  # In penalty box
        normalize_x(player_x),              # Absolute x position
    ], dtype=np.float32)

    return features


def extract_global_features(
    ball_x: float,
    ball_y: float,
    teammates: List[Dict[str, float]],
    opponents: List[Dict[str, float]]
) -> np.ndarray:
    """
    Extract global/aggregate features about the game state.
    Returns: [20] dimensional vector (expanded from 10)
    """
    n_teammates = len(teammates)
    n_opponents = len(opponents)

    # Defensive pressure: sum of 1/distance for nearby opponents
    defensive_pressure = 0.0
    nearest_defender_dist = PITCH_LENGTH
    for opp in opponents:
        dist = euclidean_distance(ball_x, ball_y, opp['x'], opp['y'])
        if dist > 0:
            defensive_pressure += 1.0 / max(dist, 1.0)
        nearest_defender_dist = min(nearest_defender_dist, dist)

    # Count players in box
    teammates_in_box = sum(1 for p in teammates if is_in_box(p['x'], p['y']))
    opponents_in_box = sum(1 for p in opponents if is_in_box(p['x'], p['y']))

    # Shooting lane blocked percentage
    shooting_lane_blocked = calculate_shooting_lane_blocked(
        ball_x, ball_y, opponents)

    # Nearest teammate distance
    nearest_teammate_dist = PITCH_LENGTH
    for tm in teammates:
        dist = euclidean_distance(ball_x, ball_y, tm['x'], tm['y'])
        nearest_teammate_dist = min(nearest_teammate_dist, dist)

    # NEW: Passing options analysis
    viable_passes, best_lane_quality = calculate_passing_options(
        ball_x, ball_y, teammates, opponents)

    # NEW: Space control
    space_control = calculate_space_control(
        ball_x, ball_y, teammates, opponents)

    # NEW: Defensive line position
    defensive_line = calculate_defensive_line(opponents)
    behind_defensive_line = 1.0 if ball_x > defensive_line else 0.0

    # NEW: Numerical advantage in final third
    teammates_final_third = sum(1 for p in teammates if p['x'] > 80)
    opponents_final_third = sum(1 for p in opponents if p['x'] > 80)
    final_third_advantage = (teammates_final_third -
                             opponents_final_third) / 6.0  # Normalize

    # NEW: Central vs wide position
    is_central = 1.0 if 25 < ball_y < 55 else 0.0  # Central channel

    # NEW: Counter-attack indicator (few defenders, ball in transition zone)
    counter_attack = 1.0 if (n_opponents < 6 and ball_x >
                             60 and nearest_defender_dist > 5) else 0.0

    features = np.array([
        # Original features
        n_teammates / 11.0,                          # 0: Normalized teammate count
        n_opponents / 11.0,                          # 1: Normalized opponent count
        # 2: Capped and normalized pressure
        min(defensive_pressure, 5.0) / 5.0,
        nearest_defender_dist / PITCH_LENGTH,        # 3: Normalized nearest defender
        # 4: Normalized (max ~5 in box)
        teammates_in_box / 5.0,
        # 5: Normalized (max ~6 in box)
        opponents_in_box / 6.0,
        shooting_lane_blocked,                       # 6: Already [0, 1]
        nearest_teammate_dist / PITCH_LENGTH,        # 7: Normalized nearest teammate
        # 8: No opponents visible (rare)
        1.0 if n_opponents == 0 else 0.0,
        1.0 if nearest_defender_dist < 3.0 else 0.0,  # 9: Under immediate pressure
        # New features
        # 10: Viable passing options
        min(viable_passes, 5) / 5.0,
        best_lane_quality,                           # 11: Best passing lane quality
        space_control,                               # 12: Space control score
        defensive_line / PITCH_LENGTH,               # 13: Defensive line position
        behind_defensive_line,                       # 14: Ball behind defensive line
        final_third_advantage,                       # 15: Numerical advantage
        is_central,                                  # 16: Central position
        counter_attack,                              # 17: Counter-attack situation
        teammates_final_third / 6.0,                 # 18: Teammates in final third
        opponents_final_third / 8.0,                 # 19: Opponents in final third
    ], dtype=np.float32)

    return features


def process_row(row: pd.Series) -> Optional[Dict]:
    """
    Process a single row from the chained dataset.

    Returns:
        Dict with keys: ball_features, teammate_features, opponent_features, 
                       global_features, label
        Or None if row should be skipped (no player data)
    """
    # Get ball position
    ball_x = row['start_x'] if pd.notna(row['start_x']) else None
    ball_y = row['start_y'] if pd.notna(row['start_y']) else None

    if ball_x is None or ball_y is None:
        ball_x, ball_y = 60.0, 40.0  # Default to center if missing

    # NOTE: p{i}_team is a binary flag (0.0 or 1.0), NOT the actual team_id
    # 1.0 = same team as the actor (attacking/ball-possessing team)
    # 0.0 = opponent team
    # So we compare with 1.0, not team_id
    TEAMMATE_FLAG = 1.0

    # Collect players
    teammates = []
    opponents = []
    has_any_player = False

    # Process outfield players (p0 - p19)
    for i in range(20):
        px = row.get(f'p{i}_x')
        py = row.get(f'p{i}_y')
        pteam = row.get(f'p{i}_team')

        if pd.notna(px) and pd.notna(py) and pd.notna(pteam):
            has_any_player = True
            player_data = {'x': float(px), 'y': float(py), 'is_keeper': False}

            # pteam == 1.0 means teammate, pteam == 0.0 means opponent
            if pteam == TEAMMATE_FLAG:
                teammates.append(player_data)
            else:
                opponents.append(player_data)

    # Process goalkeepers
    for k in [1, 2]:
        kx = row.get(f'keeper{k}_x')
        ky = row.get(f'keeper{k}_y')
        kteam = row.get(f'keeper{k}_team')

        if pd.notna(kx) and pd.notna(ky) and pd.notna(kteam):
            has_any_player = True
            player_data = {'x': float(kx), 'y': float(ky), 'is_keeper': True}

            if kteam == TEAMMATE_FLAG:
                teammates.append(player_data)
            else:
                opponents.append(player_data)

    # Skip rows with no player data at all
    if not has_any_player:
        return None

    # Extract ball features
    ball_features = extract_ball_features(row)

    # Extract per-player features
    teammate_features = []
    for p in teammates:
        feat = extract_player_features(
            p['x'], p['y'], ball_x, ball_y,
            is_teammate=True, is_keeper=p['is_keeper']
        )
        teammate_features.append(feat)

    opponent_features = []
    for p in opponents:
        feat = extract_player_features(
            p['x'], p['y'], ball_x, ball_y,
            is_teammate=False, is_keeper=p['is_keeper']
        )
        opponent_features.append(feat)

    # Extract global features
    global_features = extract_global_features(
        ball_x, ball_y, teammates, opponents)

    # Get label
    label = float(row['chain_goal']) if pd.notna(row['chain_goal']) else 0.0

    return {
        'ball_features': ball_features,
        'teammate_features': teammate_features,
        'opponent_features': opponent_features,
        'global_features': global_features,
        'label': label,
        'match_id': row['match_id'],
    }


def preprocess_dataset(
    filepath: str,
    output_path: Optional[str] = None,
    verbose: bool = True,
    chunk_size: int = 50000
) -> List[Dict]:
    """
    Preprocess the entire chained dataset for neural network training.
    Memory-efficient: processes in chunks and saves incrementally.

    Args:
        filepath: Path to statsbomb_chained_dataset.csv
        output_path: Optional path to save processed data as pickle
        verbose: Print progress
        chunk_size: Number of rows to process at a time (for memory efficiency)

    Returns:
        List of processed sample dictionaries
    """
    import gc

    if verbose:
        print(f"Loading dataset from {filepath}...")

    # Count total rows first
    total_rows = sum(1 for _ in open(filepath)) - 1  # -1 for header
    if verbose:
        print(
            f"Total rows: {total_rows}. Processing in chunks of {chunk_size}...")

    processed_samples = []
    skipped = 0
    rows_processed = 0

    # Process in chunks to save memory
    for chunk_df in pd.read_csv(filepath, chunksize=chunk_size):
        chunk_results = []

        for idx, row in chunk_df.iterrows():
            result = process_row(row)
            if result is not None:
                chunk_results.append(result)
            else:
                skipped += 1

        processed_samples.extend(chunk_results)
        rows_processed += len(chunk_df)

        if verbose:
            print(f"  Processed {rows_processed}/{total_rows} rows... "
                  f"({len(processed_samples)} samples, {skipped} skipped)")

        # Force garbage collection after each chunk
        del chunk_df, chunk_results
        gc.collect()

    if verbose:
        print(f"\nProcessing complete!")
        print(f"  Total samples: {len(processed_samples)}")
        print(f"  Skipped (no player data): {skipped}")

        # Stats on player counts
        tm_counts = [len(s['teammate_features']) for s in processed_samples]
        opp_counts = [len(s['opponent_features']) for s in processed_samples]
        print(
            f"  Teammates per sample: mean={np.mean(tm_counts):.1f}, max={max(tm_counts)}")
        print(
            f"  Opponents per sample: mean={np.mean(opp_counts):.1f}, max={max(opp_counts)}")

        # Label distribution
        labels = [s['label'] for s in processed_samples]
        print(f"  Goal rate: {np.mean(labels)*100:.2f}%")

    if output_path:
        if verbose:
            print(f"\nSaving to {output_path}...")

        # Save in chunks to avoid memory spike
        import pickle
        import tempfile
        import shutil

        # Use a temporary file to avoid corruption if crash during save
        temp_path = output_path + '.tmp'

        try:
            # Save using protocol 4 which is more memory efficient
            with open(temp_path, 'wb') as f:
                pickle.dump(processed_samples, f, protocol=4)

            # Atomic rename
            shutil.move(temp_path, output_path)

            if verbose:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  Saved! File size: {file_size_mb:.1f} MB")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    return processed_samples


def process_inference_input(
    ball_x: float,
    ball_y: float,
    teammates: List[Dict[str, float]],
    opponents: List[Dict[str, float]],
    team_keepers: List[Dict[str, float]] = None,
    opponent_keepers: List[Dict[str, float]] = None
) -> Dict:
    """
    Process input for inference (from frontend).

    Args:
        ball_x, ball_y: Ball position
        teammates: List of {'x': float, 'y': float}
        opponents: List of {'x': float, 'y': float}
        team_keepers: Optional list of own team keepers
        opponent_keepers: Optional list of opponent keepers

    Returns:
        Dict ready for model inference
    """
    # Add keepers to respective lists with is_keeper flag
    all_teammates = [{'x': p['x'], 'y': p['y'], 'is_keeper': False}
                     for p in teammates]
    all_opponents = [{'x': p['x'], 'y': p['y'], 'is_keeper': False}
                     for p in opponents]

    if team_keepers:
        for k in team_keepers:
            all_teammates.append({'x': k['x'], 'y': k['y'], 'is_keeper': True})

    if opponent_keepers:
        for k in opponent_keepers:
            all_opponents.append({'x': k['x'], 'y': k['y'], 'is_keeper': True})

    # Create a mock row for ball features
    mock_row = pd.Series({'start_x': ball_x, 'start_y': ball_y})
    ball_features = extract_ball_features(mock_row)

    # Extract player features
    teammate_features = []
    for p in all_teammates:
        feat = extract_player_features(
            p['x'], p['y'], ball_x, ball_y,
            is_teammate=True, is_keeper=p['is_keeper']
        )
        teammate_features.append(feat)

    opponent_features = []
    for p in all_opponents:
        feat = extract_player_features(
            p['x'], p['y'], ball_x, ball_y,
            is_teammate=False, is_keeper=p['is_keeper']
        )
        opponent_features.append(feat)

    # Global features
    global_features = extract_global_features(
        ball_x, ball_y, all_teammates, all_opponents)

    return {
        'ball_features': ball_features,
        'teammate_features': teammate_features,
        'opponent_features': opponent_features,
        'global_features': global_features,
        'label': 0,  # Dummy label for inference
    }


if __name__ == "__main__":
    # Test preprocessing
    import os

    module_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(module_dir, "statsbomb_chained_dataset.csv")
    output_path = os.path.join(module_dir, "nn_processed_data.pkl")

    if os.path.exists(input_path):
        samples = preprocess_dataset(input_path, output_path)
    else:
        print(f"Dataset not found at {input_path}")
