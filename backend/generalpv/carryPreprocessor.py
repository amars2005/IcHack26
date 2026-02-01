"""
Carry Model Preprocessor.
Extracts rich features for predicting xT gain from carries.
Labels each carry with the xT at the end of its carry sequence.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
import pickle

# Pitch dimensions (StatsBomb coordinates)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
GOAL_Y = 40.0
GOAL_WIDTH = 8.0

# Zone boundaries
BOX_X_MIN = 102.0  # Penalty box
BOX_Y_MIN = 18.0
BOX_Y_MAX = 62.0
FINAL_THIRD_X = 80.0
CENTRAL_Y_MIN = 25.0
CENTRAL_Y_MAX = 55.0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


def is_in_final_third(x: float) -> bool:
    """Check if x is in the final third."""
    return x >= FINAL_THIRD_X


def is_central(y: float) -> bool:
    """Check if y is in the central channel."""
    return CENTRAL_Y_MIN <= y <= CENTRAL_Y_MAX


# ============================================================================
# PLAYER EXTRACTION
# ============================================================================

def extract_players_from_row(row: pd.Series, ball_team_id: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract teammates and opponents from a row.
    
    Returns:
        (teammates, opponents) - lists of dicts with 'x', 'y' keys
    """
    teammates = []
    opponents = []
    
    # Extract outfield players (p0 to p19)
    for i in range(20):
        px = row.get(f'p{i}_x')
        py = row.get(f'p{i}_y')
        pteam = row.get(f'p{i}_team')
        
        if pd.notna(px) and pd.notna(py) and pd.notna(pteam):
            player = {'x': float(px), 'y': float(py)}
            # pteam == 1 means same team as ball carrier, 0 means opponent
            if pteam == 1:
                teammates.append(player)
            else:
                opponents.append(player)
    
    # Extract keepers
    for keeper in ['keeper1', 'keeper2']:
        kx = row.get(f'{keeper}_x')
        ky = row.get(f'{keeper}_y')
        kteam = row.get(f'{keeper}_team')
        
        if pd.notna(kx) and pd.notna(ky) and pd.notna(kteam):
            player = {'x': float(kx), 'y': float(ky), 'is_keeper': True}
            if kteam == 1:
                teammates.append(player)
            else:
                opponents.append(player)
    
    return teammates, opponents


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_ball_position_features(ball_x: float, ball_y: float) -> Dict[str, float]:
    """
    Extract features related to the ball/carry starting position.
    Returns 8 features.
    """
    dist_to_goal = euclidean_distance(ball_x, ball_y, GOAL_X, GOAL_Y)
    angle_to_goal = angle_to_point(ball_x, ball_y, GOAL_X, GOAL_Y)
    
    return {
        'norm_start_x': normalize_x(ball_x),
        'norm_start_y': normalize_y(ball_y),
        'distance_to_goal': dist_to_goal / PITCH_LENGTH,
        'angle_to_goal': angle_to_goal / math.pi,  # Normalize to [-1, 1]
        'in_penalty_box': 1.0 if is_in_box(ball_x, ball_y) else 0.0,
        'in_final_third': 1.0 if is_in_final_third(ball_x) else 0.0,
        'is_central': 1.0 if is_central(ball_y) else 0.0,
        'progress': ball_x / PITCH_LENGTH,
    }


def extract_opponent_pressure_features(
    ball_x: float, 
    ball_y: float, 
    opponents: List[Dict]
) -> Dict[str, float]:
    """
    Extract features related to opponent pressure.
    Returns 8 features.
    """
    if not opponents:
        return {
            'nearest_opponent_distance': 1.0,
            'opponents_within_5m': 0.0,
            'opponents_within_10m': 0.0,
            'opponents_ahead': 0.0,
            'defensive_pressure': 0.0,
            'opponents_in_cone_to_goal': 0.0,
            'defensive_line_x': 1.0,
            'space_to_defensive_line': 1.0,
        }
    
    nearest_dist = float('inf')
    within_5m = 0
    within_10m = 0
    ahead_count = 0
    pressure = 0.0
    in_cone = 0
    
    angle_to_goal = angle_to_point(ball_x, ball_y, GOAL_X, GOAL_Y)
    
    for opp in opponents:
        dist = euclidean_distance(ball_x, ball_y, opp['x'], opp['y'])
        nearest_dist = min(nearest_dist, dist)
        
        if dist < 5:
            within_5m += 1
        if dist < 10:
            within_10m += 1
        
        # Opponent ahead of ball (between ball and goal)
        if opp['x'] > ball_x:
            ahead_count += 1
            
            # Check if in shooting cone (within ~20 degrees of goal direction)
            angle_to_opp = angle_to_point(ball_x, ball_y, opp['x'], opp['y'])
            if abs(angle_to_opp - angle_to_goal) < 0.35:  # ~20 degrees
                in_cone += 1
        
        # Pressure: inverse distance weighting
        if dist > 0:
            pressure += 1.0 / max(dist, 1.0)
    
    # Defensive line: 2nd highest x position (ignore keeper)
    opp_x_positions = sorted([o['x'] for o in opponents], reverse=True)
    defensive_line = opp_x_positions[1] if len(opp_x_positions) > 1 else opp_x_positions[0]
    space_to_line = max(0, defensive_line - ball_x)
    
    return {
        'nearest_opponent_distance': min(nearest_dist, PITCH_LENGTH) / PITCH_LENGTH,
        'opponents_within_5m': min(within_5m, 5) / 5.0,
        'opponents_within_10m': min(within_10m, 8) / 8.0,
        'opponents_ahead': min(ahead_count, 10) / 10.0,
        'defensive_pressure': min(pressure, 5.0) / 5.0,
        'opponents_in_cone_to_goal': min(in_cone, 5) / 5.0,
        'defensive_line_x': defensive_line / PITCH_LENGTH,
        'space_to_defensive_line': space_to_line / PITCH_LENGTH,
    }


def extract_space_features(
    ball_x: float, 
    ball_y: float, 
    opponents: List[Dict]
) -> Dict[str, float]:
    """
    Extract features related to available carrying space.
    Returns 6 features.
    """
    if not opponents:
        return {
            'open_carry_angle': 1.0,
            'space_ahead': 1.0,
            'space_central': 1.0,
            'space_wide_left': 1.0,
            'space_wide_right': 1.0,
            'can_beat_defensive_line': 1.0,
        }
    
    # Calculate open angles (simplified: check 8 directions)
    directions = [
        (1, 0),    # Forward
        (1, 0.5),  # Forward-right
        (1, -0.5), # Forward-left
        (0.7, 0.7),  # Diagonal right
        (0.7, -0.7), # Diagonal left
    ]
    
    open_directions = 0
    space_ahead = float('inf')
    space_central = float('inf')
    space_wide_left = float('inf')
    space_wide_right = float('inf')
    
    for opp in opponents:
        if opp['x'] <= ball_x:
            continue  # Only care about opponents ahead
        
        # Distance and angle to this opponent
        dist = euclidean_distance(ball_x, ball_y, opp['x'], opp['y'])
        angle = angle_to_point(ball_x, ball_y, opp['x'], opp['y'])
        
        # Check which lanes this opponent blocks
        # Forward (angle near 0)
        if abs(angle) < 0.3:
            space_ahead = min(space_ahead, dist)
        
        # Central (y between 25-55)
        if is_central(opp['y']) and opp['x'] > ball_x:
            space_central = min(space_central, dist)
        
        # Wide left (y > 55)
        if opp['y'] > 55 and opp['x'] > ball_x:
            space_wide_left = min(space_wide_left, dist)
        
        # Wide right (y < 25)
        if opp['y'] < 25 and opp['x'] > ball_x:
            space_wide_right = min(space_wide_right, dist)
    
    # Check directions for open carry angle
    for dx, dy in directions:
        # Normalize direction
        length = math.sqrt(dx**2 + dy**2)
        dx, dy = dx / length, dy / length
        
        # Check if any opponent blocks this direction (within 3m of ray)
        blocked = False
        for opp in opponents:
            if opp['x'] <= ball_x:
                continue
            
            # Project opponent onto ray
            to_opp_x = opp['x'] - ball_x
            to_opp_y = opp['y'] - ball_y
            
            # Dot product to find projection length
            proj_len = to_opp_x * dx + to_opp_y * dy
            if proj_len < 0 or proj_len > 20:  # Behind or too far
                continue
            
            # Perpendicular distance to ray
            perp_dist = abs(to_opp_x * dy - to_opp_y * dx)
            if perp_dist < 3:
                blocked = True
                break
        
        if not blocked:
            open_directions += 1
    
    # Can beat defensive line (space to get behind)
    opp_x_positions = sorted([o['x'] for o in opponents], reverse=True)
    defensive_line = opp_x_positions[1] if len(opp_x_positions) > 1 else opp_x_positions[0]
    can_beat = 1.0 if (space_ahead > 5 and ball_x < defensive_line) else 0.0
    
    return {
        'open_carry_angle': open_directions / 5.0,  # 5 directions checked
        'space_ahead': min(space_ahead, 30) / 30.0 if space_ahead != float('inf') else 1.0,
        'space_central': min(space_central, 30) / 30.0 if space_central != float('inf') else 1.0,
        'space_wide_left': min(space_wide_left, 30) / 30.0 if space_wide_left != float('inf') else 1.0,
        'space_wide_right': min(space_wide_right, 30) / 30.0 if space_wide_right != float('inf') else 1.0,
        'can_beat_defensive_line': can_beat,
    }


def extract_teammate_features(
    ball_x: float, 
    ball_y: float, 
    teammates: List[Dict],
    opponents: List[Dict]
) -> Dict[str, float]:
    """
    Extract features related to teammate support.
    Returns 4 features.
    """
    if not teammates:
        return {
            'teammates_ahead': 0.0,
            'nearest_teammate_distance': 1.0,
            'teammates_in_box': 0.0,
            'passing_options_if_stopped': 0.0,
        }
    
    ahead_count = 0
    nearest_dist = float('inf')
    in_box_count = 0
    passing_options = 0
    
    for tm in teammates:
        dist = euclidean_distance(ball_x, ball_y, tm['x'], tm['y'])
        nearest_dist = min(nearest_dist, dist)
        
        if tm['x'] > ball_x:
            ahead_count += 1
        
        if is_in_box(tm['x'], tm['y']):
            in_box_count += 1
        
        # Count as passing option if within reasonable range and has clear lane
        if 3 < dist < 30:
            # Simplified: check if any opponent blocks direct pass
            lane_clear = True
            for opp in opponents:
                # Point-to-line distance
                dx = tm['x'] - ball_x
                dy = tm['y'] - ball_y
                line_len_sq = dx**2 + dy**2
                
                if line_len_sq == 0:
                    continue
                
                t = max(0, min(1, ((opp['x'] - ball_x) * dx + (opp['y'] - ball_y) * dy) / line_len_sq))
                closest_x = ball_x + t * dx
                closest_y = ball_y + t * dy
                
                dist_to_line = euclidean_distance(opp['x'], opp['y'], closest_x, closest_y)
                if dist_to_line < 2.5:
                    lane_clear = False
                    break
            
            if lane_clear:
                passing_options += 1
    
    return {
        'teammates_ahead': min(ahead_count, 8) / 8.0,
        'nearest_teammate_distance': min(nearest_dist, PITCH_LENGTH) / PITCH_LENGTH,
        'teammates_in_box': min(in_box_count, 5) / 5.0,
        'passing_options_if_stopped': min(passing_options, 5) / 5.0,
    }


def extract_global_features(
    ball_x: float,
    ball_y: float,
    teammates: List[Dict],
    opponents: List[Dict]
) -> Dict[str, float]:
    """
    Extract global context features.
    Returns 4 features.
    """
    # Numerical advantage in final third
    tm_final_third = sum(1 for p in teammates if p['x'] > FINAL_THIRD_X)
    opp_final_third = sum(1 for p in opponents if p['x'] > FINAL_THIRD_X)
    advantage = (tm_final_third - opp_final_third) / 6.0  # Normalize
    
    # Counter-attack indicator
    nearest_opp_dist = min(
        (euclidean_distance(ball_x, ball_y, o['x'], o['y']) for o in opponents),
        default=PITCH_LENGTH
    )
    counter_attack = 1.0 if (len(opponents) < 8 and ball_x > 50 and nearest_opp_dist > 5) else 0.0
    
    # Compactness: std of opponent x positions
    if opponents:
        opp_x = [o['x'] for o in opponents]
        compactness = np.std(opp_x) / PITCH_LENGTH
    else:
        compactness = 0.5
    
    # Space control (simplified Voronoi-like)
    control_score = 0.0
    n_points = 0
    for dx in range(0, 25, 5):  # Forward-biased grid
        for dy in range(-15, 16, 5):
            sample_x = ball_x + dx
            sample_y = ball_y + dy
            
            if sample_x < 0 or sample_x > PITCH_LENGTH or sample_y < 0 or sample_y > PITCH_WIDTH:
                continue
            
            n_points += 1
            
            nearest_tm = min(
                (euclidean_distance(sample_x, sample_y, t['x'], t['y']) for t in teammates),
                default=PITCH_LENGTH
            )
            nearest_opp = min(
                (euclidean_distance(sample_x, sample_y, o['x'], o['y']) for o in opponents),
                default=PITCH_LENGTH
            )
            
            if nearest_tm < nearest_opp:
                control_score += 1
    
    space_control = control_score / max(n_points, 1)
    
    return {
        'numerical_advantage_final_third': np.clip(advantage, -1, 1),
        'counter_attack_indicator': counter_attack,
        'opponent_compactness': compactness,
        'space_control': space_control,
    }


def extract_all_features(row: pd.Series) -> Optional[Dict[str, float]]:
    """
    Extract all features from a single carry row.
    Returns dict of ~30 features or None if invalid.
    """
    ball_x = row.get('start_x')
    ball_y = row.get('start_y')
    
    if pd.isna(ball_x) or pd.isna(ball_y):
        return None
    
    ball_x = float(ball_x)
    ball_y = float(ball_y)
    
    # Get team ID for player extraction
    team_id = row.get('team_id', 0)
    
    # Extract players
    teammates, opponents = extract_players_from_row(row, team_id)
    
    # Extract all feature groups
    features = {}
    features.update(extract_ball_position_features(ball_x, ball_y))
    features.update(extract_opponent_pressure_features(ball_x, ball_y, opponents))
    features.update(extract_space_features(ball_x, ball_y, opponents))
    features.update(extract_teammate_features(ball_x, ball_y, teammates, opponents))
    features.update(extract_global_features(ball_x, ball_y, teammates, opponents))
    
    return features


# ============================================================================
# CARRY SEQUENCE LABELING
# ============================================================================

def has_spatial_data(row: dict) -> bool:
    """Check if row has any player spatial data."""
    for i in range(20):
        if pd.notna(row.get(f'p{i}_x')) and pd.notna(row.get(f'p{i}_y')):
            return True
    return False


def find_carry_sequences(df: pd.DataFrame, max_chains: int = None) -> List[Dict]:
    """
    Find sequences of consecutive carries and label each with the xT at sequence end.
    Uses batched GPU inference for speed.
    
    For: Carry A → Carry B → Carry C → Pass
    Labels A, B, C with xT at C's end position.
    
    Args:
        df: DataFrame with chained dataset
        max_chains: Maximum chains to process (for testing). None = all.
    
    Returns list of dicts with features and target_xt.
    """
    # Import xT model for labeling
    from backend.generalpv.expectedThreatModelNN import ExpectedThreatModelNN
    import torch
    
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"  🚀 Using GPU for xT inference: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print(f"  Using CPU for xT inference")
    
    print("  Loading xT model...")
    xt_model = ExpectedThreatModelNN(device=device)
    xt_model.load_model()
    
    # Phase 1: Collect all carry sequences that need xT labeling
    print("  Phase 1: Finding carry sequences...")
    
    sequences_to_label = []  # List of (sequence_info, xt_data)
    skipped_no_spatial = 0
    processed_chains = 0
    total_chains = df['chain_id'].nunique()
    
    if max_chains:
        print(f"    (Limited to {max_chains:,} chains for testing)")
    
    for chain_id, chain_df in df.groupby('chain_id'):
        processed_chains += 1
        
        if max_chains and processed_chains > max_chains:
            break
        
        if processed_chains % 10000 == 0:
            print(f"    Scanned {processed_chains:,}/{total_chains:,} chains ({100*processed_chains/total_chains:.1f}%)")
        
        chain_df = chain_df.sort_index()
        rows = chain_df.to_dict('records')
        indices = chain_df.index.tolist()
        
        i = 0
        while i < len(rows):
            if rows[i]['type'] != 'Carry':
                i += 1
                continue
            
            # Collect consecutive carries
            carry_sequence = []
            team_id = rows[i]['team_id']
            
            j = i
            while j < len(rows) and rows[j]['type'] == 'Carry' and rows[j]['team_id'] == team_id:
                carry_sequence.append((indices[j], rows[j]))
                j += 1
            
            if not carry_sequence:
                i += 1
                continue
            
            last_carry = carry_sequence[-1][1]
            end_x = last_carry.get('end_x')
            end_y = last_carry.get('end_y')
            
            # Skip if no spatial data
            if not has_spatial_data(last_carry):
                skipped_no_spatial += len(carry_sequence)
                i = j
                continue
            
            if pd.notna(end_x) and pd.notna(end_y):
                # Build xT data dict - only include non-NaN values
                xt_data = {'start_x': end_x, 'start_y': end_y}
                
                for k in range(20):
                    for suffix in ['x', 'y', 'team']:
                        key = f'p{k}_{suffix}'
                        val = last_carry.get(key)
                        if pd.notna(val):
                            xt_data[key] = val
                
                for keeper in ['keeper1', 'keeper2']:
                    for suffix in ['x', 'y', 'team']:
                        key = f'{keeper}_{suffix}'
                        val = last_carry.get(key)
                        if pd.notna(val):
                            xt_data[key] = val
                
                sequences_to_label.append((carry_sequence, xt_data))
            
            i = j
    
    print(f"    Found {len(sequences_to_label):,} carry sequences to label")
    print(f"    Skipped {skipped_no_spatial:,} carries with no spatial data")
    
    # Phase 2: Batch inference for xT values
    print("  Phase 2: Batched xT inference...")
    
    xt_data_list = [seq[1] for seq in sequences_to_label]
    
    # Process in larger batches for progress reporting
    batch_size = 5000
    all_xt_values = []
    
    for batch_start in range(0, len(xt_data_list), batch_size):
        batch_end = min(batch_start + batch_size, len(xt_data_list))
        batch = xt_data_list[batch_start:batch_end]
        
        xt_values = xt_model.calculate_expected_threat_batch(batch)
        all_xt_values.extend(xt_values)
        
        print(f"    Processed {batch_end:,}/{len(xt_data_list):,} sequences ({100*batch_end/len(xt_data_list):.1f}%)")
    
    # Phase 3: Extract features for each carry
    print("  Phase 3: Extracting features...")
    
    samples = []
    skipped_xt_none = 0
    
    for i, (carry_sequence, _) in enumerate(sequences_to_label):
        target_xt = all_xt_values[i]
        
        if target_xt is None:
            skipped_xt_none += 1
            continue
        
        for idx, carry_row in carry_sequence:
            if not has_spatial_data(carry_row):
                skipped_no_spatial += 1
                continue
            
            features = extract_all_features(pd.Series(carry_row))
            if features is not None:
                features['target_xt'] = target_xt
                features['carry_sequence_length'] = len(carry_sequence)
                features['position_in_sequence'] = [x[0] for x in carry_sequence].index(idx)
                samples.append(features)
        
        if (i + 1) % 10000 == 0:
            print(f"    Extracted features for {i+1:,}/{len(sequences_to_label):,} sequences ({len(samples):,} samples)")
    
    print(f"  Done! Generated {len(samples):,} training samples")
    print(f"    Skipped {skipped_xt_none:,} sequences due to xT errors")
    
    return samples


def preprocess_dataset(
    csv_path: str = None,
    output_path: str = None,
    max_samples: int = None,
    max_chains: int = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Preprocess the chained dataset for carry model training.
    
    Args:
        csv_path: Path to CSV file
        output_path: Optional path to save preprocessed data
        max_samples: Max samples to return (random subset if exceeded)
        max_chains: Max chains to process in Phase 1 (for faster testing)
    
    Returns:
        X: Feature matrix [n_samples, n_features]
        y: Target xT values [n_samples]
        feature_names: List of feature names
    """
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(__file__), 
            'statsbomb_chained_dataset.csv'
        )
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total rows: {len(df):,}")
    print(f"Carries: {len(df[df['type'] == 'Carry']):,}")
    
    print("Finding carry sequences and extracting features...")
    samples = find_carry_sequences(df, max_chains=max_chains)
    
    if max_samples and len(samples) > max_samples:
        import random
        samples = random.sample(samples, max_samples)
        print(f"Randomly sampled {max_samples:,} from {len(samples):,} samples")
    
    print(f"Final sample count: {len(samples):,}")
    
    if not samples:
        raise ValueError("No valid samples generated!")
    
    # Convert to arrays
    feature_names = [k for k in samples[0].keys() if k != 'target_xt']
    
    X = np.array([[s[f] for f in feature_names] for s in samples], dtype=np.float32)
    y = np.array([s['target_xt'] for s in samples], dtype=np.float32)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump({
                'X': X,
                'y': y,
                'feature_names': feature_names
            }, f)
        print(f"Saved preprocessed data to {output_path}")
    
    return X, y, feature_names


def process_inference_input(data: Dict) -> Optional[Dict[str, float]]:
    """
    Process input data for inference (same format as training).
    Used when predicting carry value for a new situation.
    """
    # Map ball_x/ball_y to start_x/start_y if needed
    inference_data = data.copy()
    if 'start_x' not in inference_data and 'ball_x' in inference_data:
        inference_data['start_x'] = inference_data['ball_x']
    if 'start_y' not in inference_data and 'ball_y' in inference_data:
        inference_data['start_y'] = inference_data['ball_y']
    
    # Convert dict to Series-like for feature extraction
    row = pd.Series(inference_data)
    return extract_all_features(row)


if __name__ == '__main__':
    # Test preprocessing
    X, y, feature_names = preprocess_dataset(max_samples=10000)
    
    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")
    
    print(f"\nSample features (first row):")
    for name, val in zip(feature_names, X[0]):
        print(f"  {name}: {val:.4f}")
    
    print(f"\nSample target: {y[0]:.4f}")
