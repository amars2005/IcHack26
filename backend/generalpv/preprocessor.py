import pandas as pd
import numpy as np
import math
from tqdm import tqdm # For progress bar

# --- CONFIGURATION ---
INPUT_FILE = "statsbomb_chained_dataset.csv"
OUTPUT_FILE = "statsbomb_normalised_dataset.csv"
K_NEAREST_OPP = 5   # Number of nearest opponents to keep (including keeper)
K_NEAREST_TM = 5    # Number of nearest teammates to keep (including keeper)
GOAL_X = 120.0
GOAL_Y = 40.0

class DatasetPreprocessor:
    def __init__(self):
        self.k_opp = K_NEAREST_OPP
        self.k_tm = K_NEAREST_TM

    def _rotate_point(self, x, y, angle_rad):
        """Rotates a 2D point around (0,0) by angle_rad."""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Standard rotation formula
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        return x_new, y_new

    def process_row(self, row):
        """
        Takes a single row from the raw CSV and returns a feature dictionary.
        
        Normalization:
        1. Translate so ball is at origin (0, 0)
        2. Rotate so ball->goal vector aligns with +X axis
        3. All player positions stored as (dist, angle) in this normalized space
        """
        # 1. SETUP BALL & CONTEXT
        ball_x = row['start_x']
        ball_y = row['start_y']
        
        # Handle bad data (if ball coords are missing)
        if pd.isna(ball_x) or pd.isna(ball_y):
            return None 

        # Vector from Ball -> Goal
        goal_vec_x = GOAL_X - ball_x
        goal_vec_y = GOAL_Y - ball_y
        dist_to_goal = math.sqrt(goal_vec_x**2 + goal_vec_y**2)
        
        # Angle of attack (Ball -> Goal) relative to global pitch
        # We rotate by NEGATIVE this angle to align attack with +X axis
        attack_angle = math.atan2(goal_vec_y, goal_vec_x)
        rotation_angle = -attack_angle

        # 2. PARSE ALL PLAYERS (outfield p0...p19 + keepers)
        opponents = []
        teammates = []

        # Loop through p0 to p19 (outfield players)
        for i in range(20):
            px_col = f'p{i}_x'
            py_col = f'p{i}_y'
            pteam_col = f'p{i}_team'
            
            # Skip if column doesn't exist or value is null
            if px_col not in row or pd.isna(row[px_col]):
                continue

            p_x = row[px_col]
            p_y = row[py_col]
            is_teammate = bool(row[pteam_col]) if pd.notna(row[pteam_col]) else False

            # A. Translate (Center at Ball)
            rel_x = p_x - ball_x
            rel_y = p_y - ball_y
            
            # B. Rotate (Align to Goal Axis)
            rot_x, rot_y = self._rotate_point(rel_x, rel_y, rotation_angle)
            
            # C. Polar Coords
            dist = math.sqrt(rot_x**2 + rot_y**2)
            angle = math.atan2(rot_y, rot_x)
            
            # D. Filter out the "Actor" (The player with the ball)
            # If a teammate is < 0.2m from ball, it's likely the actor
            if is_teammate and dist < 0.2:
                continue

            player_feat = {
                'dist': dist,
                'angle': angle
            }

            if is_teammate:
                teammates.append(player_feat)
            else:
                opponents.append(player_feat)

        # 2b. PARSE KEEPERS (keeper1 and keeper2) - treat as normal players
        for i in [1, 2]:
            kx_col = f'keeper{i}_x'
            ky_col = f'keeper{i}_y'
            kteam_col = f'keeper{i}_team'
            
            if kx_col not in row or pd.isna(row[kx_col]):
                continue
            
            k_x = row[kx_col]
            k_y = row[ky_col]
            is_teammate = bool(row[kteam_col]) if pd.notna(row[kteam_col]) else False
            
            # Translate and rotate
            rel_x = k_x - ball_x
            rel_y = k_y - ball_y
            rot_x, rot_y = self._rotate_point(rel_x, rel_y, rotation_angle)
            dist = math.sqrt(rot_x**2 + rot_y**2)
            angle = math.atan2(rot_y, rot_x)
            
            keeper_feat = {
                'dist': dist,
                'angle': angle
            }
            
            # Add keepers to their respective team lists
            if is_teammate:
                teammates.append(keeper_feat)
            else:
                opponents.append(keeper_feat)

        # 3. SORT BY DISTANCE (non-null values will naturally come first)
        opponents.sort(key=lambda p: p['dist'])
        teammates.sort(key=lambda p: p['dist'])

        # 4. COMPUTE BALL END POSITION (normalized from start ball origin)
        end_x = row.get('end_x')
        end_y = row.get('end_y')
        
        if pd.notna(end_x) and pd.notna(end_y):
            # Translate relative to start ball position
            end_rel_x = end_x - ball_x
            end_rel_y = end_y - ball_y
            
            # Rotate to align with attack axis
            end_rot_x, end_rot_y = self._rotate_point(end_rel_x, end_rel_y, rotation_angle)
            
            # Convert to polar coords
            ball_end_dist = math.sqrt(end_rot_x**2 + end_rot_y**2)
            ball_end_angle = math.atan2(end_rot_y, end_rot_x)
        else:
            ball_end_dist = np.nan
            ball_end_angle = np.nan

        # 5. CONSTRUCT OUTPUT FEATURES
        features = {
            'match_id': row['match_id'],
            'timestamp': row['timestamp'],
            'type': row['type'],
            'chain_id': row['chain_id'],
            'chain_goal': row['chain_goal'],  # LABEL
            
            # Global Context
            'dist_to_goal': dist_to_goal,
            'ball_end_dist': ball_end_dist,
            'ball_end_angle': ball_end_angle,
            'visible_opponents': len(opponents),
            'visible_teammates': len(teammates)
        }

        # Fill Opponents (Nearest K) - only dist and angle
        for k in range(self.k_opp):
            prefix = f"opp_{k+1}"
            if k < len(opponents):
                features[f"{prefix}_dist"] = opponents[k]['dist']
                features[f"{prefix}_angle"] = opponents[k]['angle']
            else:
                features[f"{prefix}_dist"] = np.nan
                features[f"{prefix}_angle"] = np.nan

        # Fill Teammates (Nearest K) - only dist and angle
        for k in range(self.k_tm):
            prefix = f"tm_{k+1}"
            if k < len(teammates):
                features[f"{prefix}_dist"] = teammates[k]['dist']
                features[f"{prefix}_angle"] = teammates[k]['angle']
            else:
                features[f"{prefix}_dist"] = np.nan
                features[f"{prefix}_angle"] = np.nan

        return features

    def preprocess_dataframe(self, df, max_workers=1):
        """
        Process entire dataframe.
        
        Args:
            df: Input DataFrame
            max_workers: Number of parallel workers. Default 1 (single process)
                        is recommended as the per-row overhead is low and
                        multiprocessing adds serialization cost.
        """
        from tqdm import tqdm
        import gc
        
        total_rows = len(df)
        print(f"Preprocessing {total_rows} rows with {max_workers} worker(s)...")
        
        if max_workers == 1:
            # Single-threaded path - simpler and often faster for this workload
            processed_rows = []
            for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Processing"):
                result = self.process_row(row)
                if result is not None:
                    processed_rows.append(result)
            
            if not processed_rows:
                return pd.DataFrame()
            
            return pd.DataFrame(processed_rows)
        
        # Multi-process path (for large datasets where parallelism helps)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # Convert to list of dicts for easier parallel processing
        rows = df.to_dict('records')
        
        processed_rows = []
        
        # Process in chunks to reduce overhead
        chunk_size = 1000
        chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
        total_chunks = len(chunks)
        
        # Free the original rows list - we have chunks now
        del rows
        gc.collect()
        
        # Process in batches to limit memory usage
        # Only submit max_workers * 2 chunks at a time
        batch_size = max_workers * 2
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch_start in range(0, total_chunks, batch_size):
                batch_end = min(batch_start + batch_size, total_chunks)
                batch_chunks = chunks[batch_start:batch_end]
                
                # Submit this batch
                futures = {executor.submit(self._process_chunk, chunk): i 
                          for i, chunk in enumerate(batch_chunks)}
                
                # Wait for all futures in this batch to complete
                for future in as_completed(futures):
                    chunk_results = future.result()
                    processed_rows.extend(chunk_results)
                    completed += 1
                    if completed % 50 == 0 or completed == total_chunks:
                        print(f"Completed {completed}/{total_chunks} chunks...")
                
                # Clear references to help GC
                del futures
                del batch_chunks
                gc.collect()
        
        # Filter out None results and create DataFrame
        processed_rows = [r for r in processed_rows if r is not None]
        
        if not processed_rows:
            return pd.DataFrame()
        
        return pd.DataFrame(processed_rows)
    
    def _process_chunk(self, chunk):
        """Process a chunk of rows - used for parallel processing."""
        results = []
        for row_dict in chunk:
            # Convert dict to Series-like object for process_row
            row = pd.Series(row_dict)
            result = self.process_row(row)
            if result is not None:
                results.append(result)
        return results

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    
    print(f"Loading raw data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter for relevant events only
    relevant_types = ['Pass', 'Carry', 'Shot'] 
    df = df[df['type'].isin(relevant_types)]

    print("Processing rows...")
    processor = DatasetPreprocessor()
    
    # Apply processing with tqdm progress bar
    processed_rows = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        feat = processor.process_row(row)
        if feat:
            processed_rows.append(feat)

    print("Creating DataFrame...")
    df_processed = pd.DataFrame(processed_rows)

    print(f"Saving to {OUTPUT_FILE}...")
    df_processed.to_csv(OUTPUT_FILE, index=False)
    
    print("Done! Ready for XGBoost.")
    print(f"Total rows: {len(df_processed)}")
    print(df_processed.head())