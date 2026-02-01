import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression

# Handle imports for both direct execution and module import
try:
    from .basemodel import BaseModel
except ImportError:
    from basemodel import BaseModel

class ExpectedGoalModel(BaseModel):
    def __init__(self, filepath="statsbomb_chained_dataset.csv", skip_training=False):
        self.filepath = filepath
        self.scaler = None
        
        if skip_training:
            # Initialize with empty data for inference-only mode
            self.feature_names = ['dist_to_goal', 'angle_to_goal', 'dist_to_keeper', 
                   'keeper_off_line', 'keeper_lateral_dist', 'keeper_angle_diff',
                   'defenders_in_cone', 'nearest_defender_dist', 'defenders_in_penalty_box',
                   'defenders_blocking_path', 'in_penalty_box', 'in_six_yard_box',
                   'central_shot', 'distance_from_center']
            self.model = None
        else:
            df_encoded, features = self.load_and_prep()
            
            # We use 'success' as the target for individual shots
            super().__init__(df=df_encoded, feature_names=features, output='success')
            
            # Override the model with Logistic Regression instead of LightGBM
            self.model = LogisticRegression(
                max_iter=2000,  # Increased for better convergence
                random_state=42,
                class_weight='balanced',  # Handle class imbalance (goals are rare)
                solver='lbfgs',
                verbose=0
            )

    def load_model(self, filepath):
        """Load a pre-trained model and scaler from a pickle file."""
        import joblib
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        print(f"Model and scaler loaded from {filepath}")
    
    def calculate_expected_goal(self, **kwargs):
        """
        Calculate xG for a single shot position.
        
        Args:
            **kwargs: Shot data including:
                - ball_x, ball_y: Position of the ball (shot location)
                - keeper_1_x, keeper_1_y: Opposing goalkeeper position
                - p_0_x through p_20_x, p_0_y through p_20_y: Player positions
                - p_0_team through p_20_team: Player teams (0=defender, 1=attacker)
                
        Returns:
            float: Expected goal probability (0 to 1)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Extract shot position
        shot_x = kwargs.get('ball_x', 100)
        shot_y = kwargs.get('ball_y', 40)
        
        # Extract goalkeeper position (opponent's keeper)
        keeper_x = kwargs.get('keeper_2_x', 120)  # keeper_2 is the opponent keeper
        keeper_y = kwargs.get('keeper_2_y', 40)
        
        # Basic geometry features
        dist_to_goal = np.sqrt((120 - shot_x)**2 + (40 - shot_y)**2)
        angle_to_goal = np.abs(np.arctan2(40 - shot_y, 120 - shot_x))
        
        # Goalkeeper features
        dist_to_keeper = np.sqrt((keeper_x - shot_x)**2 + (keeper_y - shot_y)**2)
        keeper_off_line = np.abs(120 - keeper_x)
        keeper_lateral_dist = np.abs(40 - keeper_y)
        angle_to_keeper = np.arctan2(keeper_y - shot_y, keeper_x - shot_x)
        keeper_angle_diff = np.abs(angle_to_goal - np.abs(angle_to_keeper))
        
        # Calculate defensive pressure features from player positions
        defenders_in_cone = 0
        nearest_defender_dist = 999.0
        defenders_in_penalty_box = 0
        defenders_blocking_path = 0
        
        # Check all 22 player positions (p_0 through p_21)
        for i in range(22):
            px = kwargs.get(f'p_{i}_x')
            py = kwargs.get(f'p_{i}_y')
            p_team = kwargs.get(f'p_{i}_team')
            
            if px is None or py is None or p_team is None:
                continue
            
            # Skip if same team (team 1 = attackers, team 0 = defenders)
            if p_team == 1:
                continue
            
            # Calculate distance to this defender
            defender_dist = np.sqrt((px - shot_x)**2 + (py - shot_y)**2)
            nearest_defender_dist = min(nearest_defender_dist, defender_dist)
            
            # Count defenders in penalty box
            if px > 102:
                defenders_in_penalty_box += 1
            
            # Check if defender is between shot and goal
            if px > shot_x:
                goal_x, goal_y = 120, 40
                angle_to_goal_center = np.arctan2(goal_y - shot_y, goal_x - shot_x)
                angle_to_player = np.arctan2(py - shot_y, px - shot_x)
                angle_diff = np.abs(angle_to_goal_center - angle_to_player)
                
                # Defender in shooting cone (30-degree cone)
                if angle_diff < 0.26:
                    defenders_in_cone += 1
                
                # Blocking path (narrow cone, close distance)
                if angle_diff < 0.1 and defender_dist < 10:
                    defenders_blocking_path += 1
        
        # Shot location categories
        in_penalty_box = 1 if shot_x > 102 else 0
        in_six_yard_box = 1 if shot_x > 114 else 0
        central_shot = 1 if np.abs(shot_y - 40) < 10 else 0
        distance_from_center = np.abs(shot_y - 40)
        
        # Create feature array
        features_dict = {
            'dist_to_goal': dist_to_goal,
            'angle_to_goal': angle_to_goal,
            'dist_to_keeper': dist_to_keeper,
            'keeper_off_line': keeper_off_line,
            'keeper_lateral_dist': keeper_lateral_dist,
            'keeper_angle_diff': keeper_angle_diff,
            'defenders_in_cone': defenders_in_cone,
            'nearest_defender_dist': nearest_defender_dist,
            'defenders_in_penalty_box': defenders_in_penalty_box,
            'defenders_blocking_path': defenders_blocking_path,
            'in_penalty_box': in_penalty_box,
            'in_six_yard_box': in_six_yard_box,
            'central_shot': central_shot,
            'distance_from_center': distance_from_center
        }
        
        # Create DataFrame with features in correct order
        df = pd.DataFrame([features_dict])[self.feature_names]
        
        # Scale and predict
        X_scaled = self.scaler.transform(df)
        xg_value = self.model.predict_proba(X_scaled)[0, 1]
        
        return float(xg_value)

    def load_and_prep(self):
        print("🛠️ Engineering Features for xG with Goalkeeper and Defensive Pressure...")
        df = pd.read_csv(self.filepath)
        
        # 1. Filter for Shots
        df = df[df['type'] == 'Shot'].copy()

        # 2. Basic Geometry - distance and angle to goal
        df['dist_to_goal'] = np.sqrt((120 - df['start_x'])**2 + (40 - df['start_y'])**2)
        df['angle_to_goal'] = np.abs(np.arctan2(40 - df['start_y'], 120 - df['start_x']))

        # 3. Goalkeeper Position Features
        # Fill NaN goalkeeper positions with default position (center of goal)
        df['keeper1_x'] = df['keeper1_x'].fillna(120)
        df['keeper1_y'] = df['keeper1_y'].fillna(40)
        
        # Distance from shot to goalkeeper
        df['dist_to_keeper'] = np.sqrt((df['keeper1_x'] - df['start_x'])**2 + 
                                       (df['keeper1_y'] - df['start_y'])**2)
        
        # Goalkeeper distance from goal line (how far off their line)
        df['keeper_off_line'] = np.abs(120 - df['keeper1_x'])
        
        # Goalkeeper lateral position (distance from center of goal)
        df['keeper_lateral_dist'] = np.abs(40 - df['keeper1_y'])
        
        # Angle between shot-goal line and shot-keeper line
        angle_to_keeper = np.arctan2(df['keeper1_y'] - df['start_y'], 
                                     df['keeper1_x'] - df['start_x'])
        df['keeper_angle_diff'] = np.abs(df['angle_to_goal'] - np.abs(angle_to_keeper))

        # 4. Defensive Pressure Features
        print("   Calculating defensive pressure features...")
        df = self._calculate_defensive_pressure(df)
        
        # 5. Shot Location Categories
        df['in_penalty_box'] = (df['start_x'] > 102).astype(int)
        df['in_six_yard_box'] = (df['start_x'] > 114).astype(int)
        df['central_shot'] = (np.abs(df['start_y'] - 40) < 10).astype(int)
        df['distance_from_center'] = np.abs(df['start_y'] - 40)

        # Final Feature List
        features = ['dist_to_goal', 'angle_to_goal', 'dist_to_keeper', 
                   'keeper_off_line', 'keeper_lateral_dist', 'keeper_angle_diff',
                   'defenders_in_cone', 'nearest_defender_dist', 'defenders_in_penalty_box',
                   'defenders_blocking_path', 'in_penalty_box', 'in_six_yard_box',
                   'central_shot', 'distance_from_center']

        # Numeric Clean
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)

        return df, features
    
    def _calculate_defensive_pressure(self, df):
        """Calculate defensive pressure features from freeze frame player positions"""
        # Initialize features
        df['defenders_in_cone'] = 0
        df['nearest_defender_dist'] = 999.0  # Large default value
        df['defenders_in_penalty_box'] = 0
        df['defenders_blocking_path'] = 0
        
        for idx, row in df.iterrows():
            shot_x = row['start_x']
            shot_y = row['start_y']
            shooting_team = row['team_id']
            
            defenders_in_cone = 0
            defenders_blocking = 0
            defenders_in_box = 0
            min_defender_dist = 999.0
            
            # Check all 20 player positions
            for i in range(20):
                px = row[f'p{i}_x']
                py = row[f'p{i}_y']
                p_team = row[f'p{i}_team']
                
                # Skip if no player data
                if pd.isna(px) or pd.isna(py) or pd.isna(p_team):
                    continue
                
                # Skip if same team (p_team == 1.0 means same team as shooter)
                if p_team == 1.0:
                    continue
                    
                # Calculate distance to this defender
                defender_dist = np.sqrt((px - shot_x)**2 + (py - shot_y)**2)
                min_defender_dist = min(min_defender_dist, defender_dist)
                
                # Count defenders in penalty box (defensive third)
                if px > 102:
                    defenders_in_box += 1
                
                # Check if defender is between shot and goal
                if px > shot_x:  # Defender is closer to goal
                    # Calculate if defender is in shooting cone (30-degree cone to goal)
                    # Vector from shot to goal center
                    goal_x, goal_y = 120, 40
                    
                    # Angle to goal center
                    angle_to_goal_center = np.arctan2(goal_y - shot_y, goal_x - shot_x)
                    
                    # Angle to this player
                    angle_to_player = np.arctan2(py - shot_y, px - shot_x)
                    
                    # Angular difference
                    angle_diff = np.abs(angle_to_goal_center - angle_to_player)
                    
                    # Consider player in cone if within 15 degrees (~0.26 radians)
                    if angle_diff < 0.26:
                        defenders_in_cone += 1
                    
                    # Blocking path: very narrow cone directly to goal center
                    if angle_diff < 0.1 and defender_dist < 10:  # Within 10 meters
                        defenders_blocking += 1
            
            df.at[idx, 'defenders_in_cone'] = defenders_in_cone
            df.at[idx, 'nearest_defender_dist'] = min_defender_dist
            df.at[idx, 'defenders_in_penalty_box'] = defenders_in_box
            df.at[idx, 'defenders_blocking_path'] = defenders_blocking
        
        return df

    def train(self):
        """Override train method to show coefficients instead of feature importances"""
        from sklearn.metrics import roc_auc_score, brier_score_loss
        from sklearn.preprocessing import StandardScaler
        
        df = self.df
        output = self.output
        features = self.feature_names
        match_ids = df['match_id'].unique()
        train_size = int(len(match_ids) * 0.8)
        train_matches = match_ids[:train_size]
        test_matches = match_ids[train_size:]

        print(f"Training on {len(train_matches)} matches, Testing on {len(test_matches)} matches.")
        
        train_data = df[df['match_id'].isin(train_matches)]
        test_data = df[df['match_id'].isin(test_matches)]

        X_train = train_data[features]
        y_train = train_data[output]
        X_test = test_data[features]
        y_test = test_data[output]

        # Scale features for better convergence
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Training Logistic Regression...")
        self.model.fit(X_train_scaled, y_train)

        probs = self.model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        
        print(f"\n--- Model Results ---")
        print(f"ROC AUC Score: {auc:.3f}")
        print(f"Brier Score: {brier:.3f}")
        
        # Show feature coefficients (weights) for Logistic Regression
        coef_df = pd.DataFrame({
            'feature': features,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        print(f"\n--- Feature Coefficients ---")
        print(coef_df.to_string(index=False))
        
        # Apply to whole dataset
        all_scaled = self.scaler.transform(df[features])
        all_probs = self.model.predict_proba(all_scaled)[:, 1]
        df['pred_value'] = all_probs
        return df

    def run_pipeline(self):
        """Standard execution flow: Train -> Visualize with different keeper positions -> Return."""
        trained_df = self.train()
        
        # Generate heatmaps with different goalkeeper positions
        keeper_positions = [
            {'x': 120, 'y': 40, 'name': 'centered'},
            {'x': 118, 'y': 40, 'name': 'off_line_2m'},
            {'x': 120, 'y': 38, 'name': 'left_post'},
            {'x': 120, 'y': 42, 'name': 'right_post'},
            {'x': 115, 'y': 40, 'name': 'very_advanced'}
        ]
        
        for pos in keeper_positions:
            self.visualize_value_map(keeper_x=pos['x'], keeper_y=pos['y'], 
                                    position_name=pos['name'])
        
        return trained_df

    def visualize_value_map(self, keeper_x=120, keeper_y=40, position_name='default'):
        print(f"🎨 Generating xG Heatmap with keeper at ({keeper_x}, {keeper_y})...")
        x_range = np.linspace(0, 120, 100)
        y_range = np.linspace(0, 80, 80)
        xx, yy = np.meshgrid(x_range, y_range)
        
        grid = pd.DataFrame({'start_x': xx.ravel(), 'start_y': yy.ravel()})
        
        # Basic geometry features
        grid['dist_to_goal'] = np.sqrt((120 - grid['start_x'])**2 + (40 - grid['start_y'])**2)
        grid['angle_to_goal'] = np.abs(np.arctan2(40 - grid['start_y'], 120 - grid['start_x']))
        
        # Goalkeeper position features with specified keeper position
        grid['keeper1_x'] = keeper_x
        grid['keeper1_y'] = keeper_y
        
        grid['dist_to_keeper'] = np.sqrt((keeper_x - grid['start_x'])**2 + 
                                        (keeper_y - grid['start_y'])**2)
        grid['keeper_off_line'] = np.abs(120 - keeper_x)
        grid['keeper_lateral_dist'] = np.abs(40 - keeper_y)
        
        angle_to_keeper = np.arctan2(keeper_y - grid['start_y'], 
                                     keeper_x - grid['start_x'])
        grid['keeper_angle_diff'] = np.abs(grid['angle_to_goal'] - np.abs(angle_to_keeper))
        
        # Defensive pressure features - use average values from training data
        # These will be the same across the grid (baseline scenario)
        grid['defenders_in_cone'] = 1  # Assume 1 defender on average
        grid['nearest_defender_dist'] = 5  # Assume 5 meters on average
        grid['defenders_in_penalty_box'] = 4  # Typical defensive setup
        grid['defenders_blocking_path'] = 0  # Optimistic scenario - no direct block
        
        # Shot location categories
        grid['in_penalty_box'] = (grid['start_x'] > 102).astype(int)
        grid['in_six_yard_box'] = (grid['start_x'] > 114).astype(int)
        grid['central_shot'] = (np.abs(grid['start_y'] - 40) < 10).astype(int)
        grid['distance_from_center'] = np.abs(grid['start_y'] - 40)
        
        # Ensure all features are present
        for col in self.feature_names:
            if col not in grid.columns:
                grid[col] = -1

        # Scale features using the same scaler from training
        grid_scaled = self.scaler.transform(grid[self.feature_names])
        z = self.model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        contour = ax.contourf(xx, yy, z, levels=50, cmap='magma')
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('xG Probability', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        self._draw_pitch_markings(ax)
        
        # Draw goalkeeper position
        ax.scatter(keeper_x, keeper_y, c='cyan', s=200, marker='o', 
                  edgecolors='white', linewidths=2, alpha=0.8, label='Goalkeeper')
        ax.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='white', 
                 labelcolor='white', fontsize=10)
        
        ax.set_title(f"Expected Goals (xG) Map - Keeper: {position_name.replace('_', ' ').title()}", 
                    color='white', fontsize=15)
        ax.set_xlim(0, 120)
        ax.set_ylim(80, 0)
        ax.axis('off')
        
        filename = f"xg_heatmap_keeper_{position_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Heatmap saved to {filename}")

    def _draw_pitch_markings(self, ax):
        lc, lw, alpha = 'white', 1.5, 0.6
        ax.plot([0, 0, 120, 120, 0], [0, 80, 80, 0, 0], color=lc, linewidth=lw, alpha=alpha)
        ax.plot([60, 60], [0, 80], color=lc, linewidth=lw, alpha=alpha)
        ax.add_patch(mpatches.Circle((60, 40), 10, color=lc, fill=False, linewidth=lw, alpha=alpha))
        # Penalty Areas
        ax.plot([18, 18, 0], [18, 62, 62], color=lc, linewidth=lw, alpha=alpha)
        ax.plot([102, 102, 120], [18, 62, 62], color=lc, linewidth=lw, alpha=alpha)

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Path Setup
    DATA_PATH = "statsbomb_chained_dataset.csv"
    
    try:
        # 2. Run Pipeline
        print(f"--- Starting xG Model Training ---")
        xg_model = ExpectedGoalModel(filepath=DATA_PATH)
        final_df = xg_model.run_pipeline()
        
        # 3. Save Results
        final_df.to_csv("xg_model_output.csv", index=False)
        
        # 4. Save Model Artifact (both model and scaler)
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "xg_model_360.pkl")
        
        # Save both model and scaler together
        model_data = {
            'model': xg_model.model,
            'scaler': xg_model.scaler
        }
        joblib.dump(model_data, model_path)
        
        print(f"\n✅ Pipeline Complete.")
        print(f"📊 Results: xg_model_output.csv")
        print(f"💾 Model: {model_path}")
        print(f"🖼️ Heatmaps generated for 5 different goalkeeper positions")
        print(f"   - xg_heatmap_keeper_centered.png")
        print(f"   - xg_heatmap_keeper_off_line_2m.png")
        print(f"   - xg_heatmap_keeper_left_post.png")
        print(f"   - xg_heatmap_keeper_right_post.png")
        print(f"   - xg_heatmap_keeper_very_advanced.png")

    except FileNotFoundError:
        print(f"❌ Error: '{DATA_PATH}' not found. Check your file path.")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")