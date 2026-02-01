import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Handle imports for both direct execution and module import
try:
    from .basemodel import BaseModel
except ImportError:
    from basemodel import BaseModel

class PassProbabilityModel(BaseModel):
    def __init__(self, filepath="statsbomb_chained_dataset.csv", skip_training=False):
        self.filepath = filepath
        self.scaler = None
        
        if skip_training:
            # Initialize with empty data for inference-only mode
            self.feature_names = [
                'pass_distance', 'pass_angle', 'pass_forward_component', 'pass_lateral_component',
                'start_distance_from_goal', 'end_distance_from_goal',
                'nearest_opponent_to_passer', 'nearest_opponent_to_target',
                'opponents_in_corridor', 'opponents_near_target',
                'teammates_near_target',
                'in_final_third', 'crossing_penalty_box', 'is_backwards', 'is_long_ball'
            ]
            self.model = None
        else:
            df_encoded, features = self.load_and_prep()
            
            # We use 'success' as the target for passes
            super().__init__(df=df_encoded, feature_names=features, output='success')
            
            # Override the model with Logistic Regression
            self.model = LogisticRegression(
                max_iter=2000,
                random_state=42,
                class_weight='balanced',  # Handle class imbalance (80% success rate)
                solver='lbfgs',
                penalty='l2',
                C=1.0,
                verbose=1
            )

    def load_model(self, filepath):
        """Load a pre-trained model and scaler from a pickle file."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        print(f"Model and scaler loaded from {filepath}")
    
    def load_and_prep(self):
        """Load and prepare the dataset for training."""
        print("Loading dataset...")
        df = pd.read_csv(self.filepath)
        
        # Filter for Pass events only
        df = df[df['type'] == 'Pass'].copy()
        print(f"Filtered to {len(df)} pass events")
        
        # Drop rows with missing critical data
        initial_count = len(df)
        df = df.dropna(subset=['start_x', 'start_y', 'end_x', 'end_y', 'success'])
        print(f"Dropped {initial_count - len(df)} rows with missing pass coordinates")
        
        # Calculate features
        print("Engineering features...")
        df = self._calculate_features(df)
        
        # Define feature columns
        feature_columns = [
            'pass_distance', 'pass_angle', 'pass_forward_component', 'pass_lateral_component',
            'start_distance_from_goal', 'end_distance_from_goal',
            'nearest_opponent_to_passer', 'nearest_opponent_to_target',
            'opponents_in_corridor', 'opponents_near_target',
            'teammates_near_target',
            'in_final_third', 'crossing_penalty_box', 'is_backwards', 'is_long_ball'
        ]
        
        # Drop any rows with NaN in features
        df = df.dropna(subset=feature_columns)
        print(f"Final dataset size: {len(df)} passes")
        
        # Normalize continuous features
        continuous_features = [
            'pass_distance', 'pass_angle', 'pass_forward_component', 'pass_lateral_component',
            'start_distance_from_goal', 'end_distance_from_goal',
            'nearest_opponent_to_passer', 'nearest_opponent_to_target',
            'opponents_in_corridor', 'opponents_near_target', 'teammates_near_target'
        ]
        
        self.scaler = StandardScaler()
        df[continuous_features] = self.scaler.fit_transform(df[continuous_features])
        
        return df, feature_columns
    
    def _calculate_features(self, df):
        """Calculate all features for the pass model."""
        
        # Basic pass geometry
        df['pass_distance'] = np.sqrt(
            (df['end_x'] - df['start_x'])**2 + 
            (df['end_y'] - df['start_y'])**2
        )
        
        df['pass_angle'] = np.arctan2(
            df['end_y'] - df['start_y'],
            df['end_x'] - df['start_x']
        )
        
        df['pass_forward_component'] = df['end_x'] - df['start_x']
        df['pass_lateral_component'] = np.abs(df['end_y'] - df['start_y'])
        
        # Position features (assuming goal is at x=120)
        df['start_distance_from_goal'] = np.sqrt(
            (120 - df['start_x'])**2 + (40 - df['start_y'])**2
        )
        df['end_distance_from_goal'] = np.sqrt(
            (120 - df['end_x'])**2 + (40 - df['end_y'])**2
        )
        
        # Defensive pressure features
        print("Calculating defensive pressure features...")
        df['nearest_opponent_to_passer'] = df.apply(
            lambda row: self._calculate_nearest_opponent(
                row, row['start_x'], row['start_y'], row['team_id']
            ), axis=1
        )
        
        df['nearest_opponent_to_target'] = df.apply(
            lambda row: self._calculate_nearest_opponent(
                row, row['end_x'], row['end_y'], row['team_id']
            ), axis=1
        )
        
        df['opponents_in_corridor'] = df.apply(
            lambda row: self._count_opponents_in_corridor(row), axis=1
        )
        
        df['opponents_near_target'] = df.apply(
            lambda row: self._count_players_near_point(
                row, row['end_x'], row['end_y'], row['team_id'], radius=5.0, same_team=False
            ), axis=1
        )
        
        # Teammate support
        print("Calculating teammate support features...")
        df['teammates_near_target'] = df.apply(
            lambda row: self._count_players_near_point(
                row, row['end_x'], row['end_y'], row['team_id'], radius=5.0, same_team=True
            ), axis=1
        )
        
        # Boolean/categorical features
        df['in_final_third'] = (df['end_x'] >= 80).astype(int)
        df['crossing_penalty_box'] = (
            (df['start_x'] < 102) & (df['end_x'] >= 102) &
            (df['end_y'] >= 18) & (df['end_y'] <= 62)
        ).astype(int)
        df['is_backwards'] = (df['pass_forward_component'] < 0).astype(int)
        df['is_long_ball'] = (df['pass_distance'] > 30).astype(int)
        
        return df
    
    def _calculate_nearest_opponent(self, row, target_x, target_y, team_id):
        """Calculate distance to nearest opponent from a given point."""
        min_dist = 999.0
        
        for i in range(20):  # p0 through p19
            px = row.get(f'p{i}_x')
            py = row.get(f'p{i}_y')
            p_team = row.get(f'p{i}_team')
            
            if pd.isna(px) or pd.isna(py) or pd.isna(p_team):
                continue
            
            # Check if opponent (different team)
            # team_id is the passing team, p_team: 1 = same team, 0 = opponent
            if p_team == 0:  # Opponent
                dist = np.sqrt((px - target_x)**2 + (py - target_y)**2)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _count_opponents_in_corridor(self, row):
        """Count opponents in the passing corridor (simplified cone)."""
        start_x, start_y = row['start_x'], row['start_y']
        end_x, end_y = row['end_x'], row['end_y']
        
        # Define corridor width (meters on each side of pass line)
        corridor_width = 3.0
        
        count = 0
        for i in range(20):
            px = row.get(f'p{i}_x')
            py = row.get(f'p{i}_y')
            p_team = row.get(f'p{i}_team')
            
            if pd.isna(px) or pd.isna(py) or pd.isna(p_team):
                continue
            
            if p_team == 0:  # Opponent
                # Check if point is in corridor
                if self._point_in_corridor(start_x, start_y, end_x, end_y, px, py, corridor_width):
                    count += 1
        
        return count
    
    def _point_in_corridor(self, x1, y1, x2, y2, px, py, width):
        """Check if point (px, py) is within corridor of width from line (x1,y1)-(x2,y2)."""
        # Calculate perpendicular distance from point to line
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if line_length < 0.1:  # Avoid division by zero
            return False
        
        # Calculate distance
        dist = np.abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / line_length
        
        # Check if point is between start and end (projection)
        dot_product = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length ** 2)
        
        return dist <= width and 0 <= dot_product <= 1
    
    def _count_players_near_point(self, row, target_x, target_y, team_id, radius=5.0, same_team=True):
        """Count players (teammates or opponents) near a target point."""
        count = 0
        
        for i in range(20):
            px = row.get(f'p{i}_x')
            py = row.get(f'p{i}_y')
            p_team = row.get(f'p{i}_team')
            
            if pd.isna(px) or pd.isna(py) or pd.isna(p_team):
                continue
            
            # Check team: p_team=1 is same team, p_team=0 is opponent
            if same_team and p_team == 1:
                dist = np.sqrt((px - target_x)**2 + (py - target_y)**2)
                if dist <= radius:
                    count += 1
            elif not same_team and p_team == 0:
                dist = np.sqrt((px - target_x)**2 + (py - target_y)**2)
                if dist <= radius:
                    count += 1
        
        return count
    
    def train(self):
        """Override train method to handle LogisticRegression specifics."""
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

        print("Training Logistic Regression...")
        self.model.fit(X_train, y_train)

        from sklearn.metrics import roc_auc_score, brier_score_loss
        probs = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        
        print(f"\n--- Model Results ---")
        print(f"ROC AUC Score: {auc:.3f}")
        print(f"Brier Score: {brier:.3f}")
        
        # Show feature importance (coefficients for LogisticRegression)
        importance_df = pd.DataFrame({
            'feature': features,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        print(f"\n--- Top 15 Feature Coefficients ---")
        print(importance_df.head(15).to_string(index=False))
        
        # Apply to whole dataset
        all_probs = self.model.predict_proba(df[features])[:, 1]
        df['pred_value'] = all_probs
        return df
    
    def calculate_pass_probability(self, **kwargs):
        """
        Calculate pass success probability for a single pass.
        
        Args:
            **kwargs: Pass data including:
                - start_x, start_y: Starting position of the pass
                - end_x, end_y: Target position of the pass
                - team_id: Team making the pass
                - p_0_x through p_19_x, p_0_y through p_19_y: Player positions
                - p_0_team through p_19_team: Player teams (0=opponent, 1=teammate)
                
        Returns:
            float: Pass success probability (0 to 1)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Call load_model() first or train the model.")
        
        # Extract pass details
        start_x = kwargs.get('start_x', 60)
        start_y = kwargs.get('start_y', 40)
        end_x = kwargs.get('end_x', 80)
        end_y = kwargs.get('end_y', 40)
        team_id = kwargs.get('team_id', 0)
        
        # Create a temporary row for feature calculation
        temp_row = pd.Series(kwargs)
        temp_row['start_x'] = start_x
        temp_row['start_y'] = start_y
        temp_row['end_x'] = end_x
        temp_row['end_y'] = end_y
        temp_row['team_id'] = team_id
        
        # Calculate features
        features = {}
        
        # Basic geometry
        features['pass_distance'] = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        features['pass_angle'] = np.arctan2(end_y - start_y, end_x - start_x)
        features['pass_forward_component'] = end_x - start_x
        features['pass_lateral_component'] = np.abs(end_y - start_y)
        
        # Position features
        features['start_distance_from_goal'] = np.sqrt((120 - start_x)**2 + (40 - start_y)**2)
        features['end_distance_from_goal'] = np.sqrt((120 - end_x)**2 + (40 - end_y)**2)
        
        # Pressure features
        features['nearest_opponent_to_passer'] = self._calculate_nearest_opponent(
            temp_row, start_x, start_y, team_id
        )
        features['nearest_opponent_to_target'] = self._calculate_nearest_opponent(
            temp_row, end_x, end_y, team_id
        )
        features['opponents_in_corridor'] = self._count_opponents_in_corridor(temp_row)
        features['opponents_near_target'] = self._count_players_near_point(
            temp_row, end_x, end_y, team_id, radius=5.0, same_team=False
        )
        features['teammates_near_target'] = self._count_players_near_point(
            temp_row, end_x, end_y, team_id, radius=5.0, same_team=True
        )
        
        # Boolean features
        features['in_final_third'] = int(end_x >= 80)
        features['crossing_penalty_box'] = int(
            start_x < 102 and end_x >= 102 and 18 <= end_y <= 62
        )
        features['is_backwards'] = int(features['pass_forward_component'] < 0)
        features['is_long_ball'] = int(features['pass_distance'] > 30)
        
        # Create feature array in correct order
        feature_array = np.array([[features[f] for f in self.feature_names]])
        
        # Scale continuous features
        continuous_features = [
            'pass_distance', 'pass_angle', 'pass_forward_component', 'pass_lateral_component',
            'start_distance_from_goal', 'end_distance_from_goal',
            'nearest_opponent_to_passer', 'nearest_opponent_to_target',
            'opponents_in_corridor', 'opponents_near_target', 'teammates_near_target'
        ]
        continuous_indices = [i for i, f in enumerate(self.feature_names) if f in continuous_features]
        feature_array[0, continuous_indices] = self.scaler.transform(
            feature_array[0, continuous_indices].reshape(1, -1)
        ).flatten()
        
        # Predict
        probability = self.model.predict_proba(feature_array)[0, 1]
        
        return probability


if __name__ == "__main__":
    # Train the model
    print("=" * 60)
    print("TRAINING PASS PROBABILITY MODEL")
    print("=" * 60)
    
    model = PassProbabilityModel(filepath="../../statsbomb_chained_dataset.csv")
    model.train()
    
    # Save the model
    model_path = "../../models/pass_probability_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump({
        'model': model.model,
        'scaler': model.scaler
    }, model_path)
    
    print(f"\nModel saved to {model_path}")
    print("=" * 60)