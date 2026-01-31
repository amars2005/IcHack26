# model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from basemodel import BaseModel
from preprocessor import DatasetPreprocessor

class ExpectedThreatModel(BaseModel):
    def __init__(self, filepath="statsbomb_normalised_dataset.csv", skip_training=False):
        self.filepath = filepath
        self.preprocessor = DatasetPreprocessor()
        
        if skip_training:
            # Initialize with empty data for inference-only mode
            self.feature_names = None
            self.model = None
        else:
            df_encoded, features = self.load_and_prep()
            super().__init__(df=df_encoded, feature_names=features, output='chain_goal')

    def load_model(self, filepath):
        """Load a pre-trained model from a pickle file."""
        import joblib
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
    
    def calculate_expected_threat(self, **kwargs):
        """
        Calculate xT for a single datapoint.
        
        Args:
            **kwargs: Raw event data matching chained dataset columns (all optional, default None):
                - match_id, timestamp, type, chain_id, chain_goal
                - start_x, start_y, end_x, end_y
                - p0_x, p0_y, p0_team, ..., p19_x, p19_y, p19_team
                - keeper1_x, keeper1_y, keeper1_team, keeper2_x, keeper2_y, keeper2_team
        
        Returns:
            float: Expected threat value (probability chain leads to goal)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Build full row with all possible columns, defaulting to None
        row_data = {
            'match_id': None, 'timestamp': None, 'type': None,
            'chain_id': None, 'chain_goal': None,
            'start_x': None, 'start_y': None, 'end_x': None, 'end_y': None,
        }
        # Add player columns
        for i in range(20):
            row_data[f'p{i}_x'] = None
            row_data[f'p{i}_y'] = None
            row_data[f'p{i}_team'] = None
        # Add keeper columns
        for i in [1, 2]:
            row_data[f'keeper{i}_x'] = None
            row_data[f'keeper{i}_y'] = None
            row_data[f'keeper{i}_team'] = None
        
        # Override with provided kwargs
        row_data.update(kwargs)
        
        # Create a single-row Series from row_data
        row = pd.Series(row_data)
        
        # Run through preprocessor
        processed = self.preprocessor.process_row(row)
        
        if processed is None:
            return None
        
        # Convert to DataFrame for feature engineering
        df = pd.DataFrame([processed])
        
        # One-hot encode the type
        action_type = row_data.get('type') or 'Pass'
        df['type_Pass'] = 1 if action_type == 'Pass' else 0
        df['type_Carry'] = 1 if action_type == 'Carry' else 0
        df['type_Shot'] = 1 if action_type == 'Shot' else 0
        
        # Build feature list (same order as training)
        player_feats = []
        for i in range(1, 6):  # 5 opponents
            player_feats.extend([f'opp_{i}_dist', f'opp_{i}_angle'])
        for i in range(1, 6):  # 5 teammates
            player_feats.extend([f'tm_{i}_dist', f'tm_{i}_angle'])
        player_feats.extend(['ball_end_dist', 'ball_end_angle'])
        
        features = ['dist_to_goal', 'type_Carry', 'type_Pass', 'type_Shot'] + player_feats
        
        # Ensure all feature columns exist (LightGBM handles NaN natively)
        for col in features:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Predict
        X = df[features]
        xt_value = self.model.predict_proba(X)[0, 1]
        
        return xt_value

    def load_and_prep(self):
        print("Loading and Engineering 360 Features...")
        df = pd.read_csv(self.filepath)

        player_feats = []
        
        # Count unique opponent indices (columns are 1-indexed: opp_1_dist, opp_2_dist, etc.)
        opp_count = len([c for c in df.columns if c.endswith('_dist') and c.startswith('opp_')])
        
        for i in range(1, opp_count + 1):
            opp_dist_col = f'opp_{i}_dist'
            opp_angle_col = f'opp_{i}_angle'

            player_feats.extend([opp_dist_col, opp_angle_col])

        # Count unique teammate indices (columns are 1-indexed: tm_1_dist, tm_2_dist, etc.)
        tm_count = len([c for c in df.columns if c.endswith('_dist') and c.startswith('tm_')])

        for i in range(1, tm_count + 1):
            tm_dist_col = f'tm_{i}_dist'
            tm_angle_col = f'tm_{i}_angle'
            player_feats.extend([tm_dist_col, tm_angle_col])

        # Add ball end features
        player_feats.extend(['ball_end_dist', 'ball_end_angle'])

        # Action Encoding
        df_encoded = pd.get_dummies(df, columns=['type'], prefix='type')
        action_cols = [c for c in df_encoded.columns if 'type_' in c]

        # Final Feature Selection
        features = ['dist_to_goal'] + action_cols + player_feats
        
        # Clean numeric data for LightGBM (NaN handled natively)
        for col in features:
            if col not in df_encoded.columns:
                df_encoded[col] = np.nan
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')

        return df_encoded, features


# --- EXECUTION ---
if __name__ == "__main__":
    import joblib
    import os
    
    try:
        xt_model = ExpectedThreatModel()
        df_with_preds = xt_model.run_pipeline()
        df_with_preds.to_csv("xt_model_output.csv", index=False)
        
        # Save the trained model
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "xt_model.pkl")
        joblib.dump(xt_model.model, model_path)
        
        print(f"Model saved to {model_path}")
    except FileNotFoundError:
        print("Error: 'statsbomb_chained_dataset.csv' not found.")
        print("Please run 'python main.py' first to generate the data.")