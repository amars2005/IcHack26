# model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from basemodel import BaseModel 

class passProbabilityModel(BaseModel):
    def __init__(self, filepath="statsbomb_chained_dataset.csv"):
        self.filepath = filepath
        df_encoded, features = self.load_and_prep()
        
        super().__init__(df=df_encoded, feature_names=features, output='success')

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

        df = df[df['type'] == 'Pass']

        # Final Feature Selection
        features = ['dist_to_goal'] + player_feats
        
        # Clean numeric data for LightGBM
        for col in features:
            if col not in df.columns:
                df[col] = -1
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)

        return df, features

        print("Loading and Engineering 360 Features...")
        df = pd.read_csv(self.filepath)

    def run_pipeline(self):
        """Helper to train and visualize in one go."""
        df_with_preds = self.train()
        self.visualize_value_map(df_with_preds)
        return df_with_preds

# --- EXECUTION ---
if __name__ == "__main__":
    # NOTE: Ensure statsbomb_chained_dataset.csv exists first!
    try:
        pp_model = passProbabilityModel()
        df_with_preds = pp_model.run_pipeline
        df_with_preds.to_csv("pp_model_output.csv", index=False)
    except FileNotFoundError:
        print("Error: 'statsbomb_chained_dataset.csv' not found.")
        print("Please run 'python main.py' first to generate the data.")