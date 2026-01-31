# model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from basemodel import BaseModel 

class ExpectedThreatModel(BaseModel):
    def __init__(self, filepath="statsbomb_normalised_dataset.csv"):
        self.filepath = filepath
        df_encoded, features = self.load_and_prep()
        
        super().__init__(df=df_encoded, feature_names=features, output='chain_goal')

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
        
        # Clean numeric data for LightGBM
        for col in features:
            if col not in df_encoded.columns:
                df_encoded[col] = -1
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(-1)

        return df_encoded, features


# --- EXECUTION ---
if __name__ == "__main__":
    # NOTE: Ensure statsbomb_chained_dataset.csv exists first!
    try:
        xt_model = ExpectedThreatModel()
        df_with_preds = xt_model.run_pipeline()
        df_with_preds.to_csv("xt_model_output.csv", index=False)
    except FileNotFoundError:
        print("Error: 'statsbomb_chained_dataset.csv' not found.")
        print("Please run 'python main.py' first to generate the data.")