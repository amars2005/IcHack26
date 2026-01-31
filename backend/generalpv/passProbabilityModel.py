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
        
        super().__init__(df=df_encoded, feature_names=features, output='chain_goal')

    def load_and_prep(self):
        print("Loading and Engineering 360 Features...")
        df = pd.read_csv(self.filepath)

        # 1. Ball Spatial Features
        df['dist_to_goal'] = np.sqrt((120 - df['start_x'])**2 + (40 - df['start_y'])**2)
        df['angle_to_goal'] = np.arctan2(40 - df['start_y'], 120 - df['start_x'])

        # 2. Relative 360 Features (Distance and Angle)
        player_feats = []
        # Dynamically find how many p{i}_x columns exist
        p_count = len([c for c in df.columns if c.endswith('_x') and c.startswith('p')])
        
        for i in range(p_count):
            px, py, pt = f'p{i}_x', f'p{i}_y', f'p{i}_team'
            
            dist_col = f'p{i}_rel_dist'
            ang_col = f'p{i}_rel_ang'
            
            # Distance: sqrt((px-bx)^2 + (py-by)^2)
            df[dist_col] = np.where(df[px] != -1,
                np.sqrt((df[px] - df['start_x'])**2 + (df[py] - df['start_y'])**2), -1)
            # Angle: atan2(py-by, px-bx)
            df[ang_col] = np.where(df[px] != -1,
                np.arctan2(df[py] - df['start_y'], df[px] - df['start_x']), -1)
            
            player_feats.extend([dist_col, ang_col, pt])

        df = df[df['type'] == 'Pass']

        # 4. Final Feature Selection
        features = ['start_x', 'start_y', 'dist_to_goal', 'angle_to_goal'] + player_feats
        
        # Clean numeric data for LightGBM
        for col in features:
            if col not in df.columns:
                df[col] = -1
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)

        

        return df, features

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