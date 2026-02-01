import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from basemodel import BaseModel

class ExpectedGoalModel(BaseModel):
    def __init__(self, filepath="statsbomb_chained_dataset.csv"):
        self.filepath = filepath
        df_encoded, features = self.load_and_prep()
        
        # We use 'success' as the target for individual shots
        super().__init__(df=df_encoded, feature_names=features, output='success')

    def load_and_prep(self):
        print("🛠️ Engineering Relative 360 Features for xG...")
        df = pd.read_csv(self.filepath)
        
        # 1. Filter for Shots
        df = df[df['type'] == 'Shot'].copy()

        # 2. Geometry
        df['dist_to_goal'] = np.sqrt((120 - df['start_x'])**2 + (40 - df['start_y'])**2)
        df['angle_to_goal'] = np.abs(np.arctan2(40 - df['start_y'], 120 - df['start_x']))

        # 3. 360 Relative Features
        player_features = []
        # Calculate distances for the 20 players (p0...p19)
        for i in range(20):
            px, py, pt = f'p{i}_x', f'p{i}_y', f'p{i}_team'
            dist_col = f'p{i}_dist'
            
            df[dist_col] = np.where(
                df[px] != -1, 
                np.sqrt((df[px] - df['start_x'])**2 + (df[py] - df['start_y'])**2), 
                -1
            )
            player_features.extend([dist_col, pt])

        # 4. Keeper Context
        for k in [1, 2]:
            kx, ky = f'keeper{k}_x', f'keeper{k}_y'
            k_dist = f'keeper{k}_dist'
            df[k_dist] = np.where(
                df[kx] != -1,
                np.sqrt((df[kx] - df['start_x'])**2 + (df[ky] - df['start_y'])**2),
                -1
            )
            player_features.append(k_dist)

        # Final Feature List
        features = ['dist_to_goal', 'angle_to_goal']

        # Numeric Clean
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)

        return df, features

    def run_pipeline(self):
        """Standard execution flow: Train -> Visualize -> Return."""
        trained_df = self.train()
        self.visualize_value_map()
        return trained_df

    def visualize_value_map(self):
        print("🎨 Generating xG Heatmap...")
        x_range = np.linspace(0, 120, 100)
        y_range = np.linspace(0, 80, 80)
        xx, yy = np.meshgrid(x_range, y_range)
        
        grid = pd.DataFrame({'start_x': xx.ravel(), 'start_y': yy.ravel()})
        grid['dist_to_goal'] = np.sqrt((120 - grid['start_x'])**2 + (40 - grid['start_y'])**2)
        grid['angle_to_goal'] = np.abs(np.arctan2(40 - grid['start_y'], 120 - grid['start_x']))
        
        for col in self.feature_names:
            if col not in grid.columns:
                grid[col] = -1

        z = self.model.predict_proba(grid[self.feature_names])[:, 1].reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        contour = ax.contourf(xx, yy, z, levels=50, cmap='magma')
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('xG Probability', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        self._draw_pitch_markings(ax)
        
        ax.set_title("Expected Goals (xG) Baseline Map", color='white', fontsize=15)
        ax.set_xlim(0, 120)
        ax.set_ylim(80, 0)
        ax.axis('off')
        
        plt.savefig("xg_heatmap.png", dpi=300, bbox_inches='tight')
        print("✅ Heatmap saved to xg_heatmap.png")

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
        
        # 4. Save Model Artifact
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "xg_model_360.pkl")
        joblib.dump(xg_model.model, model_path)
        
        print(f"\n✅ Pipeline Complete.")
        print(f"📊 Results: xg_model_output.csv")
        print(f"💾 Model: {model_path}")
        print(f"🖼️ Heatmap: xg_heatmap.png")

    except FileNotFoundError:
        print(f"❌ Error: '{DATA_PATH}' not found. Check your file path.")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")