import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

class BaseModel():
    def __init__(self, df, feature_names, output):
        self.df = df
        # Increased depth slightly to capture more complex spatial patterns
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=10,
            random_state=42,
            importance_type='gain'
        )
        self.feature_names = feature_names
        self.output = output

    def train(self):
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

        print("Training Lightgbm...")
        self.model.fit(X_train, y_train)

        probs = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        
        print(f"\n--- Model Results ---")
        print(f"ROC AUC Score: {auc:.3f}")
        print(f"Brier Score: {brier:.3f}")
        
        # Show feature importance
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n--- Top 15 Feature Importances ---")
        print(importance_df.head(15).to_string(index=False))
        
        # Apply to whole dataset
        all_probs = self.model.predict_proba(df[features])[:, 1]
        df['pred_value'] = all_probs
        return df

    def _draw_pitch_markings(self, ax):
        """
        Helper function to draw standard football pitch markings 
        on StatsBomb coordinates (120x80).
        """
        lc = 'white' # Line color
        lw = 1.5     # Line width

        # 1. Outline & Center Line
        ax.plot([0, 0], [0, 80], color=lc, linewidth=lw)       # Left goal line
        ax.plot([120, 120], [0, 80], color=lc, linewidth=lw)   # Right goal line
        ax.plot([0, 120], [0, 0], color=lc, linewidth=lw)      # Bottom touchline
        ax.plot([0, 120], [80, 80], color=lc, linewidth=lw)    # Top touchline
        ax.plot([60, 60], [0, 80], color=lc, linewidth=lw)     # Halfway line

        # 2. Center Circle & Spot (Radius ~10 units)
        circle = mpatches.Circle((60, 40), 10, color=lc, fill=False, linewidth=lw)
        ax.add_patch(circle)
        ax.scatter(60, 40, color=lc, s=15) # Center spot

        # 3. Penalty Areas (18 yard box)
        # Left (y spans from 18 to 62, x spans 0 to 18)
        ax.plot([18, 18], [18, 62], color=lc, linewidth=lw)
        ax.plot([0, 18], [18, 18], color=lc, linewidth=lw)
        ax.plot([0, 18], [62, 62], color=lc, linewidth=lw)
        # Right (x spans 102 to 120)
        ax.plot([102, 102], [18, 62], color=lc, linewidth=lw)
        ax.plot([120, 102], [18, 18], color=lc, linewidth=lw)
        ax.plot([120, 102], [62, 62], color=lc, linewidth=lw)

        # 4. 6-Yard Boxes (Goal Areas)
        # Left (y spans 30 to 50, x spans 0 to 6)
        ax.plot([6, 6], [30, 50], color=lc, linewidth=lw)
        ax.plot([0, 6], [30, 30], color=lc, linewidth=lw)
        ax.plot([0, 6], [50, 50], color=lc, linewidth=lw)
        # Right (x spans 114 to 120)
        ax.plot([114, 114], [30, 50], color=lc, linewidth=lw)
        ax.plot([120, 114], [30, 30], color=lc, linewidth=lw)
        ax.plot([120, 114], [50, 50], color=lc, linewidth=lw)


    def visualize_value_map(self, df):
        """
        Creates a heatmap of the pitch showing average predicted xT value in each zone.
        Uses actual data predictions aggregated by location bins with smoothing.
        """
        from scipy.ndimage import gaussian_filter
        
        print("Generating Heatmap from actual predictions...")
        
        # Need original coordinates - reconstruct from dist_to_goal
        try:
            raw_df = pd.read_csv("statsbomb_chained_dataset.csv")
            df_with_coords = df[['pred_value']].copy()
            df_with_coords['start_x'] = raw_df['start_x'].values[:len(df)]
            df_with_coords['start_y'] = raw_df['start_y'].values[:len(df)]
        except:
            print("Warning: Could not load raw coordinates, using dist_to_goal approximation")
            df_with_coords = df[['pred_value', 'dist_to_goal']].copy()
            df_with_coords['start_x'] = 120 - df_with_coords['dist_to_goal']
            df_with_coords['start_y'] = 40
        
        # Higher resolution bins
        x_bins = np.linspace(0, 120, 61)  # 60 bins
        y_bins = np.linspace(0, 80, 41)   # 40 bins
        
        df_with_coords['x_bin'] = pd.cut(df_with_coords['start_x'], bins=x_bins, labels=False)
        df_with_coords['y_bin'] = pd.cut(df_with_coords['start_y'], bins=y_bins, labels=False)
        
        # Aggregate: mean xT per zone
        zone_avg = df_with_coords.groupby(['x_bin', 'y_bin'])['pred_value'].mean().reset_index()
        
        # Create heatmap grid
        z = np.zeros((len(y_bins)-1, len(x_bins)-1))
        z[:] = np.nan
        
        for _, row in zone_avg.iterrows():
            if pd.notna(row['x_bin']) and pd.notna(row['y_bin']):
                z[int(row['y_bin']), int(row['x_bin'])] = row['pred_value']
        
        # Fill NaN with nearest neighbor then smooth
        from scipy.ndimage import generic_filter
        
        # Fill NaNs with local mean
        mask = np.isnan(z)
        z_filled = np.where(mask, np.nanmean(z), z)
        
        # Apply Gaussian smoothing for a cleaner look
        z_smooth = gaussian_filter(z_filled, sigma=1.5)
        
        # Get bin centers for plotting
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2
        y_centers = (y_bins[:-1] + y_bins[1:]) / 2
        xx, yy = np.meshgrid(x_centers, y_centers)
        
        # --- PLOTTING ---
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Determine color scale from actual data
        vmax = np.nanpercentile(z_smooth, 99)
        vmin = np.nanpercentile(z_smooth, 1)
        
        # Draw smooth heatmap using contourf
        contour = ax.contourf(xx, yy, z_smooth, levels=30, cmap='magma', vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Average xT (Probability Chain Leads to Goal)')
        
        # Overlay pitch markings
        self._draw_pitch_markings(ax)
        
        # Final touches
        ax.set_title(f'Expected Threat Map (from {len(df):,} actual events)')
        ax.set_xlim(0, 120)
        ax.set_ylim(80, 0)  # Invert Y axis
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig("xt_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to xt_heatmap.png (value range: {vmin:.4f} - {vmax:.4f})")

    def run_pipeline(self):
        """Helper to train and visualize in one go."""
        df_with_preds = self.train()
        self.visualize_value_map(df_with_preds)
        return df_with_preds
