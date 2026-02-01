import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from basemodel import BaseModel
from preprocessor import DatasetPreprocessor
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

class PossessionValueLinearModel(BaseModel):
    def __init__(self, filepath="statsbomb_chained_dataset.csv"):
        self.filepath = filepath
        #self.preprocessor = DatasetPreprocessor()
        
        # Added class_weight='balanced' to help the model learn from rare goal events
        self.linear_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                penalty='l2', 
                C=1.0, 
                solver='lbfgs', 
                max_iter=1000, 
                class_weight='balanced' 
            ))
        ])

        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.linear_pipeline,
            method='isotonic', 
            cv=5 # Uses cross-validation to ensure calibration is honest
        )
        
        df_engineered, features = self.load_and_prep()
        super().__init__(df=df_engineered, feature_names=features, output='chain_goal')
        self.model = self.calibrated_model

    def _calculate_packing_rate(self, row):
        
        start_x = float(row['start_x'])
        end_x = float(row['end_x'])
        
            
        bypassed = 0
        # Check all 20 players in the 360 frames
        for i in range(20):
            x_col = f'p{i}_x'
            team_col = f'p{i}_team'
            
            if x_col in row and team_col in row:
                
                # Robust check: assumes 0 is teammate, 1 is opponent (or vice versa)
                # Adjust 'opponent' string if your dataset uses IDs
                if str(row[team_col]).lower() in ['opponent', '0.0']: 
                    
                    opp_x = row[x_col]
                    if pd.notnull(opp_x):
                        
                        if start_x < opp_x <= end_x:
                            
                            bypassed += 1
        return bypassed

    def load_and_prep(self):
        print("Engineering Advanced Linear Features...")
        df = pd.read_csv(self.filepath)

        # 1. Spatial Features
        df['dist_to_goal'] = np.sqrt((120 - df['start_x'])**2 + (40 - df['start_y'])**2)
        df['inv_dist_to_goal'] = 1 / (df['dist_to_goal'] + 1)
        
        # 2. 360 Features
        df['packing_rate'] = df.apply(self._calculate_packing_rate, axis=1)
        
        # Validation Print: Let's see if we actually caught any packing
        print(f"Mean Packing Rate: {df['packing_rate'].mean():.4f}")

        # 3. Action Encoding
        df = pd.get_dummies(df, columns=['type'], prefix='type')
        
        # 4. Interaction Terms (The 'Force' features)
        # This tells the model: "A pass is valuable specifically when it's near the goal"
        df['pass_danger'] = df.get('type_Pass', 0) * df['inv_dist_to_goal']
        df['packing_danger'] = df['packing_rate'] * df['inv_dist_to_goal']

        # List features
        action_cols = [c for c in df.columns if 'type_' in c]
        features = [
            'inv_dist_to_goal', 
            'packing_rate', 
            'pass_danger',
            'packing_danger'
        ] + action_cols

        # Final Clean
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df, features

    def fit(self):
        X = self.df[self.feature_names]
        y = self.df['chain_goal']
        print(f"Training and calibrating on {len(X)} samples...")
        self.model.fit(X, y)

    def validate(self):
        X = self.df[self.feature_names]
        y = self.df['chain_goal']
        
        # Get probabilities
        probs = self.model.predict_proba(X)[:, 1]
        
        auc = roc_auc_score(y, probs)
        brier = brier_score_loss(y, probs)
        
        print(f"\n--- Validation Metrics ---")
        print(f"ROC-AUC: {auc:.4f} (1.0 is perfect, 0.5 is random)")
        print(f"Brier Score: {brier:.4f} (Lower is better, 0.0 is perfect)")
    
    def predict_scenario(self, start_x, start_y, action_type, packing=0):
        # 1. Feature Engineering
        dist = np.sqrt((120 - start_x)**2 + (40 - start_y)**2)
        inv_dist = 1 / (dist + 1)
        
        # Create a template dataframe with all features as 0
        row_df = pd.DataFrame(0, index=[0], columns=self.feature_names)
        
        # 2. Fill Values
        row_df['inv_dist_to_goal'] = inv_dist
        row_df['packing_rate'] = packing
        row_df[f'type_{action_type}'] = 1
        
        # Interactions
        if action_type == 'Pass':
            row_df['pass_danger'] = inv_dist
        row_df['packing_danger'] = packing * inv_dist
        
        # 3. Predict
        pv = self.model.predict_proba(row_df)[0, 1]
        return pv

    
# --- EXECUTION ---
if __name__ == "__main__":
    try:
        pv_model = PossessionValueLinearModel()
        pv_model.fit()
        
        # Accessing coefficients from the pipeline
        # Step 1: Get the LogisticRegression object ('clf') from the pipeline
        first_fold_pipeline = pv_model.model.calibrated_classifiers_[0].estimator
        logit_step = first_fold_pipeline.named_steps['clf']

        coeffs = pd.Series(
            logit_step.coef_[0], 
            index=pv_model.feature_names
        ).sort_values(ascending=False)

        print("\n--- Calibrated Model Coefficients (First Fold) ---")
        print(coeffs)

        pv_model.validate()
        # Test them out
        scenarios = [
            (20, 40, 'Pass', 0),   # Deep
            (60, 40, 'Pass', 3),   # Line breaker
            (100, 70, 'Carry', 1), # Wing attack
            (115, 40, 'Shot', 0)   # Shot
        ]

        print("\n--- Scenario Testing ---")
        for x, y, act, pack in scenarios:
            val = pv_model.predict_scenario(x, y, act, pack)
            print(f"{act} from ({x},{y}) with {pack} packing: PV = {val:.4f}")

        

    except FileNotFoundError:
        print("Error: CSV file not found.")