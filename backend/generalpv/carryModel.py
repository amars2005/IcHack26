"""
Carry Model - Predicts xT gain from carrying the ball.
Uses XGBoost with GPU acceleration for fast training.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from backend.generalpv.carryPreprocessor import (
    process_inference_input,
    preprocess_dataset,
    extract_all_features,
    extract_players_from_row
)
from backend.generalpv.expectedThreatModelNN import ExpectedThreatModelNN


# ============================================================================
# MODEL CLASS
# ============================================================================

class CarryModel:
    """
    XGBoost model to predict xT at the end of a carry sequence.
    Uses GPU acceleration for fast training.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize CarryModel.
        
        Args:
            model_path: Path to saved model. If None, uses default location.
        """
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: List[str] = []
        self.is_trained = False
        
        # Default model path
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'models',
                'carry_model.pkl'
            )
        self.model_path = model_path
        
        # Try to load existing model
        self._load_model()
        
        # xT model for calculating current position xT
        self._xt_model = None
    
    @property
    def xt_model(self) -> ExpectedThreatModelNN:
        """Lazy load xT model."""
        if self._xt_model is None:
            self._xt_model = ExpectedThreatModelNN()
            self._xt_model.load_model()
        return self._xt_model
    
    def _load_model(self) -> bool:
        """Load trained model from disk."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                self.model = data['model']
                self.feature_names = data['feature_names']
                self.is_trained = True
                print(f"Loaded carry model from {self.model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        return False
    
    def _save_model(self):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
            }, f)
        print(f"Saved carry model to {self.model_path}")
    
    def train(
        self,
        csv_path: str = None,
        preprocessed_path: str = None,
        save_preprocessed: str = None,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        max_samples: int = None,
        max_chains: int = None,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
        use_gpu: bool = True,
    ) -> Dict[str, float]:
        """
        Train the carry model using XGBoost with GPU.
        
        Args:
            csv_path: Path to chained dataset
            preprocessed_path: Path to load preprocessed data (skips preprocessing)
            save_preprocessed: Path to save preprocessed data for future use
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            subsample: Fraction of samples for each tree
            colsample_bytree: Fraction of features for each tree
            max_samples: Max samples to use (for faster training)
            max_chains: Max chains to process in Phase 1 (for faster testing)
            val_size: Fraction for validation set
            test_size: Fraction for test set
            random_state: Random seed
            use_gpu: Whether to use GPU acceleration
        
        Returns:
            Dict with training metrics
        """
        print("=" * 60)
        print("TRAINING CARRY MODEL (XGBoost)")
        print("=" * 60)
        
        # Preprocess data or load from cache
        print("\n[1/4] PREPROCESSING DATA")
        print("-" * 40)
        
        if preprocessed_path and os.path.exists(preprocessed_path):
            print(f"Loading preprocessed data from {preprocessed_path}")
            with open(preprocessed_path, 'rb') as f:
                data = pickle.load(f)
            X = data['X']
            y = data['y']
            feature_names = data['feature_names']
            print(f"Loaded {len(X):,} samples with {len(feature_names)} features")
        else:
            X, y, feature_names = preprocess_dataset(
                csv_path=csv_path,
                output_path=save_preprocessed,
                max_samples=max_samples,
                max_chains=max_chains
            )
        self.feature_names = feature_names
        
        # Train/Val/Test split
        print("\n[2/4] SPLITTING DATA")
        print("-" * 40)
        
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)  # Adjust ratio for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio, random_state=random_state
        )
        
        print(f"  Total samples:      {len(X):,}")
        print(f"  Training samples:   {len(X_train):,} ({100*len(X_train)/len(X):.1f}%)")
        print(f"  Validation samples: {len(X_val):,} ({100*len(X_val)/len(X):.1f}%)")
        print(f"  Test samples:       {len(X_test):,} ({100*len(X_test)/len(X):.1f}%)")
        print(f"\n  Target stats (training):")
        print(f"    Min:  {y_train.min():.4f}")
        print(f"    Max:  {y_train.max():.4f}")
        print(f"    Mean: {y_train.mean():.4f}")
        print(f"    Std:  {y_train.std():.4f}")
        
        # Determine device
        print("\n[3/4] TRAINING MODEL")
        print("-" * 40)
        
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                    tree_method = 'hist'
                    print(f"  🚀 Using GPU acceleration (CUDA)")
                    print(f"     Device: {torch.cuda.get_device_name(0)}")
                else:
                    device = 'cpu'
                    tree_method = 'hist'
                    print(f"  ⚠️ CUDA not available, using CPU")
            except:
                device = 'cpu'
                tree_method = 'hist'
                print(f"  Using CPU")
        else:
            device = 'cpu'
            tree_method = 'hist'
            print(f"  Using CPU (GPU disabled)")
        
        print(f"\n  Hyperparameters:")
        print(f"    n_estimators:     {n_estimators}")
        print(f"    max_depth:        {max_depth}")
        print(f"    learning_rate:    {learning_rate}")
        print(f"    subsample:        {subsample}")
        print(f"    colsample_bytree: {colsample_bytree}")
        print(f"    early_stopping:   50 rounds")
        
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method=tree_method,
            device=device,
            random_state=random_state,
            verbosity=1,
            early_stopping_rounds=50,
        )
        
        print(f"\n  Training...")
        # Train with early stopping on validation set
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50  # Print every 50 rounds
        )
        self.is_trained = True
        
        print(f"\n  Best iteration: {self.model.best_iteration}")
        
        # Evaluate
        print("\n[4/4] EVALUATING MODEL")
        print("-" * 40)
        
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_mse': mean_squared_error(y_val, val_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_r2': r2_score(y_val, val_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_r2': r2_score(y_test, test_pred),
        }
        
        print(f"\n  {'Dataset':<12} {'MSE':>10} {'MAE':>10} {'R²':>10}")
        print(f"  {'-'*44}")
        print(f"  {'Train':<12} {metrics['train_mse']:>10.6f} {metrics['train_mae']:>10.6f} {metrics['train_r2']:>10.4f}")
        print(f"  {'Validation':<12} {metrics['val_mse']:>10.6f} {metrics['val_mae']:>10.6f} {metrics['val_r2']:>10.4f}")
        print(f"  {'Test':<12} {metrics['test_mse']:>10.6f} {metrics['test_mae']:>10.6f} {metrics['test_r2']:>10.4f}")
        
        # Feature importance
        print(f"\n  Top 10 Feature Importances:")
        importances = list(zip(self.feature_names, self.model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        for i, (name, imp) in enumerate(importances[:10], 1):
            print(f"    {i:2d}. {name:<35} {imp:.4f}")
        
        # Save model
        print("\n" + "=" * 60)
        print("SAVING MODEL")
        print("=" * 60)
        self._save_model()
        
        print("\n✅ Training complete!")
        
        return metrics
    
    def predict(self, data: Dict) -> Optional[float]:
        """
        Predict xT at the end of carrying from current position.
        
        Args:
            data: Dict with start_x, start_y, and player positions
        
        Returns:
            Predicted xT after carrying, or None if prediction fails
        """
        if not self.is_trained:
            print("Warning: Carry model not trained, returning None")
            return None
        
        # Extract features
        features = process_inference_input(data)
        if features is None:
            return None
        
        # Add sequence features (assume single carry for inference)
        features['carry_sequence_length'] = 1
        features['position_in_sequence'] = 0
        
        # Build feature vector in correct order
        try:
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]], dtype=np.float32)
            prediction = self.model.predict(X)[0]
            return float(prediction)
        except Exception as e:
            print(f"Carry prediction error: {e}")
            return None
    
    def calculate_carry_score(self, data: Dict) -> Optional[Dict]:
        """
        Calculate comprehensive carry score including risk/reward.
        
        Args:
            data: Dict with position and player data (ball_x/ball_y or start_x/start_y)
        
        Returns:
            Dict with:
                - predicted_xt: xT after carrying
                - current_xt: xT at current position
                - xt_gain: predicted_xt - current_xt
                - score: final carry score
        """
        # Normalize data format - map ball_x/ball_y to start_x/start_y
        normalized_data = data.copy()
        if 'start_x' not in normalized_data and 'ball_x' in normalized_data:
            normalized_data['start_x'] = normalized_data['ball_x']
        if 'start_y' not in normalized_data and 'ball_y' in normalized_data:
            normalized_data['start_y'] = normalized_data['ball_y']
        
        # Get current xT
        try:
            current_xt = self.xt_model.calculate_expected_threat(**normalized_data)
        except Exception as e:
            print(f"Error calculating current xT: {e}")
            return None
        
        if current_xt is None:
            return None
        
        # Get predicted xT after carrying
        predicted_xt = self.predict(normalized_data)
        if predicted_xt is None:
            print(f"Carry prediction returned None")
            return None
        
        # Calculate xT gain
        xt_gain = predicted_xt - current_xt
        
        # Score is simply the expected xT gain
        # Could add risk factors here later (e.g., probability of losing ball)
        score = xt_gain
        
        return {
            'predicted_xt': predicted_xt,
            'current_xt': current_xt,
            'xt_gain': xt_gain,
            'score': score,
        }
    
    def evaluate_carry_options(
        self, 
        data: Dict,
        carry_directions: List[Tuple[float, float]] = None
    ) -> List[Dict]:
        """
        Evaluate multiple carry direction options.
        
        Args:
            data: Current position and player data
            carry_directions: List of (dx, dy) tuples for carry directions
                             If None, uses default directions
        
        Returns:
            List of dicts with carry evaluations, sorted by score
        """
        if carry_directions is None:
            # Default: forward, forward-left, forward-right
            carry_directions = [
                (10, 0),    # Forward
                (8, 5),     # Forward-right
                (8, -5),    # Forward-left
                (5, 10),    # Wide right
                (5, -10),   # Wide left
            ]
        
        results = []
        base_x = data.get('start_x', 60)
        base_y = data.get('start_y', 40)
        
        for dx, dy in carry_directions:
            # Create hypothetical end position
            end_x = min(120, base_x + dx)
            end_y = max(0, min(80, base_y + dy))
            
            # Create data dict for this carry option
            carry_data = data.copy()
            carry_data['end_x'] = end_x
            carry_data['end_y'] = end_y
            
            result = self.calculate_carry_score(carry_data)
            if result:
                result['direction'] = (dx, dy)
                result['end_x'] = end_x
                result['end_y'] = end_y
                results.append(result)
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results


# ============================================================================
# MAIN
# ============================================================================

def train_carry_model(max_samples: int = None):
    """Train the carry model."""
    model = CarryModel()
    metrics = model.train(max_samples=max_samples)
    return model, metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or test carry model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--max-samples', type=int, default=None, help='Max training samples')
    parser.add_argument('--test', action='store_true', help='Test with sample data')
    
    args = parser.parse_args()
    
    if args.train:
        model, metrics = train_carry_model(max_samples=args.max_samples)
    elif args.test:
        model = CarryModel()
        if model.is_trained:
            # Test with sample data
            test_data = {
                'start_x': 70,
                'start_y': 40,
                'team_id': 1,
            }
            # Add dummy player positions
            for i in range(10):
                test_data[f'p{i}_x'] = 50 + i * 5
                test_data[f'p{i}_y'] = 30 + (i % 3) * 10
                test_data[f'p{i}_team'] = 1 if i < 5 else 0
            
            result = model.calculate_carry_score(test_data)
            print(f"\nTest prediction:")
            print(f"  Current xT: {result['current_xt']:.4f}")
            print(f"  Predicted xT after carry: {result['predicted_xt']:.4f}")
            print(f"  xT gain: {result['xt_gain']:.4f}")
            print(f"  Score: {result['score']:.4f}")
        else:
            print("Model not trained. Run with --train first.")
    else:
        print("Usage: python carryModel.py --train [--max-samples N]")
        print("       python carryModel.py --test")
