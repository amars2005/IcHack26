"""
Neural Network Expected Threat Model for inference.
Provides the same interface as ExpectedThreatModelSimple for easy integration.
"""

import os
import sys

# Fix for Apple Silicon segfault - must be set BEFORE importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
# Force CPU to avoid MPS issues on Apple Silicon
torch.set_default_device('cpu')

import joblib
import numpy as np
import pandas as pd

# Add project root to path
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.generalpv.nn_model import ExpectedThreatNN, xTDataset
from backend.generalpv.nn_preprocessor import process_inference_input


class ExpectedThreatModelNN:
    """
    Neural Network Expected Threat model for inference.
    Drop-in replacement for ExpectedThreatModelSimple.
    """
    
    def __init__(self, model_path: str = None, calibrator_path: str = None, device: str = None):
        """
        Initialize the NN model for inference.
        
        Args:
            model_path: Path to the saved model checkpoint (.pt file)
            calibrator_path: Path to the calibrator (.pkl file)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model = None
        self.calibrator = None
        self.model_config = None
        self.max_teammates = 12
        self.max_opponents = 12
        
        # Force CPU to avoid segfaults on Apple Silicon
        self.device = torch.device('cpu')
        
        # Default paths
        if model_path is None:
            model_path = os.path.join(_MODULE_DIR, '..', 'models', 'xt_nn_model.pt')
        if calibrator_path is None:
            calibrator_path = os.path.join(_MODULE_DIR, '..', 'models', 'xt_nn_model_calibrator.pkl')
        
        self.model_path = model_path
        self.calibrator_path = calibrator_path
    
    def load_model(self, model_path: str = None):
        """Load the pre-trained model and calibrator."""
        if model_path is not None:
            self.model_path = model_path
            self.calibrator_path = model_path.replace('.pt', '_calibrator.pkl')
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        self.model_config = checkpoint['model_config']
        self.max_teammates = checkpoint.get('max_teammates', 12)
        self.max_opponents = checkpoint.get('max_opponents', 12)
        
        # Create model with saved config
        self.model = ExpectedThreatNN(**self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load calibrator
        if os.path.exists(self.calibrator_path):
            self.calibrator = joblib.load(self.calibrator_path)
            print(f"Model and calibrator loaded from {self.model_path}")
        else:
            print(f"Model loaded from {self.model_path} (no calibrator found)")
    
    def calculate_expected_threat(self, **kwargs) -> float:
        """
        Calculate xT for a single datapoint.
        
        Args:
            **kwargs: Raw event data matching chained dataset columns:
                - start_x, start_y: Ball position
                - p0_x, p0_y, p0_team, ..., p19_x, p19_y, p19_team: Player positions
                - keeper1_x, keeper1_y, keeper1_team, keeper2_x, keeper2_y, keeper2_team
        
        Returns:
            float: Expected threat value (probability chain leads to goal)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Parse ball position
        ball_x = kwargs.get('start_x', 60.0)
        ball_y = kwargs.get('start_y', 40.0)
        
        # Parse players into teammates and opponents
        teammates = []
        opponents = []
        team_keepers = []
        opponent_keepers = []
        
        # Process outfield players (p0-p19)
        for i in range(20):
            px = kwargs.get(f'p{i}_x')
            py = kwargs.get(f'p{i}_y')
            pteam = kwargs.get(f'p{i}_team')
            
            if px is not None and py is not None and pteam is not None:
                player = {'x': float(px), 'y': float(py)}
                if pteam == 1:  # Attacking team
                    teammates.append(player)
                else:
                    opponents.append(player)
        
        # Process keepers
        for k in [1, 2]:
            kx = kwargs.get(f'keeper{k}_x')
            ky = kwargs.get(f'keeper{k}_y')
            kteam = kwargs.get(f'keeper{k}_team')
            
            if kx is not None and ky is not None and kteam is not None:
                keeper = {'x': float(kx), 'y': float(ky)}
                if kteam == 1:
                    team_keepers.append(keeper)
                else:
                    opponent_keepers.append(keeper)
        
        # Process through NN preprocessor
        sample = process_inference_input(
            ball_x, ball_y,
            teammates, opponents,
            team_keepers, opponent_keepers
        )
        
        if sample is None:
            print("Warning: Could not process input data")
            return 0.0
        
        # Create dataset item
        dataset = xTDataset([sample], self.max_teammates, self.max_opponents)
        item = dataset[0]
        
        # Add batch dimension and move to device
        ball_feat = item['ball_features'].unsqueeze(0).to(self.device)
        tm_feat = item['teammate_features'].unsqueeze(0).to(self.device)
        tm_mask = item['teammate_mask'].unsqueeze(0).to(self.device)
        opp_feat = item['opponent_features'].unsqueeze(0).to(self.device)
        opp_mask = item['opponent_mask'].unsqueeze(0).to(self.device)
        global_feat = item['global_features'].unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(ball_feat, tm_feat, tm_mask, opp_feat, opp_mask, global_feat)
            prob = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Apply calibration if available
        if self.calibrator is not None:
            prob = self.calibrator.predict([prob])[0]
        
        return float(prob)
    
    def generate_heatmap(self, grid_size: tuple = (24, 16), **kwargs) -> dict:
        """
        Generate an xT heatmap across the pitch for a given player configuration.
        
        Args:
            grid_size: (width_steps, height_steps) - resolution of the heatmap
            **kwargs: Player data (same format as calculate_expected_threat, but start_x/start_y ignored)
        
        Returns:
            dict with:
                - 'heatmap': 2D list of xT values [rows][cols]
                - 'x_coords': list of x coordinates
                - 'y_coords': list of y coordinates
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        width_steps, height_steps = grid_size
        
        # Create coordinate grids (pitch is 120x80)
        x_coords = [i * 120.0 / (width_steps - 1) for i in range(width_steps)]
        y_coords = [i * 80.0 / (height_steps - 1) for i in range(height_steps)]
        
        # Parse players ONCE (same for all ball positions)
        teammates = []
        opponents = []
        team_keepers = []
        opponent_keepers = []
        
        for i in range(20):
            px = kwargs.get(f'p{i}_x')
            py = kwargs.get(f'p{i}_y')
            pteam = kwargs.get(f'p{i}_team')
            
            if px is not None and py is not None and pteam is not None:
                player = {'x': float(px), 'y': float(py)}
                if pteam == 1:
                    teammates.append(player)
                else:
                    opponents.append(player)
        
        for k in [1, 2]:
            kx = kwargs.get(f'keeper{k}_x')
            ky = kwargs.get(f'keeper{k}_y')
            kteam = kwargs.get(f'keeper{k}_team')
            
            if kx is not None and ky is not None and kteam is not None:
                keeper = {'x': float(kx), 'y': float(ky)}
                if kteam == 1:
                    team_keepers.append(keeper)
                else:
                    opponent_keepers.append(keeper)
        
        # Process all ball positions
        all_samples = []
        for y in y_coords:
            for x in x_coords:
                sample = process_inference_input(
                    x, y,
                    teammates, opponents,
                    team_keepers, opponent_keepers
                )
                if sample is not None:
                    all_samples.append(sample)
                else:
                    # Fallback: use neutral values
                    all_samples.append(process_inference_input(
                        x, y, [], [], [], []
                    ))
        
        # Create dataset for batch processing
        dataset = xTDataset(all_samples, self.max_teammates, self.max_opponents)
        
        # Batch all items
        batch_size = len(all_samples)
        ball_feats = torch.stack([dataset[i]['ball_features'] for i in range(batch_size)]).to(self.device)
        tm_feats = torch.stack([dataset[i]['teammate_features'] for i in range(batch_size)]).to(self.device)
        tm_masks = torch.stack([dataset[i]['teammate_mask'] for i in range(batch_size)]).to(self.device)
        opp_feats = torch.stack([dataset[i]['opponent_features'] for i in range(batch_size)]).to(self.device)
        opp_masks = torch.stack([dataset[i]['opponent_mask'] for i in range(batch_size)]).to(self.device)
        global_feats = torch.stack([dataset[i]['global_features'] for i in range(batch_size)]).to(self.device)
        
        # Run batched inference
        with torch.no_grad():
            logits = self.model(ball_feats, tm_feats, tm_masks, opp_feats, opp_masks, global_feats)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Apply calibration if available
        if self.calibrator is not None:
            probs = self.calibrator.predict(probs)
        
        # Reshape into 2D grid (rows = y, cols = x)
        heatmap = []
        idx = 0
        for _ in y_coords:
            row = []
            for _ in x_coords:
                row.append(float(probs[idx]))
                idx += 1
            heatmap.append(row)
        
        return {
            'heatmap': heatmap,
            'x_coords': x_coords,
            'y_coords': y_coords
        }


# Quick test
if __name__ == "__main__":
    print("Testing ExpectedThreatModelNN...")
    
    model = ExpectedThreatModelNN()
    model.load_model()
    
    # Test with sample data (attacking team on left, ball in attacking third)
    test_data = {
        'start_x': 100,  # Ball near opponent's box
        'start_y': 40,   # Central
        # Attackers (team=1)
        'p0_x': 95, 'p0_y': 35, 'p0_team': 1,  # Striker
        'p1_x': 95, 'p1_y': 45, 'p1_team': 1,  # Striker
        'p2_x': 85, 'p2_y': 30, 'p2_team': 1,  # Winger
        'p3_x': 85, 'p3_y': 50, 'p3_team': 1,  # Winger
        'p4_x': 75, 'p4_y': 40, 'p4_team': 1,  # Mid
        'p5_x': 70, 'p5_y': 30, 'p5_team': 1,  # Mid
        'p6_x': 70, 'p6_y': 50, 'p6_team': 1,  # Mid
        'p7_x': 50, 'p7_y': 20, 'p7_team': 1,  # Defender
        'p8_x': 50, 'p8_y': 60, 'p8_team': 1,  # Defender
        'p9_x': 45, 'p9_y': 40, 'p9_team': 1,  # Defender
        # Defenders (team=0)
        'p10_x': 105, 'p10_y': 35, 'p10_team': 0,
        'p11_x': 105, 'p11_y': 45, 'p11_team': 0,
        'p12_x': 108, 'p12_y': 30, 'p12_team': 0,
        'p13_x': 108, 'p13_y': 50, 'p13_team': 0,
        'p14_x': 110, 'p14_y': 40, 'p14_team': 0,
        'p15_x': 112, 'p15_y': 35, 'p15_team': 0,
        'p16_x': 112, 'p16_y': 45, 'p16_team': 0,
        'p17_x': 100, 'p17_y': 20, 'p17_team': 0,
        'p18_x': 100, 'p18_y': 60, 'p18_team': 0,
        'p19_x': 90, 'p19_y': 40, 'p19_team': 0,
        # Keepers
        'keeper1_x': 10, 'keeper1_y': 40, 'keeper1_team': 1,  # Attacking keeper
        'keeper2_x': 118, 'keeper2_y': 40, 'keeper2_team': 0,  # Defending keeper
    }
    
    xT = model.calculate_expected_threat(**test_data)
    print(f"\nExpected Threat: {xT:.4f} ({xT*100:.2f}%)")
