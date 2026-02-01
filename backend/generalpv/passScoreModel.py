"""
Pass Score Model - Calculates comprehensive pass evaluation metrics.

This model combines:
- xT (Expected Threat) at target position using the NN model
- Pass success probability
- Risk assessment (opponent's xT if pass is intercepted)
- Overall pass score using expected value calculation

Formula:
  score = p_succ × (target_xT - original_xT) - (1 - p_succ) × (original_xT + opponent_xT_at_interception)
"""

import os
import math
from typing import Dict, List, Tuple, Optional, Any

# Module paths
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))


class PassScoreModel:
    """
    Comprehensive pass evaluation model combining xT, pass probability, and risk assessment.
    """
    
    def __init__(self):
        self.xt_model = None
        self.pass_model = None
        self._models_loaded = False
    
    def load_models(self):
        """Load all required models (xT NN and pass probability)."""
        if self._models_loaded:
            return
            
        from backend.generalpv.expectedThreatModelNN import ExpectedThreatModelNN
        from backend.generalpv.passProbabilityModel import PassProbabilityModel
        
        # Load xT Neural Network Model
        self.xt_model = ExpectedThreatModelNN()
        self.xt_model.load_model()
        
        # Load Pass Probability Model
        self.pass_model = PassProbabilityModel(skip_training=True)
        pass_model_path = os.path.join(_PROJECT_ROOT, "models/pass_probability_model.pkl")
        self.pass_model.load_model(pass_model_path)
        
        self._models_loaded = True
        print("PassScoreModel: All models loaded successfully")
    
    def _build_xt_data_dict(
        self, 
        ball_x: float, 
        ball_y: float, 
        attackers: List[Dict], 
        defenders: List[Dict], 
        keepers: List[Dict]
    ) -> Dict[str, Any]:
        """Build data dict for xT model with given ball position."""
        xt_dict = {
            "start_x": ball_x,
            "start_y": ball_y,
        }
        # Attackers: p0 through p9
        for idx in range(len(attackers)):
            xt_dict[f"p{idx}_x"] = attackers[idx]["x"]
            xt_dict[f"p{idx}_y"] = attackers[idx]["y"]
            xt_dict[f'p{idx}_team'] = 1
        # Defenders: p10 through p19
        for idx in range(len(defenders)):
            xt_dict[f"p{idx+10}_x"] = defenders[idx]["x"]
            xt_dict[f"p{idx+10}_y"] = defenders[idx]["y"]
            xt_dict[f'p{idx+10}_team'] = 0
        # Keepers (note: NN model expects keeper1_x not keeper_1_x)
        xt_dict["keeper1_x"] = keepers[0]["x"]
        xt_dict["keeper1_y"] = keepers[0]["y"]
        xt_dict['keeper1_team'] = 1
        xt_dict["keeper2_x"] = keepers[1]["x"]
        xt_dict["keeper2_y"] = keepers[1]["y"]
        xt_dict['keeper2_team'] = 0
        return xt_dict
    
    def _build_flipped_xt_data_dict(
        self, 
        ball_x: float, 
        ball_y: float, 
        attackers: List[Dict], 
        defenders: List[Dict], 
        keepers: List[Dict]
    ) -> Dict[str, Any]:
        """
        Build flipped data dict for opponent xT calculation.
        Flips coordinates (120-x, 80-y) and swaps team labels.
        This calculates xT from the opponent's perspective.
        """
        xt_dict = {
            "start_x": 120.0 - ball_x,
            "start_y": 80.0 - ball_y,
        }
        # Original attackers become opponents (team=0), positions flipped
        for idx in range(len(attackers)):
            xt_dict[f"p{idx+10}_x"] = 120.0 - attackers[idx]["x"]
            xt_dict[f"p{idx+10}_y"] = 80.0 - attackers[idx]["y"]
            xt_dict[f'p{idx+10}_team'] = 0  # Now opponents
        # Original defenders become attackers (team=1), positions flipped
        for idx in range(len(defenders)):
            xt_dict[f"p{idx}_x"] = 120.0 - defenders[idx]["x"]
            xt_dict[f"p{idx}_y"] = 80.0 - defenders[idx]["y"]
            xt_dict[f'p{idx}_team'] = 1  # Now attackers
        # Flip keepers and swap teams
        xt_dict["keeper1_x"] = 120.0 - keepers[1]["x"]
        xt_dict["keeper1_y"] = 80.0 - keepers[1]["y"]
        xt_dict['keeper1_team'] = 1  # Opponent keeper becomes team keeper
        xt_dict["keeper2_x"] = 120.0 - keepers[0]["x"]
        xt_dict["keeper2_y"] = 80.0 - keepers[0]["y"]
        xt_dict['keeper2_team'] = 0  # Team keeper becomes opponent keeper
        return xt_dict
    
    def _find_interception_point(
        self, 
        start_pos: Tuple[float, float], 
        end_pos: Tuple[float, float], 
        opponent_positions: List[Dict]
    ) -> Tuple[float, float]:
        """
        Find the interception point on the pass line.
        
        Algorithm:
        1. For each opponent, calculate perpendicular distance to the pass line segment
        2. Only consider opponents whose projection falls ON the segment (0 <= t <= 1)
        3. Return the projection point of the closest such opponent
        4. If no opponent projects onto the segment, return closest opponent to target
        
        Returns:
            (x, y) coordinates of the interception point
        """
        sx, sy = start_pos
        ex, ey = end_pos
        
        # Direction vector of the pass
        dx = ex - sx
        dy = ey - sy
        line_length_sq = dx * dx + dy * dy
        
        if line_length_sq == 0:
            # Start and end are same point
            return end_pos
        
        best_projection = None
        best_dist = float('inf')
        
        for opp in opponent_positions:
            ox, oy = opp['x'], opp['y']
            
            # Vector from start to opponent
            vx = ox - sx
            vy = oy - sy
            
            # Project onto pass line: t = (v · d) / |d|²
            t = (vx * dx + vy * dy) / line_length_sq
            
            # Check if projection falls on the segment [0, 1]
            if 0 <= t <= 1:
                # Projection point
                proj_x = sx + t * dx
                proj_y = sy + t * dy
                
                # Perpendicular distance from opponent to pass line
                perp_dist = math.sqrt((ox - proj_x)**2 + (oy - proj_y)**2)
                
                if perp_dist < best_dist:
                    best_dist = perp_dist
                    best_projection = (proj_x, proj_y)
        
        # If we found a valid projection, return it
        if best_projection is not None:
            return best_projection
        
        # Otherwise, find closest opponent to target
        closest_dist = float('inf')
        closest_pos = end_pos
        for opp in opponent_positions:
            dist = math.sqrt((opp['x'] - ex)**2 + (opp['y'] - ey)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_pos = (opp['x'], opp['y'])
        
        return closest_pos
    
    def calculate_pass_scores(
        self,
        ball_position: Dict[str, float],
        attackers: List[Dict],
        defenders: List[Dict],
        keepers: List[Dict],
        ball_id: str,
        data_dict: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate comprehensive pass scores for all possible pass targets.
        
        Args:
            ball_position: {"x": float, "y": float}
            attackers: List of attacker dicts with "x", "y", "id"
            defenders: List of defender dicts with "x", "y", "id"  
            keepers: List of keeper dicts with "x", "y"
            ball_id: ID of the player currently holding the ball
            data_dict: Original data dict for pass probability model
            
        Returns:
            Dict mapping player_id to pass metrics:
            {
                "player_id": {
                    "target_xT": float,          # xT at target position
                    "original_xT": float,        # xT at current ball position
                    "success_prob": float,       # Probability of successful pass
                    "opponent_xT": float,        # Opponent's xT if intercepted
                    "interception_point": tuple, # Where interception would occur
                    "reward": float,             # Expected gain if successful
                    "risk": float,               # Expected loss if failed
                    "score": float               # Net expected value
                }
            }
        """
        if not self._models_loaded:
            self.load_models()
        
        ball_pos = (ball_position['x'], ball_position['y'])
        
        # Calculate original xT at current ball position
        original_xt_dict = self._build_xt_data_dict(
            ball_position['x'], ball_position['y'],
            attackers, defenders, keepers
        )
        original_xT = self.xt_model.calculate_expected_threat(**original_xt_dict)
        
        results = {}
        
        for attacker in attackers:
            player_id = attacker["id"]
            
            # Skip the ball carrier
            if player_id == ball_id:
                continue
            
            target_pos = (attacker["x"], attacker["y"])
            
            # 1. Calculate pass success probability
            success_prob = self.pass_model.calculate_pass_probability(
                start_x=ball_pos[0], 
                start_y=ball_pos[1],
                end_x=target_pos[0], 
                end_y=target_pos[1],
                team_id=1, 
                **data_dict
            )
            
            # 2. Calculate target xT (xT at target position if pass succeeds)
            target_xt_dict = self._build_xt_data_dict(
                target_pos[0], target_pos[1],
                attackers, defenders, keepers
            )
            target_xT = self.xt_model.calculate_expected_threat(**target_xt_dict)
            
            # 3. Find interception point and calculate opponent xT
            interception_point = self._find_interception_point(
                ball_pos, target_pos, defenders
            )
            opponent_xt_dict = self._build_flipped_xt_data_dict(
                interception_point[0], interception_point[1],
                attackers, defenders, keepers
            )
            opponent_xT = self.xt_model.calculate_expected_threat(**opponent_xt_dict)
            
            # 4. Calculate pass score components
            # Reward: What we gain if pass succeeds
            reward = target_xT - original_xT
            
            # Risk: What we lose if pass fails (our position value + opponent gains threat)
            risk = original_xT + opponent_xT
            
            # Score: Expected value
            # score = p_succ × reward - (1 - p_succ) × risk
            score = success_prob * reward - 0.1*(1 - success_prob) * risk
            
            results[player_id] = {
                "target_xT": target_xT,
                "original_xT": original_xT,
                "success_prob": success_prob,
                "opponent_xT": opponent_xT,
                "interception_point": interception_point,
                "reward": reward,
                "risk": risk,
                "score": score
            }
        
        return results
    
    def get_current_xT(
        self,
        ball_position: Dict[str, float],
        attackers: List[Dict],
        defenders: List[Dict],
        keepers: List[Dict]
    ) -> float:
        """Calculate xT at the current ball position."""
        if not self._models_loaded:
            self.load_models()
            
        xt_dict = self._build_xt_data_dict(
            ball_position['x'], ball_position['y'],
            attackers, defenders, keepers
        )
        return self.xt_model.calculate_expected_threat(**xt_dict)
