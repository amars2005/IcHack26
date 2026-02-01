"""
Metric normalization for xG/xT model outputs.
Converts raw model scores to 0-1 range using z-score sigmoid transform.
"""
import numpy as np
import json
import os
import sys
import random
from dataclasses import dataclass

# path setup for imports
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

PITCH_LENGTH = 120
PITCH_WIDTH = 80


@dataclass
class MetricConfig:
    name: str
    mean: float = None
    std: float = None
    min_val: float = None
    max_val: float = None


# metrics we care about
DEFAULT_METRICS = {
    "xG": MetricConfig("xG"),
    "xT": MetricConfig("xT"),
    "target_xT": MetricConfig("target_xT"),
    "original_xT": MetricConfig("original_xT"),
    "opponent_xT": MetricConfig("opponent_xT"),
    "success_prob": MetricConfig("success_prob"),
    "reward": MetricConfig("reward"),
    "risk": MetricConfig("risk"),
    "score": MetricConfig("score"),
}


class MetricNormalizer:
    """Handles normalization of model output metrics to 0-1 scale."""
    
    def __init__(self, configs=None):
        self.configs = configs if configs else DEFAULT_METRICS.copy()
        self.calibrated = False

    def calibrate(self, data):
        """Compute stats from sample data for each metric."""
        for name, values in data.items():
            if not values:
                continue
            if name not in self.configs:
                self.configs[name] = MetricConfig(name=name)
            
            arr = np.array(values)
            cfg = self.configs[name]
            cfg.mean = float(np.mean(arr))
            cfg.std = float(np.std(arr)) if len(arr) > 1 else 1.0
            cfg.min_val = float(np.min(arr))
            cfg.max_val = float(np.max(arr))
        
        self.calibrated = True

    def normalize_value(self, metric_name, value):
        """
        Normalize a single value using sigmoid(z-score).
        Returns value in [0, 1] range where 0.5 = mean.
        """
        if metric_name not in self.configs:
            return value
        
        cfg = self.configs[metric_name]
        if cfg.std is None or cfg.mean is None:
            return value
        if cfg.std == 0:
            return 0.5
        
        z = (value - cfg.mean) / cfg.std
        return 1.0 / (1.0 + np.exp(-z))

    def save_calibration(self, filepath):
        """Save calibration data to json."""
        out = {
            "configs": {
                name: {
                    "name": cfg.name,
                    "mean": cfg.mean,
                    "std": cfg.std,
                    "min_val": cfg.min_val,
                    "max_val": cfg.max_val,
                }
                for name, cfg in self.configs.items()
            },
            "calibrated": self.calibrated
        }
        with open(filepath, 'w') as f:
            json.dump(out, f, indent=2)

    def load_calibration(self, filepath):
        """Load calibration data from json."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.calibrated = data.get("calibrated", False)
        for name, cfg_data in data.get("configs", {}).items():
            self.configs[name] = MetricConfig(
                name=cfg_data["name"],
                mean=cfg_data.get("mean"),
                std=cfg_data.get("std"),
                min_val=cfg_data.get("min_val"),
                max_val=cfg_data.get("max_val"),
            )


# singleton normalizer instance
_normalizer = None

def get_normalizer():
    global _normalizer
    if _normalizer is None:
        _normalizer = MetricNormalizer()
        calib_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "metric_calibration.json"
        )
        if os.path.exists(calib_path):
            _normalizer.load_calibration(calib_path)
    return _normalizer


def normalize_value(metric_name, value):
    """Quick access to normalize a single metric value."""
    return get_normalizer().normalize_value(metric_name, value)


# ---- calibration script stuff below ----

def _rand_pos(x_min=0, x_max=PITCH_LENGTH, y_min=0, y_max=PITCH_WIDTH):
    return {"x": random.uniform(x_min, x_max), "y": random.uniform(y_min, y_max)}


def _make_scenario():
    """Generate random player positions for calibration."""
    attackers = [{"id": str(i + 2), **_rand_pos()} for i in range(10)]
    defenders = [{"id": f"d{i + 2}", **_rand_pos()} for i in range(10)]
    keepers = [
        {"x": random.uniform(2, 8), "y": random.uniform(35, 45)},
        {"x": random.uniform(112, 118), "y": random.uniform(35, 45)}
    ]
    
    ball_idx = random.randint(0, 9)
    ball_id = attackers[ball_idx]["id"]
    ball_pos = {"x": attackers[ball_idx]["x"], "y": attackers[ball_idx]["y"]}
    
    # build data dict format expected by models
    data_dict = {"ball_x": ball_pos["x"], "ball_y": ball_pos["y"]}
    for i, att in enumerate(attackers):
        data_dict[f"p_{i}_x"] = att["x"]
        data_dict[f"p_{i}_y"] = att["y"]
        data_dict[f"p_{i}_team"] = 1
    for i, d in enumerate(defenders):
        data_dict[f"p_{i+10}_x"] = d["x"]
        data_dict[f"p_{i+10}_y"] = d["y"]
        data_dict[f"p_{i+10}_team"] = 0
    data_dict["keeper_1_x"] = keepers[0]["x"]
    data_dict["keeper_1_y"] = keepers[0]["y"]
    data_dict["keeper_1_team"] = 1
    data_dict["keeper_2_x"] = keepers[1]["x"]
    data_dict["keeper_2_y"] = keepers[1]["y"]
    data_dict["keeper_2_team"] = 0
    
    return {
        "ball_position": ball_pos,
        "attackers": attackers,
        "defenders": defenders,
        "keepers": keepers,
        "ball_id": ball_id,
        "data_dict": data_dict
    }


def run_calibration(n_scenarios=500):
    """Run calibration by generating random scenarios and collecting metric distributions."""
    from backend.generalpv.passScoreModel import PassScoreModel
    from backend.generalpv.xg import ExpectedGoalModel
    
    print("loading models...")
    pass_model = PassScoreModel()
    pass_model.load_models()
    
    xg_model = ExpectedGoalModel(skip_training=True)
    # try a few paths for the xg model
    for p in [
        os.path.join(os.path.dirname(__file__), "..", "models", "xg_model_360.pkl"),
        os.path.join(_PROJECT_ROOT, "models", "xg_model_360.pkl"),
    ]:
        if os.path.exists(p):
            xg_model.load_model(p)
            break
    
    metrics = {k: [] for k in ["xG", "xT", "target_xT", "original_xT", 
                                "opponent_xT", "success_prob", "reward", "risk", "score"]}
    
    print(f"running {n_scenarios} scenarios...")
    for i in range(n_scenarios):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_scenarios}")
        
        scenario = _make_scenario()
        try:
            scores = pass_model.calculate_pass_scores(
                ball_position=scenario["ball_position"],
                attackers=scenario["attackers"],
                defenders=scenario["defenders"],
                keepers=scenario["keepers"],
                ball_id=scenario["ball_id"],
                data_dict=scenario["data_dict"]
            )
            
            for m in scores.values():
                metrics["target_xT"].append(m["target_xT"])
                metrics["original_xT"].append(m["original_xT"])
                metrics["opponent_xT"].append(m["opponent_xT"])
                metrics["success_prob"].append(m["success_prob"])
                metrics["reward"].append(m["reward"])
                metrics["risk"].append(m["risk"])
                metrics["score"].append(m["score"])
            
            if xg_model.model:
                metrics["xG"].append(xg_model.calculate_expected_goal(**scenario["data_dict"]))
            
            metrics["xT"].append(pass_model.get_current_xT(
                scenario["ball_position"], 
                scenario["attackers"], 
                scenario["defenders"], 
                scenario["keepers"]
            ))
        except Exception as e:
            print(f"  scenario {i} failed: {e}")
    
    normalizer = MetricNormalizer()
    normalizer.calibrate(metrics)
    
    print("\nresults:")
    for name, vals in metrics.items():
        if vals:
            cfg = normalizer.configs[name]
            print(f"  {name}: mean={cfg.mean:.4f}, std={cfg.std:.4f}")
    
    return normalizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--scenarios", type=int, default=200)
    parser.add_argument("-s", "--save", action="store_true")
    args = parser.parse_args()
    
    norm = run_calibration(args.scenarios)
    
    if args.save:
        out_path = os.path.join(os.path.dirname(__file__), "..", "models", "metric_calibration.json")
        norm.save_calibration(out_path)
        print(f"\nsaved to {out_path}")
