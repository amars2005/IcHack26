import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the feature importance from training output
features = [
    'nearest_opponent_to_target',
    'nearest_opponent_to_passer',
    'end_distance_from_goal',
    'start_distance_from_goal',
    'opponents_near_target',
    'is_long_ball',
    'pass_distance',
    'teammates_near_target',
    'opponents_in_corridor',
    'pass_forward_component',
    'in_final_third',
    'crossing_penalty_box',
    'is_backwards',
    'pass_lateral_component',
    'pass_angle'
]

coefficients = [
    -6.840640,
    6.755386,
    1.802840,
    -1.607487,
    -0.807436,
    -0.782933,
    0.725217,
    0.724191,
    -0.488701,
    -0.334389,
    -0.321892,
    -0.077567,
    0.071548,
    -0.012393,
    0.009520
]

# Create visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Color based on sign
colors = ['#d62728' if c < 0 else '#2ca02c' for c in coefficients]

# Create horizontal bar chart
y_pos = np.arange(len(features))
ax.barh(y_pos, coefficients, color=colors, alpha=0.7, edgecolor='black')

# Customize
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.set_xlabel('Coefficient (Impact on Pass Success)', fontsize=12, fontweight='bold')
ax.set_title('Pass Probability Model - Feature Importance\n(Logistic Regression Coefficients)', 
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', alpha=0.7, label='Positive (Increases Pass Success)'),
    Patch(facecolor='#d62728', alpha=0.7, label='Negative (Decreases Pass Success)')
]
ax.legend(handles=legend_elements, loc='lower right')

# Tight layout
plt.tight_layout()
plt.savefig('../../pass_model_feature_importance.png', dpi=300, bbox_inches='tight')
print("Feature importance plot saved to: pass_model_feature_importance.png")
plt.close()

# Create a summary stats image
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

stats_text = """
PASS PROBABILITY MODEL - SUMMARY

Model Performance:
  • ROC AUC Score: 0.885 (Excellent)
  • Brier Score: 0.138 (Good calibration)
  • Dataset: 331,982 passes
  • Success Rate: 80.72%

Top Success Factors:
  ✓ Space at passer (+6.76)
  ✓ Passes away from goal (+1.80)
  ✓ Teammate support (+0.72)

Top Failure Factors:
  ✗ Pressure at target (-6.84)
  ✗ Long balls (-0.78)
  ✗ Opponents in corridor (-0.49)

Key Insight:
  Defensive pressure AT THE TARGET is the 
  single most important factor in pass success.
"""

ax.text(0.5, 0.5, stats_text, 
        ha='center', va='center',
        fontsize=14, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('../../pass_model_summary.png', dpi=300, bbox_inches='tight')
print("Summary plot saved to: pass_model_summary.png")
plt.close()

print("\nVisualization complete!")
