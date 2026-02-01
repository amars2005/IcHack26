import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import joblib
from passProbabilityModel import PassProbabilityModel

# Set random seed for reproducibility
np.random.seed(42)

def generate_random_passes(n_passes=30):
    """Generate random pass scenarios with varying difficulty levels."""
    passes = []
    
    for i in range(n_passes):
        # Random start position (avoid extreme edges)
        start_x = np.random.uniform(10, 110)
        start_y = np.random.uniform(10, 70)
        
        # Random pass direction and distance
        # Mix of short, medium, and long passes
        pass_type = np.random.choice(['short', 'medium', 'long'], p=[0.5, 0.3, 0.2])
        if pass_type == 'short':
            distance = np.random.uniform(3, 15)
        elif pass_type == 'medium':
            distance = np.random.uniform(15, 30)
        else:
            distance = np.random.uniform(30, 50)
        
        angle = np.random.uniform(-np.pi, np.pi)
        end_x = start_x + distance * np.cos(angle)
        end_y = start_y + distance * np.sin(angle)
        
        # Clip to pitch boundaries
        end_x = np.clip(end_x, 0, 120)
        end_y = np.clip(end_y, 0, 80)
        
        # Generate random player positions (20 players)
        player_positions = {}
        team_id = 909  # Arbitrary team ID
        
        for p_idx in range(20):
            # Randomly distribute players across the pitch
            # More players in midfield
            px = np.random.uniform(20, 100)
            py = np.random.uniform(5, 75)
            
            # Roughly half teammates, half opponents
            p_team = 1 if p_idx < 10 else 0
            
            player_positions[f'p{p_idx}_x'] = px
            player_positions[f'p{p_idx}_y'] = py
            player_positions[f'p{p_idx}_team'] = p_team
        
        pass_data = {
            'pass_id': i,
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'team_id': team_id,
            **player_positions
        }
        
        passes.append(pass_data)
    
    return passes

def calculate_pass_probabilities(model, passes):
    """Calculate success probability for each pass."""
    results = []
    
    for pass_data in passes:
        # Calculate probability
        prob = model.calculate_pass_probability(**pass_data)
        
        result = {
            'pass_id': pass_data['pass_id'],
            'start_x': pass_data['start_x'],
            'start_y': pass_data['start_y'],
            'end_x': pass_data['end_x'],
            'end_y': pass_data['end_y'],
            'success_probability': prob
        }
        results.append(result)
    
    return pd.DataFrame(results)

def draw_pitch(ax):
    """Draw football pitch markings."""
    lc = 'white'
    lw = 1.5
    
    # Outline & Center Line
    ax.plot([0, 0], [0, 80], color=lc, linewidth=lw)
    ax.plot([120, 120], [0, 80], color=lc, linewidth=lw)
    ax.plot([0, 120], [0, 0], color=lc, linewidth=lw)
    ax.plot([0, 120], [80, 80], color=lc, linewidth=lw)
    ax.plot([60, 60], [0, 80], color=lc, linewidth=lw)
    
    # Center Circle
    circle = mpatches.Circle((60, 40), 10, color=lc, fill=False, linewidth=lw)
    ax.add_patch(circle)
    ax.scatter(60, 40, color=lc, s=15)
    
    # Penalty Areas
    # Left
    ax.plot([18, 18], [18, 62], color=lc, linewidth=lw)
    ax.plot([0, 18], [18, 18], color=lc, linewidth=lw)
    ax.plot([0, 18], [62, 62], color=lc, linewidth=lw)
    # Right
    ax.plot([102, 102], [18, 62], color=lc, linewidth=lw)
    ax.plot([120, 102], [18, 18], color=lc, linewidth=lw)
    ax.plot([120, 102], [62, 62], color=lc, linewidth=lw)
    
    # 6-Yard Boxes
    # Left
    ax.plot([6, 6], [30, 50], color=lc, linewidth=lw)
    ax.plot([0, 6], [30, 30], color=lc, linewidth=lw)
    ax.plot([0, 6], [50, 50], color=lc, linewidth=lw)
    # Right
    ax.plot([114, 114], [30, 50], color=lc, linewidth=lw)
    ax.plot([120, 114], [30, 30], color=lc, linewidth=lw)
    ax.plot([120, 114], [50, 50], color=lc, linewidth=lw)
    
    # Penalty Spots
    ax.scatter(12, 40, color=lc, s=20)
    ax.scatter(108, 40, color=lc, s=20)
    
    ax.set_xlim(-5, 125)
    ax.set_ylim(-5, 85)
    ax.set_aspect('equal')
    ax.axis('off')

def visualize_passes(results_df, title="Pass Success Probability Predictions"):
    """Visualize passes on a pitch with color-coded success probabilities."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor('#2d5f3f')  # Grass green
    fig.patch.set_facecolor('#1a1a1a')
    
    draw_pitch(ax)
    
    # Sort by probability so highest probability passes are drawn last
    results_df = results_df.sort_values('success_probability')
    
    # Create colormap (red = low probability, green = high probability)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#ff0000', '#ff6600', '#ffcc00', '#99ff00', '#00ff00']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('pass_prob', colors_list, N=n_bins)
    
    # Draw passes as arrows
    for idx, row in results_df.iterrows():
        prob = row['success_probability']
        color = cmap(prob)
        
        # Draw arrow
        arrow = FancyArrowPatch(
            (row['start_x'], row['start_y']),
            (row['end_x'], row['end_y']),
            arrowstyle='->,head_width=0.8,head_length=1.2',
            color=color,
            linewidth=2.5,
            alpha=0.8,
            zorder=10
        )
        ax.add_patch(arrow)
        
        # Draw start point
        ax.scatter(row['start_x'], row['start_y'], 
                  color='white', s=60, zorder=11, 
                  edgecolors='black', linewidths=1)
        
        # Draw end point
        ax.scatter(row['end_x'], row['end_y'], 
                  color=color, s=100, zorder=11, 
                  edgecolors='black', linewidths=1.5, marker='o')
        
        # Add probability label near end point
        ax.text(row['end_x'] + 2, row['end_y'] + 2, 
               f"{prob:.0%}", 
               fontsize=8, fontweight='bold',
               color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
               zorder=12)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                       pad=0.02, shrink=0.6, aspect=30)
    cbar.set_label('Pass Success Probability', fontsize=12, 
                   fontweight='bold', color='white')
    cbar.ax.tick_params(labelsize=10, colors='white')
    
    # Title and stats
    avg_prob = results_df['success_probability'].mean()
    min_prob = results_df['success_probability'].min()
    max_prob = results_df['success_probability'].max()
    
    ax.set_title(
        f"{title}\n"
        f"Average: {avg_prob:.1%} | Min: {min_prob:.1%} | Max: {max_prob:.1%}",
        fontsize=16, fontweight='bold', color='white', pad=20
    )
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Pass Start',
                  markerfacecolor='white', markeredgecolor='black', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Pass Target',
                  markerfacecolor='#00ff00', markeredgecolor='black', markersize=10),
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
             fontsize=10, facecolor='#1a1a1a', edgecolor='white',
             labelcolor='white')
    
    plt.tight_layout()
    return fig, ax

def create_comparison_scenarios():
    """Create specific pass scenarios to compare difficulty levels."""
    scenarios = []
    
    # Scenario 1: Easy short pass with space
    scenarios.append({
        'name': 'Easy: Short pass with space',
        'start_x': 50, 'start_y': 40,
        'end_x': 60, 'end_y': 42,
        'team_id': 909,
        # Opponents far away
        **{f'p{i}_x': np.random.uniform(80, 110) for i in range(10)},
        **{f'p{i}_y': np.random.uniform(10, 70) for i in range(10)},
        **{f'p{i}_team': 0 for i in range(10)},
        # Teammates nearby
        **{f'p{i}_x': np.random.uniform(45, 65) for i in range(10, 20)},
        **{f'p{i}_y': np.random.uniform(35, 45) for i in range(10, 20)},
        **{f'p{i}_team': 1 for i in range(10, 20)},
    })
    
    # Scenario 2: Risky long ball
    scenarios.append({
        'name': 'Hard: Long ball under pressure',
        'start_x': 30, 'start_y': 40,
        'end_x': 95, 'end_y': 35,
        'team_id': 909,
        # Opponents near target
        **{f'p{i}_x': np.random.uniform(90, 100) for i in range(10)},
        **{f'p{i}_y': np.random.uniform(30, 45) for i in range(10)},
        **{f'p{i}_team': 0 for i in range(10)},
        # Teammates far from target
        **{f'p{i}_x': np.random.uniform(25, 40) for i in range(10, 20)},
        **{f'p{i}_y': np.random.uniform(35, 50) for i in range(10, 20)},
        **{f'p{i}_team': 1 for i in range(10, 20)},
    })
    
    # Scenario 3: Through ball into space
    scenarios.append({
        'name': 'Medium: Through ball into space',
        'start_x': 60, 'start_y': 40,
        'end_x': 85, 'end_y': 45,
        'team_id': 909,
        # Opponents behind the pass
        **{f'p{i}_x': np.random.uniform(50, 70) for i in range(10)},
        **{f'p{i}_y': np.random.uniform(30, 50) for i in range(10)},
        **{f'p{i}_team': 0 for i in range(10)},
        # One teammate near target
        'p10_x': 87, 'p10_y': 46, 'p10_team': 1,
        **{f'p{i}_x': np.random.uniform(55, 65) for i in range(11, 20)},
        **{f'p{i}_y': np.random.uniform(35, 45) for i in range(11, 20)},
        **{f'p{i}_team': 1 for i in range(11, 20)},
    })
    
    return scenarios

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PASS PROBABILITY MODEL")
    print("=" * 60)
    
    # Load the trained model
    print("\nLoading trained model...")
    model = PassProbabilityModel(skip_training=True)
    model.load_model("../../models/pass_probability_model.pkl")
    
    # Generate random passes
    print("\nGenerating random pass scenarios...")
    random_passes = generate_random_passes(n_passes=30)
    
    # Calculate probabilities
    print("Calculating success probabilities...")
    results = calculate_pass_probabilities(model, random_passes)
    
    # Display statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total passes analyzed: {len(results)}")
    print(f"Average success probability: {results['success_probability'].mean():.1%}")
    print(f"Min probability: {results['success_probability'].min():.1%}")
    print(f"Max probability: {results['success_probability'].max():.1%}")
    print(f"Std deviation: {results['success_probability'].std():.3f}")
    
    # Show top 5 easiest and hardest passes
    print("\n--- Top 5 Easiest Passes ---")
    easiest = results.nlargest(5, 'success_probability')
    for idx, row in easiest.iterrows():
        dist = np.sqrt((row['end_x'] - row['start_x'])**2 + (row['end_y'] - row['start_y'])**2)
        print(f"Pass {row['pass_id']}: {row['success_probability']:.1%} "
              f"(distance: {dist:.1f}m, from ({row['start_x']:.1f}, {row['start_y']:.1f}) "
              f"to ({row['end_x']:.1f}, {row['end_y']:.1f}))")
    
    print("\n--- Top 5 Hardest Passes ---")
    hardest = results.nsmallest(5, 'success_probability')
    for idx, row in hardest.iterrows():
        dist = np.sqrt((row['end_x'] - row['start_x'])**2 + (row['end_y'] - row['start_y'])**2)
        print(f"Pass {row['pass_id']}: {row['success_probability']:.1%} "
              f"(distance: {dist:.1f}m, from ({row['start_x']:.1f}, {row['start_y']:.1f}) "
              f"to ({row['end_x']:.1f}, {row['end_y']:.1f}))")
    
    # Visualize
    print("\nCreating visualization...")
    fig, ax = visualize_passes(results, title="Random Pass Scenarios - Success Probability")
    plt.savefig("../../pass_probability_visualization.png", dpi=300, bbox_inches='tight')
    print("Saved: pass_probability_visualization.png")
    plt.close()
    
    # Test specific scenarios
    print("\n" + "=" * 60)
    print("TESTING SPECIFIC SCENARIOS")
    print("=" * 60)
    scenarios = create_comparison_scenarios()
    scenario_results = []
    
    for scenario in scenarios:
        name = scenario.pop('name')
        prob = model.calculate_pass_probability(**scenario)
        print(f"\n{name}: {prob:.1%}")
        
        scenario_results.append({
            'pass_id': len(scenario_results),
            'start_x': scenario['start_x'],
            'start_y': scenario['start_y'],
            'end_x': scenario['end_x'],
            'end_y': scenario['end_y'],
            'success_probability': prob
        })
    
    # Visualize scenarios
    scenario_df = pd.DataFrame(scenario_results)
    fig, ax = visualize_passes(scenario_df, title="Specific Pass Scenarios Comparison")
    plt.savefig("../../pass_scenarios_comparison.png", dpi=300, bbox_inches='tight')
    print("\nSaved: pass_scenarios_comparison.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • pass_probability_visualization.png")
    print("  • pass_scenarios_comparison.png")
