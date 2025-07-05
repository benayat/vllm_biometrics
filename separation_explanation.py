#!/usr/bin/env python3
"""
Visual explanation of separation and discrimination concepts
"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_separation_concept():
    """Create a visual explanation of separation and discrimination."""
    
    # Your actual results
    genuine_mean = 0.8825
    genuine_std = 0.0414
    impostor_mean = 0.7519
    impostor_std = 0.0651
    separation = genuine_mean - impostor_mean
    discrimination = separation / (genuine_std + impostor_std)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Generate sample data
    x = np.linspace(0.6, 1.0, 1000)
    
    # Genuine distribution
    genuine_dist = np.exp(-0.5 * ((x - genuine_mean) / genuine_std)**2) / (genuine_std * np.sqrt(2 * np.pi))
    
    # Impostor distribution  
    impostor_dist = np.exp(-0.5 * ((x - impostor_mean) / impostor_std)**2) / (impostor_std * np.sqrt(2 * np.pi))
    
    # Plot 1: Your actual results
    ax1.fill_between(x, genuine_dist, alpha=0.6, color='green', label=f'Genuine pairs\n(μ={genuine_mean:.4f}, σ={genuine_std:.4f})')
    ax1.fill_between(x, impostor_dist, alpha=0.6, color='red', label=f'Impostor pairs\n(μ={impostor_mean:.4f}, σ={impostor_std:.4f})')
    
    # Mark the separation
    ax1.axvline(genuine_mean, color='green', linestyle='--', alpha=0.8)
    ax1.axvline(impostor_mean, color='red', linestyle='--', alpha=0.8)
    ax1.annotate('', xy=(genuine_mean, 0.5), xytext=(impostor_mean, 0.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax1.text((genuine_mean + impostor_mean)/2, 0.6, f'Separation = {separation:.4f}', 
             ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Probability Density')
    ax1.set_title(f'Your Face Recognition Results\nDiscrimination Ratio = {discrimination:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Comparison scenarios
    scenarios = [
        {"name": "Poor (High Overlap)", "g_mean": 0.80, "g_std": 0.10, "i_mean": 0.75, "i_std": 0.10, "color": "red"},
        {"name": "Your Results (Good)", "g_mean": genuine_mean, "g_std": genuine_std, "i_mean": impostor_mean, "i_std": impostor_std, "color": "green"},
        {"name": "Excellent (No Overlap)", "g_mean": 0.90, "g_std": 0.02, "i_mean": 0.70, "i_std": 0.02, "color": "blue"}
    ]
    
    for i, scenario in enumerate(scenarios):
        sep = scenario["g_mean"] - scenario["i_mean"]
        disc = sep / (scenario["g_std"] + scenario["i_std"])
        
        # Offset for visualization
        offset = i * 0.3
        
        # Generate distributions
        g_dist = np.exp(-0.5 * ((x - scenario["g_mean"]) / scenario["g_std"])**2) / (scenario["g_std"] * np.sqrt(2 * np.pi))
        i_dist = np.exp(-0.5 * ((x - scenario["i_mean"]) / scenario["i_std"])**2) / (scenario["i_std"] * np.sqrt(2 * np.pi))
        
        ax2.plot(x, g_dist + offset, color=scenario["color"], linewidth=2, alpha=0.8, linestyle='-')
        ax2.plot(x, i_dist + offset, color=scenario["color"], linewidth=2, alpha=0.8, linestyle='--')
        
        # Labels
        ax2.text(0.95, offset + 0.15, f'{scenario["name"]}\nSep: {sep:.3f}, Disc: {disc:.2f}', 
                ha='right', va='bottom', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor=scenario["color"], alpha=0.2))
    
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Probability Density (Offset)')
    ax2.set_title('Comparison of Different Separation Scenarios\n(Solid=Genuine, Dashed=Impostor)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('separation_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print numerical explanation
    print("="*60)
    print("SEPARATION CONCEPT EXPLAINED")
    print("="*60)
    print(f"Your Results (no_projector_cosine):")
    print(f"├── Genuine pairs: {genuine_mean:.4f} ± {genuine_std:.4f}")
    print(f"├── Impostor pairs: {impostor_mean:.4f} ± {impostor_std:.4f}")
    print(f"├── Separation: {separation:.4f}")
    print(f"└── Discrimination: {discrimination:.4f}")
    print()
    print("What this means:")
    print(f"• On a 0-1 similarity scale, genuine pairs score {separation:.4f} points higher")
    print(f"• That's a {separation*100:.2f}% improvement over impostors")
    print(f"• With discrimination of {discrimination:.2f}, you have good separation")
    print(f"• Error rate would be low with proper threshold setting")
    print()
    print("Threshold Recommendation:")
    optimal_threshold = (genuine_mean + impostor_mean) / 2
    print(f"• Set threshold around {optimal_threshold:.4f}")
    print(f"• Genuine pairs > {optimal_threshold:.4f}: Accept")
    print(f"• Impostor pairs < {optimal_threshold:.4f}: Reject")

if __name__ == "__main__":
    visualize_separation_concept()
