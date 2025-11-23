#!/usr/bin/env python3
"""
Multimodal Biometric Probability Analysis

This script analyzes the probability of identity verification when combining
two biometric modalities (face and iris) with given accuracy rates.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import pandas as pd

def calculate_multimodal_probabilities(accuracy_face: float, accuracy_iris: float, 
                                     genuine_prior: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Calculate probabilities for all combinations of face and iris predictions.
    
    Args:
        accuracy_face: Face recognition accuracy (0-1)
        accuracy_iris: Iris recognition accuracy (0-1)
        genuine_prior: Prior probability that a comparison is genuine (0-1)
        
    Returns:
        Dictionary with probabilities for each scenario
    """
    
    # For simplicity, assume equal True Positive Rate (TPR) and True Negative Rate (TNR)
    # In practice, these might differ
    tpr_face = accuracy_face  # Sensitivity
    tnr_face = accuracy_face  # Specificity
    tpr_iris = accuracy_iris
    tnr_iris = accuracy_iris
    
    # False rates
    fpr_face = 1 - tnr_face  # False Positive Rate
    fnr_face = 1 - tpr_face  # False Negative Rate
    fpr_iris = 1 - tnr_iris
    fnr_iris = 1 - tpr_iris
    
    impostor_prior = 1 - genuine_prior
    
    results = {}
    
    # Scenario 1: Both predict MATCH
    # P(Face=Match, Iris=Match | Genuine) * P(Genuine)
    p_both_match_given_genuine = tpr_face * tpr_iris * genuine_prior
    # P(Face=Match, Iris=Match | Impostor) * P(Impostor)  
    p_both_match_given_impostor = fpr_face * fpr_iris * impostor_prior
    # Total probability of both matching
    p_both_match = p_both_match_given_genuine + p_both_match_given_impostor
    
    # Posterior probability of genuine given both match (Bayes' theorem)
    p_genuine_given_both_match = p_both_match_given_genuine / p_both_match if p_both_match > 0 else 0
    
    results["both_match"] = {
        "p_genuine": p_genuine_given_both_match,
        "p_impostor": 1 - p_genuine_given_both_match,
        "total_probability": p_both_match
    }
    
    # Scenario 2: Both predict NO MATCH
    p_both_nomatch_given_genuine = fnr_face * fnr_iris * genuine_prior
    p_both_nomatch_given_impostor = tnr_face * tnr_iris * impostor_prior
    p_both_nomatch = p_both_nomatch_given_genuine + p_both_nomatch_given_impostor
    
    p_genuine_given_both_nomatch = p_both_nomatch_given_genuine / p_both_nomatch if p_both_nomatch > 0 else 0
    
    results["both_nomatch"] = {
        "p_genuine": p_genuine_given_both_nomatch,
        "p_impostor": 1 - p_genuine_given_both_nomatch,
        "total_probability": p_both_nomatch
    }
    
    # Scenario 3: Face MATCH, Iris NO MATCH
    p_face_match_iris_nomatch_given_genuine = tpr_face * fnr_iris * genuine_prior
    p_face_match_iris_nomatch_given_impostor = fpr_face * tnr_iris * impostor_prior
    p_face_match_iris_nomatch = p_face_match_iris_nomatch_given_genuine + p_face_match_iris_nomatch_given_impostor
    
    p_genuine_given_face_match_iris_nomatch = p_face_match_iris_nomatch_given_genuine / p_face_match_iris_nomatch if p_face_match_iris_nomatch > 0 else 0
    
    results["face_match_iris_nomatch"] = {
        "p_genuine": p_genuine_given_face_match_iris_nomatch,
        "p_impostor": 1 - p_genuine_given_face_match_iris_nomatch,
        "total_probability": p_face_match_iris_nomatch
    }
    
    # Scenario 4: Face NO MATCH, Iris MATCH
    p_face_nomatch_iris_match_given_genuine = fnr_face * tpr_iris * genuine_prior
    p_face_nomatch_iris_match_given_impostor = tnr_face * fpr_iris * impostor_prior
    p_face_nomatch_iris_match = p_face_nomatch_iris_match_given_genuine + p_face_nomatch_iris_match_given_impostor
    
    p_genuine_given_face_nomatch_iris_match = p_face_nomatch_iris_match_given_genuine / p_face_nomatch_iris_match if p_face_nomatch_iris_match > 0 else 0
    
    results["face_nomatch_iris_match"] = {
        "p_genuine": p_genuine_given_face_nomatch_iris_match,
        "p_impostor": 1 - p_genuine_given_face_nomatch_iris_match,
        "total_probability": p_face_nomatch_iris_match
    }
    
    return results

def analyze_accuracy_range(min_acc: float = 0.80, max_acc: float = 0.90, steps: int = 11):
    """Analyze probability combinations across a range of accuracies."""
    
    accuracies = np.linspace(min_acc, max_acc, steps)
    scenarios = ["both_match", "both_nomatch", "face_match_iris_nomatch", "face_nomatch_iris_match"]
    
    results_matrix = {}
    
    for scenario in scenarios:
        results_matrix[scenario] = {
            "p_genuine": np.zeros((len(accuracies), len(accuracies))),
            "p_impostor": np.zeros((len(accuracies), len(accuracies)))
        }
    
    for i, acc_face in enumerate(accuracies):
        for j, acc_iris in enumerate(accuracies):
            probs = calculate_multimodal_probabilities(acc_face, acc_iris)
            
            for scenario in scenarios:
                results_matrix[scenario]["p_genuine"][i, j] = probs[scenario]["p_genuine"]
                results_matrix[scenario]["p_impostor"][i, j] = probs[scenario]["p_impostor"]
    
    return results_matrix, accuracies

def plot_probability_heatmaps(results_matrix, accuracies, scenario: str):
    """Plot heatmaps for a specific scenario."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot P(Genuine | Scenario)
    im1 = ax1.imshow(results_matrix[scenario]["p_genuine"], 
                     extent=[accuracies[0], accuracies[-1], accuracies[0], accuracies[-1]],
                     origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_xlabel('Iris Accuracy')
    ax1.set_ylabel('Face Accuracy')
    ax1.set_title(f'P(Genuine | {scenario.replace("_", " ").title()})')
    plt.colorbar(im1, ax=ax1)
    
    # Add contour lines
    X, Y = np.meshgrid(accuracies, accuracies)
    contours1 = ax1.contour(X, Y, results_matrix[scenario]["p_genuine"], 
                           levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='black', alpha=0.5)
    ax1.clabel(contours1, inline=True, fontsize=8)
    
    # Plot P(Impostor | Scenario)
    im2 = ax2.imshow(results_matrix[scenario]["p_impostor"],
                     extent=[accuracies[0], accuracies[-1], accuracies[0], accuracies[-1]],
                     origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1)
    ax2.set_xlabel('Iris Accuracy')
    ax2.set_ylabel('Face Accuracy')
    ax2.set_title(f'P(Impostor | {scenario.replace("_", " ").title()})')
    plt.colorbar(im2, ax=ax2)
    
    # Add contour lines
    contours2 = ax2.contour(X, Y, results_matrix[scenario]["p_impostor"],
                           levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='black', alpha=0.5)
    ax2.clabel(contours2, inline=True, fontsize=8)
    
    plt.tight_layout()
    return fig

def main():
    """Main analysis function."""
    
    print("🔍 MULTIMODAL BIOMETRIC PROBABILITY ANALYSIS")
    print("=" * 60)
    
    # Specific case: 0.84 accuracy for both modalities
    accuracy_face = 0.84
    accuracy_iris = 0.84
    
    print(f"\n📊 Analysis for Face Accuracy: {accuracy_face:.1%}, Iris Accuracy: {accuracy_iris:.1%}")
    print("-" * 60)
    
    results = calculate_multimodal_probabilities(accuracy_face, accuracy_iris)
    
    # Create summary table
    scenarios = [
        ("both_match", "Both Predict MATCH"),
        ("both_nomatch", "Both Predict NO MATCH"), 
        ("face_match_iris_nomatch", "Face MATCH, Iris NO MATCH"),
        ("face_nomatch_iris_match", "Face NO MATCH, Iris MATCH")
    ]
    
    print(f"{'Scenario':<25} {'P(Genuine)':<12} {'P(Impostor)':<12} {'Likelihood':<12}")
    print("-" * 65)
    
    for scenario_key, scenario_desc in scenarios:
        p_genuine = results[scenario_key]["p_genuine"]
        p_impostor = results[scenario_key]["p_impostor"]
        likelihood = results[scenario_key]["total_probability"]
        
        print(f"{scenario_desc:<25} {p_genuine:<12.3f} {p_impostor:<12.3f} {likelihood:<12.3f}")
    
    # Key insights
    print(f"\n🎯 Key Insights:")
    print(f"   • When BOTH modalities predict MATCH:")
    print(f"     - Probability of genuine identity: {results['both_match']['p_genuine']:.1%}")
    print(f"     - This scenario occurs {results['both_match']['total_probability']:.1%} of the time")
    
    print(f"\n   • When BOTH modalities predict NO MATCH:")
    print(f"     - Probability of genuine identity: {results['both_nomatch']['p_genuine']:.1%}")
    print(f"     - This scenario occurs {results['both_nomatch']['total_probability']:.1%} of the time")
    
    print(f"\n   • When modalities DISAGREE:")
    face_match_iris_no = results['face_match_iris_nomatch']['p_genuine']
    face_no_iris_match = results['face_nomatch_iris_match']['p_genuine']
    print(f"     - Face MATCH, Iris NO MATCH: {face_match_iris_no:.1%} genuine")
    print(f"     - Face NO MATCH, Iris MATCH: {face_no_iris_match:.1%} genuine")
    
    # Calculate overall system accuracy if we use majority vote
    # Predict MATCH if at least one modality predicts MATCH
    p_system_match_given_genuine = (
        results['both_match']['total_probability'] * results['both_match']['p_genuine'] +
        results['face_match_iris_nomatch']['total_probability'] * results['face_match_iris_nomatch']['p_genuine'] +
        results['face_nomatch_iris_match']['total_probability'] * results['face_nomatch_iris_match']['p_genuine']
    )
    
    p_system_nomatch_given_genuine = results['both_nomatch']['total_probability'] * results['both_nomatch']['p_genuine']
    
    print(f"\n📈 System-Level Performance (OR logic - match if either matches):")
    
    # For OR logic: system predicts MATCH if either face OR iris predicts MATCH
    # True Positive Rate: P(Face=Match OR Iris=Match | Genuine)
    tpr_system = accuracy_face * accuracy_iris + accuracy_face * (1-accuracy_iris) + (1-accuracy_face) * accuracy_iris
    # True Negative Rate: P(Face=NoMatch AND Iris=NoMatch | Impostor)  
    tnr_system = (1-accuracy_face) * (1-accuracy_iris) / ((1-accuracy_face) * (1-accuracy_iris) + accuracy_face * accuracy_iris)
    
    # Simplified calculation for OR system
    # TPR = 1 - (1-TPR_face) * (1-TPR_iris) = 1 - 0.16 * 0.16 = 0.9744
    tpr_or = 1 - (1-accuracy_face) * (1-accuracy_iris)
    # TNR = TNR_face * TNR_iris = 0.84 * 0.84 = 0.7056  
    tnr_or = accuracy_face * accuracy_iris
    
    accuracy_or = (tpr_or + tnr_or) / 2  # Balanced accuracy
    
    print(f"   • OR System Accuracy: {accuracy_or:.1%}")
    print(f"   • True Positive Rate: {tpr_or:.1%}")
    print(f"   • True Negative Rate: {tnr_or:.1%}")
    
    # AND logic: system predicts MATCH only if BOTH predict MATCH
    tpr_and = accuracy_face * accuracy_iris  # 0.7056
    tnr_and = 1 - (1-accuracy_face) * (1-accuracy_iris)  # 0.9744
    accuracy_and = (tpr_and + tnr_and) / 2
    
    print(f"\n📈 System-Level Performance (AND logic - match only if both match):")
    print(f"   • AND System Accuracy: {accuracy_and:.1%}")
    print(f"   • True Positive Rate: {tpr_and:.1%}")
    print(f"   • True Negative Rate: {tnr_and:.1%}")
    
    # Generate and save heatmaps for different accuracy ranges
    print(f"\n📊 Generating probability heatmaps...")
    
    results_matrix, accuracies = analyze_accuracy_range(0.75, 0.95, 21)
    
    # Save plots for each scenario
    import os
    os.makedirs("multimodal_analysis", exist_ok=True)
    
    for scenario_key, scenario_desc in scenarios:
        fig = plot_probability_heatmaps(results_matrix, accuracies, scenario_key)
        fig.suptitle(f'Probability Analysis: {scenario_desc}', fontsize=16)
        fig.savefig(f"multimodal_analysis/{scenario_key}_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create summary comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (scenario_key, scenario_desc) in enumerate(scenarios):
        im = axes[i].imshow(results_matrix[scenario_key]["p_genuine"],
                           extent=[accuracies[0], accuracies[-1], accuracies[0], accuracies[-1]],
                           origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
        axes[i].set_xlabel('Iris Accuracy')
        axes[i].set_ylabel('Face Accuracy')
        axes[i].set_title(f'P(Genuine | {scenario_desc})')
        
        # Mark the 0.84, 0.84 point
        axes[i].plot(0.84, 0.84, 'r*', markersize=15, label='Your Case (84%, 84%)')
        axes[i].legend()
        
        # Add contour lines
        X, Y = np.meshgrid(accuracies, accuracies)
        contours = axes[i].contour(X, Y, results_matrix[scenario_key]["p_genuine"],
                                  levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='black', alpha=0.5)
        axes[i].clabel(contours, inline=True, fontsize=8)
        
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    fig.savefig("multimodal_analysis/summary_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Analysis complete! Results saved to 'multimodal_analysis/' directory")
    
    return results

if __name__ == "__main__":
    results = main()
