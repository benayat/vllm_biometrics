#!/usr/bin/env python3
"""
Multi-Modal Embedding Analysis using Gemma Vision Model

This script provides comprehensive embedding analysis for both face and iris biometrics,
including dimension optimization, performance testing, and detailed result saving.
"""

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Gemma3ImageProcessor
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import time
import json
import argparse
from typing import List, Tuple, Dict, Any
import logging
from tqdm import tqdm

# Add the project root to the path for imports
sys.path.append(str(Path(__file__).parent))
from dataloader.load_iitd_dataset import LoadIITDDataset
from dataloader.load_lfw_dataset import LoadLFWDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch._dynamo.config.disable = True

# Model setup
MODEL_ID = "google/gemma-3-4b-it"

class MultiModalEmbeddingAnalyzer:
    """Comprehensive embedding analyzer for face and iris biometrics."""

    def __init__(self, modality: str = "iris"):
        """
        Initialize the analyzer.

        Args:
            modality: Either "iris" or "face"
        """
        self.modality = modality.lower()
        if self.modality not in ["iris", "face"]:
            raise ValueError("Modality must be either 'iris' or 'face'")

        print(f"🔧 Loading Gemma Vision Model for {self.modality} analysis...")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float32,  # Use float32 for compatibility
            low_cpu_mem_usage=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.processor2 = Gemma3ImageProcessor.from_pretrained(MODEL_ID)

        print("✅ Model loaded successfully!")

        # Load appropriate dataset
        print(f"📂 Loading {self.modality} dataset...")
        if self.modality == "iris":
            self.dataset = LoadIITDDataset()
        else:  # face
            self.dataset = LoadLFWDataset()

        print(f"✅ {self.modality.capitalize()} dataset loaded!")

    def get_embedding(self, image_path: str, method: str = "mean_pooling") -> np.ndarray:
        """Extract embedding from image using vision tower with different pooling strategies."""
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor2(images=image, return_tensors="pt")

            # Move inputs to device (let model handle dtype automatically)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Use the complete pipeline
                image_outputs = self.model.vision_tower(inputs['pixel_values'])
                selected_image_feature = image_outputs.last_hidden_state

                # Convert to float32 if needed to avoid BFloat16 issues
                if selected_image_feature.dtype == torch.bfloat16:
                    selected_image_feature = selected_image_feature.float()

                image_embeddings = self.model.multi_modal_projector(selected_image_feature)

                # Convert to float32 if needed
                if image_embeddings.dtype == torch.bfloat16:
                    image_embeddings = image_embeddings.float()

                if method == "mean_pooling":
                    # Original method - mean across spatial dimensions
                    image_embedding_vector = image_embeddings.mean(dim=1)
                elif method == "max_pooling":
                    # Max pooling - take maximum activation
                    image_embedding_vector = image_embeddings.max(dim=1)[0]
                elif method == "attention_pooling":
                    # Attention-weighted pooling
                    attention_weights = torch.softmax(image_embeddings.mean(dim=-1), dim=1)
                    image_embedding_vector = (image_embeddings * attention_weights.unsqueeze(-1)).sum(dim=1)
                elif method == "cls_token":
                    # Use first token (CLS-like)
                    image_embedding_vector = image_embeddings[:, 0, :]
                elif method == "std_pooling":
                    # Standard deviation pooling for capturing variation
                    mean_pooled = image_embeddings.mean(dim=1)
                    std_pooled = image_embeddings.std(dim=1)
                    image_embedding_vector = torch.cat([mean_pooled, std_pooled], dim=-1)
                elif method == "no_projector":
                    # Skip multi-modal projector, use raw vision features
                    image_embedding_vector = selected_image_feature.mean(dim=1)
                else:
                    # Default to mean pooling
                    image_embedding_vector = image_embeddings.mean(dim=1)

            # Convert to numpy and normalize
            embedding_np = image_embedding_vector.cpu().numpy().flatten()

            # Normalize to unit vector
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                embedding_np = embedding_np / norm

            return embedding_np

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, metric: str = "cosine") -> float:
        """Compute similarity between two embeddings using different metrics."""
        if embedding1 is None or embedding2 is None:
            return 0.0

        if metric == "cosine":
            # Cosine similarity (default)
            similarity = np.dot(embedding1, embedding2)
        elif metric == "euclidean":
            # Euclidean distance (convert to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
        elif metric == "manhattan":
            # Manhattan distance (convert to similarity)
            distance = np.sum(np.abs(embedding1 - embedding2))
            similarity = 1.0 / (1.0 + distance)
        elif metric == "correlation":
            # Pearson correlation
            similarity = np.corrcoef(embedding1, embedding2)[0, 1]
            if np.isnan(similarity):
                similarity = 0.0
        else:
            # Default to cosine
            similarity = np.dot(embedding1, embedding2)

        return float(similarity)

    def test_multiple_methods(self, num_genuine: int = 20, num_impostor: int = 20) -> Dict[str, Any]:
        """Test multiple embedding and similarity methods."""
        print(f"\n🔬 COMPREHENSIVE {self.modality.upper()} EMBEDDING TEST")
        print("=" * 60)

        # Get test pairs
        if hasattr(self.dataset, 'genuine_pairs') and hasattr(self.dataset, 'impostor_pairs'):
            genuine_pairs = self.dataset.genuine_pairs[:num_genuine]
            impostor_pairs = self.dataset.impostor_pairs[:num_impostor]
        else:
            # For IITD dataset, get balanced sample
            test_pairs = self.dataset.get_balanced_sample(num_genuine, num_impostor)
            genuine_pairs = [pair for pair in test_pairs if pair[2] == 1]
            impostor_pairs = [pair for pair in test_pairs if pair[2] == 0]

        print(f"Testing with {len(genuine_pairs)} genuine and {len(impostor_pairs)} impostor pairs")

        # Test different embedding methods
        embedding_methods = [
            "mean_pooling", "max_pooling", "attention_pooling",
            "cls_token", "std_pooling", "no_projector"
        ]

        # Test different similarity metrics
        similarity_metrics = ["cosine", "euclidean", "manhattan", "correlation"]

        results = {}

        for emb_method in embedding_methods:
            print(f"\n--- Testing Embedding Method: {emb_method} ---")

            for sim_metric in similarity_metrics:
                print(f"  Similarity Metric: {sim_metric}")

                genuine_similarities = []
                impostor_similarities = []

                # Test genuine pairs
                for img1, img2, _ in genuine_pairs:
                    emb1 = self.get_embedding(str(img1), method=emb_method)
                    emb2 = self.get_embedding(str(img2), method=emb_method)
                    if emb1 is not None and emb2 is not None:
                        sim = self.compute_similarity(emb1, emb2, metric=sim_metric)
                        genuine_similarities.append(sim)

                # Test impostor pairs
                for img1, img2, _ in impostor_pairs:
                    emb1 = self.get_embedding(str(img1), method=emb_method)
                    emb2 = self.get_embedding(str(img2), method=emb_method)
                    if emb1 is not None and emb2 is not None:
                        sim = self.compute_similarity(emb1, emb2, metric=sim_metric)
                        impostor_similarities.append(sim)

                if genuine_similarities and impostor_similarities:
                    avg_genuine = np.mean(genuine_similarities)
                    avg_impostor = np.mean(impostor_similarities)
                    std_genuine = np.std(genuine_similarities)
                    std_impostor = np.std(impostor_similarities)
                    separation = avg_genuine - avg_impostor

                    # Calculate discrimination ratio
                    discrimination_ratio = separation / (std_genuine + std_impostor + 1e-8)

                    results[f"{emb_method}_{sim_metric}"] = {
                        "avg_genuine": avg_genuine,
                        "avg_impostor": avg_impostor,
                        "separation": separation,
                        "discrimination_ratio": discrimination_ratio,
                        "std_genuine": std_genuine,
                        "std_impostor": std_impostor,
                        "genuine_count": len(genuine_similarities),
                        "impostor_count": len(impostor_similarities)
                    }

                    print(f"    Genuine: {avg_genuine:.4f}±{std_genuine:.4f}, Impostor: {avg_impostor:.4f}±{std_impostor:.4f}")
                    print(f"    Separation: {separation:.4f}, Discrimination: {discrimination_ratio:.4f}")

        # Find best method
        if results:
            best_method = max(results.keys(), key=lambda k: results[k]["discrimination_ratio"])
            best_result = results[best_method]

            print(f"\n🏆 BEST METHOD: {best_method}")
            print(f"   Discrimination ratio: {best_result['discrimination_ratio']:.4f}")
            print(f"   Separation: {best_result['separation']:.4f}")
            print(f"   Genuine: {best_result['avg_genuine']:.4f} ± {best_result['std_genuine']:.4f}")
            print(f"   Impostor: {best_result['avg_impostor']:.4f} ± {best_result['std_impostor']:.4f}")

        return results

    def analyze_embedding_dimensions(self, num_pairs: int = 40) -> Dict[str, Any]:
        """Analyze which embedding dimensions are most discriminative."""
        print(f"\n🔬 {self.modality.upper()} EMBEDDING DIMENSION ANALYSIS")
        print("=" * 60)

        # Get test pairs
        if hasattr(self.dataset, 'genuine_pairs') and hasattr(self.dataset, 'impostor_pairs'):
            genuine_pairs = self.dataset.genuine_pairs[:num_pairs//2]
            impostor_pairs = self.dataset.impostor_pairs[:num_pairs//2]
        else:
            # For IITD dataset
            test_pairs = self.dataset.get_balanced_sample(num_pairs//2, num_pairs//2)
            genuine_pairs = [pair for pair in test_pairs if pair[2] == 1]
            impostor_pairs = [pair for pair in test_pairs if pair[2] == 0]

        print(f"🧮 Extracting {self.modality} embeddings from {len(genuine_pairs)} genuine and {len(impostor_pairs)} impostor pairs...")

        # Extract all embeddings first
        genuine_embeddings = []
        impostor_embeddings = []

        # Get genuine pair embeddings
        for i, (img1, img2, _) in enumerate(tqdm(genuine_pairs, desc="Processing genuine pairs")):
            emb1 = self.get_embedding(str(img1), method="mean_pooling")
            emb2 = self.get_embedding(str(img2), method="mean_pooling")
            if emb1 is not None and emb2 is not None:
                genuine_embeddings.append((emb1, emb2))

        # Get impostor pair embeddings
        for i, (img1, img2, _) in enumerate(tqdm(impostor_pairs, desc="Processing impostor pairs")):
            emb1 = self.get_embedding(str(img1), method="mean_pooling")
            emb2 = self.get_embedding(str(img2), method="mean_pooling")
            if emb1 is not None and emb2 is not None:
                impostor_embeddings.append((emb1, emb2))

        if not genuine_embeddings or not impostor_embeddings:
            print("❌ Failed to extract sufficient embeddings")
            return {}

        # Convert to numpy arrays for analysis
        embedding_dim = len(genuine_embeddings[0][0])
        print(f"📊 {self.modality.capitalize()} embedding dimension: {embedding_dim}")

        # Compute per-dimension statistics
        genuine_diffs = []
        impostor_diffs = []

        # Calculate element-wise differences for each pair
        for emb1, emb2 in genuine_embeddings:
            diff = np.abs(emb1 - emb2)  # Absolute difference per dimension
            genuine_diffs.append(diff)

        for emb1, emb2 in impostor_embeddings:
            diff = np.abs(emb1 - emb2)  # Absolute difference per dimension
            impostor_diffs.append(diff)

        # Convert to arrays: [num_pairs, embedding_dim]
        genuine_diffs = np.array(genuine_diffs)
        impostor_diffs = np.array(impostor_diffs)

        # Calculate mean difference per dimension
        genuine_mean_diff = genuine_diffs.mean(axis=0)  # [embedding_dim]
        impostor_mean_diff = impostor_diffs.mean(axis=0)  # [embedding_dim]

        # Calculate discrimination score per dimension
        # Lower genuine diff and higher impostor diff = better discrimination
        discrimination_scores = impostor_mean_diff - genuine_mean_diff

        # Find most discriminative dimensions
        sorted_indices = np.argsort(discrimination_scores)[::-1]  # Descending order

        print(f"\n📈 {self.modality.capitalize()} Dimension Analysis Results:")
        print(f"   Top 10 most discriminative dimensions:")
        for i in range(min(10, len(sorted_indices))):
            dim = sorted_indices[i]
            score = discrimination_scores[dim]
            gen_diff = genuine_mean_diff[dim]
            imp_diff = impostor_mean_diff[dim]
            print(f"   Dim {dim:4d}: score={score:.6f} (genuine={gen_diff:.6f}, impostor={imp_diff:.6f})")

        return {
            'discrimination_scores': discrimination_scores.tolist(),
            'sorted_indices': sorted_indices.tolist(),
            'genuine_embeddings_count': len(genuine_embeddings),
            'impostor_embeddings_count': len(impostor_embeddings),
            'embedding_dim': embedding_dim,
            'top_10_dimensions': [
                {
                    'dimension': int(sorted_indices[i]),
                    'score': float(discrimination_scores[sorted_indices[i]]),
                    'genuine_diff': float(genuine_mean_diff[sorted_indices[i]]),
                    'impostor_diff': float(impostor_mean_diff[sorted_indices[i]])
                }
                for i in range(min(10, len(sorted_indices)))
            ]
        }

    def test_dimension_subsets(self, analysis_results: Dict[str, Any],
                             subset_sizes: List[int] = None) -> Dict[str, Any]:
        """Test performance using only subsets of most discriminative dimensions."""
        print(f"\n🎯 TESTING {self.modality.upper()} EMBEDDING SUBSETS")
        print("=" * 60)

        if not analysis_results or 'discrimination_scores' not in analysis_results:
            print("❌ No dimension analysis results provided")
            return {}

        discrimination_scores = np.array(analysis_results['discrimination_scores'])
        sorted_indices = np.array(analysis_results['sorted_indices'])
        full_dim = analysis_results['embedding_dim']

        # Default subset sizes if not provided
        if subset_sizes is None:
            subset_sizes = [5, 10, 20, 30, 50, 70, 90, 120, 150, 200, 300, 500]

        # Remove sizes that exceed the embedding dimension
        subset_sizes = [size for size in subset_sizes if size < full_dim]

        print(f"Testing {len(subset_sizes)} different subset sizes: {subset_sizes}")

        # Get fresh test pairs for subset testing
        if hasattr(self.dataset, 'genuine_pairs') and hasattr(self.dataset, 'impostor_pairs'):
            test_genuine = self.dataset.genuine_pairs[50:70]  # Different pairs from dimension analysis
            test_impostor = self.dataset.impostor_pairs[50:70]
        else:
            test_pairs = self.dataset.get_balanced_sample(20, 20)
            test_genuine = [pair for pair in test_pairs if pair[2] == 1]
            test_impostor = [pair for pair in test_pairs if pair[2] == 0]

        results = {}

        # Test different subset sizes
        for subset_size in tqdm(subset_sizes, desc="Testing subset sizes"):
            # Get indices of most discriminative dimensions
            top_indices = sorted_indices[:subset_size]

            genuine_similarities = []
            impostor_similarities = []

            # Test genuine pairs
            for img1, img2, _ in test_genuine:
                emb1 = self.get_embedding(str(img1), method="mean_pooling")
                emb2 = self.get_embedding(str(img2), method="mean_pooling")

                if emb1 is not None and emb2 is not None:
                    # Extract subset of dimensions
                    sub_emb1 = emb1[top_indices]
                    sub_emb2 = emb2[top_indices]

                    # Renormalize
                    sub_emb1 = sub_emb1 / (np.linalg.norm(sub_emb1) + 1e-8)
                    sub_emb2 = sub_emb2 / (np.linalg.norm(sub_emb2) + 1e-8)

                    # Compute similarity
                    similarity = np.dot(sub_emb1, sub_emb2)
                    genuine_similarities.append(similarity)

            # Test impostor pairs
            for img1, img2, _ in test_impostor:
                emb1 = self.get_embedding(str(img1), method="mean_pooling")
                emb2 = self.get_embedding(str(img2), method="mean_pooling")

                if emb1 is not None and emb2 is not None:
                    # Extract subset of dimensions
                    sub_emb1 = emb1[top_indices]
                    sub_emb2 = emb2[top_indices]

                    # Renormalize
                    sub_emb1 = sub_emb1 / (np.linalg.norm(sub_emb1) + 1e-8)
                    sub_emb2 = sub_emb2 / (np.linalg.norm(sub_emb2) + 1e-8)

                    # Compute similarity
                    similarity = np.dot(sub_emb1, sub_emb2)
                    impostor_similarities.append(similarity)

            # Calculate metrics
            if genuine_similarities and impostor_similarities:
                avg_genuine = np.mean(genuine_similarities)
                avg_impostor = np.mean(impostor_similarities)
                std_genuine = np.std(genuine_similarities)
                std_impostor = np.std(impostor_similarities)
                separation = avg_genuine - avg_impostor
                discrimination_ratio = separation / (std_genuine + std_impostor + 1e-8)

                results[subset_size] = {
                    'avg_genuine': avg_genuine,
                    'avg_impostor': avg_impostor,
                    'separation': separation,
                    'discrimination_ratio': discrimination_ratio,
                    'std_genuine': std_genuine,
                    'std_impostor': std_impostor,
                    'dimension_reduction_percent': (1 - subset_size/full_dim) * 100
                }

        # Find best subset size
        if results:
            best_subset = max(results.keys(), key=lambda k: results[k]["discrimination_ratio"])
            best_result = results[best_subset]

            # Find top 5 best performing subset sizes
            sorted_by_discrimination = sorted(results.items(), key=lambda x: x[1]["discrimination_ratio"], reverse=True)

            print(f"\n🏆 TOP 5 BEST {self.modality.upper()} SUBSET SIZES:")
            for i, (size, result) in enumerate(sorted_by_discrimination[:5]):
                print(f"   {i+1}. {size:4d} dims: discrimination={result['discrimination_ratio']:.6f}, reduction={result['dimension_reduction_percent']:.1f}%")

            print(f"\n🥇 OPTIMAL {self.modality.upper()} SUBSET: {best_subset} dimensions")
            print(f"   Discrimination ratio: {best_result['discrimination_ratio']:.6f}")
            print(f"   Separation: {best_result['separation']:.6f}")
            print(f"   Size reduction: {best_result['dimension_reduction_percent']:.1f}%")

            results['optimal_subset_size'] = best_subset
            results['optimal_dimensions'] = sorted_indices[:best_subset].tolist()

        return results

    def run_comprehensive_analysis(self,
                                 num_method_test_pairs: int = 40,
                                 num_dimension_pairs: int = 60,
                                 subset_sizes: List[int] = None) -> Dict[str, Any]:
        """Run comprehensive embedding analysis and save results."""

        print(f"\n🚀 STARTING COMPREHENSIVE {self.modality.upper()} ANALYSIS")
        print("=" * 70)

        start_time = time.time()

        # Step 1: Test multiple methods
        print("\n📊 Phase 1: Testing multiple embedding and similarity methods...")
        method_results = self.test_multiple_methods(
            num_genuine=num_method_test_pairs//2,
            num_impostor=num_method_test_pairs//2
        )

        # Step 2: Analyze embedding dimensions
        print("\n🔬 Phase 2: Analyzing embedding dimensions...")
        dimension_results = self.analyze_embedding_dimensions(num_pairs=num_dimension_pairs)

        # Step 3: Test dimension subsets
        print("\n🎯 Phase 3: Testing optimal dimension subsets...")
        subset_results = self.test_dimension_subsets(dimension_results, subset_sizes)

        total_time = time.time() - start_time

        # Compile comprehensive results
        comprehensive_results = {
            "analysis_info": {
                "modality": self.modality,
                "model_id": MODEL_ID,
                "total_analysis_time_seconds": total_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_info": self._get_dataset_info()
            },
            "method_comparison": {
                "results": method_results,
                "best_method": max(method_results.keys(), key=lambda k: method_results[k]["discrimination_ratio"]) if method_results else None,
                "test_pairs": num_method_test_pairs
            },
            "dimension_analysis": dimension_results,
            "subset_optimization": subset_results,
            "summary": self._generate_summary(method_results, dimension_results, subset_results)
        }

        # Save results
        self._save_results(comprehensive_results)

        # Print final summary
        self._print_final_summary(comprehensive_results, total_time)

        return comprehensive_results

    def _get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        if hasattr(self.dataset, 'get_statistics'):
            return self.dataset.get_statistics()
        else:
            # Basic info for datasets without get_statistics method
            return {
                "modality": self.modality,
                "dataset_type": type(self.dataset).__name__
            }

    def _generate_summary(self, method_results: Dict, dimension_results: Dict, subset_results: Dict) -> Dict[str, Any]:
        """Generate analysis summary."""
        summary = {
            "modality": self.modality,
            "total_methods_tested": len(method_results) if method_results else 0,
            "embedding_dimension": dimension_results.get('embedding_dim', 0),
            "optimal_subset_size": subset_results.get('optimal_subset_size', 0),
            "dimension_reduction_achieved": 0
        }

        if subset_results.get('optimal_subset_size') and dimension_results.get('embedding_dim'):
            optimal_size = subset_results['optimal_subset_size']
            full_size = dimension_results['embedding_dim']
            summary['dimension_reduction_achieved'] = (1 - optimal_size/full_size) * 100

        if method_results:
            best_method = max(method_results.keys(), key=lambda k: method_results[k]["discrimination_ratio"])
            summary['best_method'] = best_method
            summary['best_discrimination_ratio'] = method_results[best_method]['discrimination_ratio']

        return summary

    def _save_results(self, results: Dict[str, Any]):
        """Save comprehensive results to JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"optimal_{self.modality}_embeddings_{timestamp}.json"

        # Also save to the standard optimal file name for compatibility
        standard_filename = f"optimal_{self.modality}_dimensions.json"

        # Custom JSON encoder to handle numpy types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        # Save timestamped version
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=convert_numpy_types)

        # Save standard version
        with open(standard_filename, 'w') as f:
            # For compatibility with existing benchmarks, save in expected format
            if results.get('subset_optimization', {}).get('optimal_dimensions'):
                optimal_data = {
                    "optimal_dimension_count": int(results['subset_optimization']['optimal_subset_size']),
                    "top_dimension_indices": [int(x) for x in results['subset_optimization']['optimal_dimensions']],
                    "analysis_timestamp": results['analysis_info']['timestamp'],
                    "discrimination_ratio": float(results['subset_optimization'][results['subset_optimization']['optimal_subset_size']]['discrimination_ratio']),
                    "separation": float(results['subset_optimization'][results['subset_optimization']['optimal_subset_size']]['separation']),
                    "embedding_dimension": int(results['dimension_analysis']['embedding_dim']),
                    "modality": self.modality
                }
                json.dump(optimal_data, f, indent=2, default=convert_numpy_types)
            else:
                json.dump(results, f, indent=2, default=convert_numpy_types)

        print(f"\n💾 Results saved to:")
        print(f"   • Detailed: {filename}")
        print(f"   • Standard: {standard_filename}")

    def _print_final_summary(self, results: Dict[str, Any], total_time: float):
        """Print final analysis summary."""
        print(f"\n🎉 {self.modality.upper()} ANALYSIS COMPLETE!")
        print("=" * 70)

        summary = results.get('summary', {})

        print(f"📊 Final Results:")
        print(f"   • Modality: {self.modality.capitalize()}")
        print(f"   • Methods tested: {summary.get('total_methods_tested', 0)}")
        print(f"   • Full embedding dimension: {summary.get('embedding_dimension', 0)}")
        print(f"   • Optimal subset size: {summary.get('optimal_subset_size', 0)}")
        print(f"   • Dimension reduction: {summary.get('dimension_reduction_achieved', 0):.1f}%")

        if summary.get('best_method'):
            print(f"   • Best method: {summary['best_method']}")
            print(f"   • Best discrimination ratio: {summary.get('best_discrimination_ratio', 0):.6f}")

        print(f"   • Total analysis time: {total_time:.1f} seconds")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Multi-Modal Embedding Analysis")
    parser.add_argument("--modality", choices=["iris", "face"], default="iris",
                       help="Choose modality: iris or face (default: iris)")
    parser.add_argument("--method-pairs", type=int, default=40,
                       help="Number of pairs for method testing (default: 40)")
    parser.add_argument("--dimension-pairs", type=int, default=60,
                       help="Number of pairs for dimension analysis (default: 60)")
    parser.add_argument("--subset-sizes", nargs="+", type=int,
                       help="Custom subset sizes to test (default: auto-generated)")

    args = parser.parse_args()

    # Create analyzer
    analyzer = MultiModalEmbeddingAnalyzer(modality=args.modality)

    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(
        num_method_test_pairs=args.method_pairs,
        num_dimension_pairs=args.dimension_pairs,
        subset_sizes=args.subset_sizes
    )

    return results


if __name__ == "__main__":
    main()

# python transformers_embeddings.py --modality iris --method-pairs 40 --dimension-pairs 60 --subset-sizes 70 90 120 150 200
# python transformers_embeddings.py --modality face --method-pairs 40 --dimension-pairs 60 --subset-sizes 500 768 1024