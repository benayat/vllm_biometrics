#!/usr/bin/env python3
"""
Face Embedding Optimization using Gemma Vision Model

This script analyzes face embeddings from the LFW dataset to find the most discriminative
dimensions for face recognition, similar to the iris optimization approach.
"""

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Gemma3ImageProcessor
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import time
import json
from typing import List, Tuple, Dict, Any
import logging
from tqdm import tqdm

# Add the project root to the path for imports
sys.path.append(str(Path(__file__).parent))
from dataloader.load_lfw_dataset import LoadLFWDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch._dynamo.config.disable = True

# Model setup
MODEL_ID = "google/gemma-3-4b-it"

print("🔧 Loading model...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    device_map="auto",
    torch_dtype=torch.float32,  # Use float32 for compatibility
    low_cpu_mem_usage=True
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor2 = Gemma3ImageProcessor.from_pretrained(MODEL_ID)
print("✅ Model loaded successfully!")

def get_face_embedding(image_path, method="mean_pooling"):
    """Extract embedding from face image using vision tower with different pooling strategies."""
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        inputs = processor2(images=image, return_tensors="pt")

        # Move inputs to device with float32 for compatibility
        inputs = {k: v.to(model.device).float() for k, v in inputs.items()}

        with torch.no_grad():
            # Use the pipeline with float32 tensors
            image_outputs = model.vision_tower(inputs['pixel_values'])
            selected_image_feature = image_outputs.last_hidden_state
            image_embeddings = model.multi_modal_projector(selected_image_feature)

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

def extract_face_embeddings(image_paths: List[str], batch_size: int = 8, method: str = "mean_pooling") -> List[np.ndarray]:
    """Extract embeddings from multiple face images with batching."""
    embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting face embeddings"):
        batch_paths = image_paths[i:i + batch_size]
        batch_embeddings = []

        for path in batch_paths:
            embedding = get_face_embedding(path, method)
            if embedding is not None:
                batch_embeddings.append(embedding)
            else:
                logger.warning(f"Failed to extract embedding for {path}")

        embeddings.extend(batch_embeddings)

    return embeddings

def compute_similarity(embedding1, embedding2, metric="cosine"):
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

def test_multiple_face_methods():
    """Test multiple embedding and similarity methods for faces."""
    print("\n🔬 COMPREHENSIVE FACE EMBEDDING TEST")
    print("=" * 60)

    # Load LFW dataset
    print("📂 Loading LFW dataset...")
    dataset = LoadLFWDataset()

    # Get multiple test pairs for better evaluation
    genuine_pairs = dataset.genuine_pairs[:10]  # Test 10 genuine pairs
    impostor_pairs = dataset.impostor_pairs[:10]  # Test 10 impostor pairs

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
            print(f"\n  Similarity Metric: {sim_metric}")

            genuine_similarities = []
            impostor_similarities = []

            # Test genuine pairs
            for img1, img2, _ in genuine_pairs:
                emb1 = get_face_embedding(str(img1), method=emb_method)
                emb2 = get_face_embedding(str(img2), method=emb_method)
                if emb1 is not None and emb2 is not None:
                    sim = compute_similarity(emb1, emb2, metric=sim_metric)
                    genuine_similarities.append(sim)

            # Test impostor pairs
            for img1, img2, _ in impostor_pairs:
                emb1 = get_face_embedding(str(img1), method=emb_method)
                emb2 = get_face_embedding(str(img2), method=emb_method)
                if emb1 is not None and emb2 is not None:
                    sim = compute_similarity(emb1, emb2, metric=sim_metric)
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
                    "std_impostor": std_impostor
                }

                print(f"    Genuine avg: {avg_genuine:.4f} ± {std_genuine:.4f}")
                print(f"    Impostor avg: {avg_impostor:.4f} ± {std_impostor:.4f}")
                print(f"    Separation: {separation:.4f}")
                print(f"    Discrimination ratio: {discrimination_ratio:.4f}")

    # Find best method
    best_method = max(results.keys(), key=lambda k: results[k]["discrimination_ratio"])
    best_result = results[best_method]

    print(f"\n🏆 BEST METHOD: {best_method}")
    print(f"   Discrimination ratio: {best_result['discrimination_ratio']:.4f}")
    print(f"   Separation: {best_result['separation']:.4f}")
    print(f"   Genuine: {best_result['avg_genuine']:.4f} ± {best_result['std_genuine']:.4f}")
    print(f"   Impostor: {best_result['avg_impostor']:.4f} ± {best_result['std_impostor']:.4f}")

    return results

def analyze_face_embedding_dimensions():
    """Analyze which dimensions of the face embedding vector are most discriminative."""
    print("\n🔬 FACE EMBEDDING DIMENSION ANALYSIS")
    print("=" * 60)

    # Load LFW dataset
    print("📂 Loading LFW dataset...")
    dataset = LoadLFWDataset()

    # Get more pairs for better statistics
    genuine_pairs = dataset.genuine_pairs[:20]  # 20 genuine pairs
    impostor_pairs = dataset.impostor_pairs[:20]  # 20 impostor pairs

    print("🧮 Extracting face embeddings...")

    # Extract all embeddings first
    genuine_embeddings = []
    impostor_embeddings = []

    # Get genuine pair embeddings
    for i, (img1, img2, _) in enumerate(genuine_pairs):
        print(f"Processing genuine pair {i+1}/20...")
        emb1 = get_face_embedding(str(img1), method="mean_pooling")
        emb2 = get_face_embedding(str(img2), method="mean_pooling")
        if emb1 is not None and emb2 is not None:
            genuine_embeddings.append((emb1, emb2))

    # Get impostor pair embeddings
    for i, (img1, img2, _) in enumerate(impostor_pairs):
        print(f"Processing impostor pair {i+1}/20...")
        emb1 = get_face_embedding(str(img1), method="mean_pooling")
        emb2 = get_face_embedding(str(img2), method="mean_pooling")
        if emb1 is not None and emb2 is not None:
            impostor_embeddings.append((emb1, emb2))

    if not genuine_embeddings or not impostor_embeddings:
        print("❌ Failed to extract sufficient embeddings")
        return

    # Convert to numpy arrays for analysis
    embedding_dim = len(genuine_embeddings[0][0])
    print(f"📊 Face embedding dimension: {embedding_dim}")

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

    print(f"\n📈 Face Dimension Analysis Results:")
    print(f"   Top 10 most discriminative dimensions:")
    for i in range(min(10, len(sorted_indices))):
        dim = sorted_indices[i]
        score = discrimination_scores[dim]
        gen_diff = genuine_mean_diff[dim]
        imp_diff = impostor_mean_diff[dim]
        print(f"   Dim {dim:4d}: score={score:.6f} (genuine={gen_diff:.6f}, impostor={imp_diff:.6f})")

    return {
        'discrimination_scores': discrimination_scores,
        'sorted_indices': sorted_indices,
        'genuine_embeddings': genuine_embeddings,
        'impostor_embeddings': impostor_embeddings,
        'embedding_dim': embedding_dim
    }

def test_face_subset_embeddings(analysis_results, subset_sizes=None):
    """Test face performance using only subsets of most discriminative dimensions."""
    print(f"\n🎯 TESTING FACE EMBEDDING SUBSETS")
    print("=" * 60)

    discrimination_scores = analysis_results['discrimination_scores']
    sorted_indices = analysis_results['sorted_indices']
    genuine_embeddings = analysis_results['genuine_embeddings']
    impostor_embeddings = analysis_results['impostor_embeddings']
    full_dim = analysis_results['embedding_dim']

    # If no subset sizes provided, test a comprehensive range
    if subset_sizes is None:
        subset_sizes = [
            # Small subsets
            5, 10, 15, 20, 25, 30, 40, 50,
            # Medium subsets
            60, 70, 80, 90, 100, 120, 150,
            # Larger subsets
            200, 250, 300, 400, 500, 600, 700, 800,
            # Very large subsets
            1000, 1200, 1500, 1800, 2000, 2200
        ]
        # Remove sizes that exceed the embedding dimension
        subset_sizes = [size for size in subset_sizes if size < full_dim]

    print(f"Testing {len(subset_sizes)} different subset sizes for faces: {subset_sizes}")

    results = {}

    # Test different subset sizes
    for i, subset_size in enumerate(subset_sizes):
        print(f"\n--- Testing top {subset_size} dimensions ({i+1}/{len(subset_sizes)}) ---")

        # Get indices of most discriminative dimensions
        top_indices = sorted_indices[:subset_size]

        # Calculate similarities using only these dimensions
        genuine_similarities = []
        impostor_similarities = []

        # Test genuine pairs
        for emb1, emb2 in genuine_embeddings:
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
        for emb1, emb2 in impostor_embeddings:
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
            'std_impostor': std_impostor
        }

        print(f"   Genuine avg: {avg_genuine:.6f} ± {std_genuine:.6f}")
        print(f"   Impostor avg: {avg_impostor:.6f} ± {std_impostor:.6f}")
        print(f"   Separation: {separation:.6f}")
        print(f"   Discrimination ratio: {discrimination_ratio:.6f}")

    # Find best subset size
    best_subset = max(results.keys(), key=lambda k: results[k]["discrimination_ratio"])
    best_result = results[best_subset]

    # Find top 5 best performing subset sizes
    sorted_by_discrimination = sorted(results.items(), key=lambda x: x[1]["discrimination_ratio"], reverse=True)

    print(f"\n🏆 TOP 5 BEST FACE SUBSET SIZES:")
    for i, (size, result) in enumerate(sorted_by_discrimination[:5]):
        print(f"   {i+1}. {size:4d} dimensions: discrimination={result['discrimination_ratio']:.6f}, separation={result['separation']:.6f}")

    print(f"\n🥇 BEST FACE SUBSET SIZE: {best_subset} dimensions")
    print(f"   Discrimination ratio: {best_result['discrimination_ratio']:.6f}")
    print(f"   Separation: {best_result['separation']:.6f}")
    print(f"   Genuine: {best_result['avg_genuine']:.6f} ± {best_result['std_genuine']:.6f}")
    print(f"   Impostor: {best_result['avg_impostor']:.6f} ± {best_result['std_impostor']:.6f}")

    # Compare with full embedding
    print(f"\n📊 Improvement over full face embedding:")

    # Calculate full embedding performance for comparison
    full_genuine_sims = []
    full_impostor_sims = []

    for emb1, emb2 in genuine_embeddings:
        similarity = np.dot(emb1, emb2)
        full_genuine_sims.append(similarity)

    for emb1, emb2 in impostor_embeddings:
        similarity = np.dot(emb1, emb2)
        full_impostor_sims.append(similarity)

    full_separation = np.mean(full_genuine_sims) - np.mean(full_impostor_sims)
    full_discrimination = full_separation / (np.std(full_genuine_sims) + np.std(full_impostor_sims) + 1e-8)

    improvement_ratio = best_result['discrimination_ratio'] / full_discrimination
    improvement_separation = best_result['separation'] / full_separation

    print(f"   Full embedding discrimination ratio: {full_discrimination:.6f}")
    print(f"   Best subset improvement: {improvement_ratio:.2f}x better")
    print(f"   Separation improvement: {improvement_separation:.2f}x better")
    print(f"   Dimensionality reduction: {(1 - best_subset/full_dim)*100:.1f}% size reduction")

    return results, best_subset, sorted_indices[:best_subset]

def get_optimized_face_embedding(image_path, top_dimensions=None):
    """Extract optimized face embedding using only most discriminative dimensions."""
    # Get full embedding first
    full_embedding = get_face_embedding(image_path, method="mean_pooling")

    if full_embedding is None or top_dimensions is None:
        return full_embedding

    # Extract only the most discriminative dimensions
    optimized_embedding = full_embedding[top_dimensions]

    # Renormalize
    norm = np.linalg.norm(optimized_embedding)
    if norm > 0:
        optimized_embedding = optimized_embedding / norm

    return optimized_embedding

def test_optimized_face_comparison(top_dimensions):
    """Test face comparison using optimized embedding subset."""
    print(f"\n🎯 TESTING OPTIMIZED FACE COMPARISON")
    print("=" * 60)

    # Load dataset
    dataset = LoadLFWDataset()

    # Get test pairs
    genuine_pair = dataset.genuine_pairs[0]
    impostor_pair = dataset.impostor_pairs[0]

    print(f"Using top {len(top_dimensions)} dimensions out of {top_dimensions.max() + 1}")

    # Test genuine pair
    img1_gen, img2_gen, _ = genuine_pair
    emb1_gen = get_optimized_face_embedding(str(img1_gen), top_dimensions)
    emb2_gen = get_optimized_face_embedding(str(img2_gen), top_dimensions)

    if emb1_gen is not None and emb2_gen is not None:
        similarity_gen = np.dot(emb1_gen, emb2_gen)
        print(f"✅ Optimized genuine similarity: {similarity_gen:.6f}")

    # Test impostor pair
    img1_imp, img2_imp, _ = impostor_pair
    emb1_imp = get_optimized_face_embedding(str(img1_imp), top_dimensions)
    emb2_imp = get_optimized_face_embedding(str(img2_imp), top_dimensions)

    if emb1_imp is not None and emb2_imp is not None:
        similarity_imp = np.dot(emb1_imp, emb2_imp)
        print(f"✅ Optimized impostor similarity: {similarity_imp:.6f}")

    # Analysis
    if emb1_gen is not None and emb2_gen is not None and emb1_imp is not None and emb2_imp is not None:
        difference = similarity_gen - similarity_imp
        improvement = (difference / similarity_imp) * 100 if similarity_imp != 0 else 0

        print(f"\n📊 Optimized Face Analysis:")
        print(f"   Difference: {difference:.6f}")
        print(f"   Improvement: {improvement:.2f}%")
        print(f"   Embedding size reduced by: {(1 - len(top_dimensions)/2560)*100:.1f}%")

def comprehensive_face_dimension_analysis():
    """Run comprehensive dimension analysis for faces."""
    # Step 1: Analyze embedding dimensions
    analysis_results = analyze_face_embedding_dimensions()

    if analysis_results is None:
        return

    # Step 2: Test different subset sizes
    subset_results, best_subset, top_dimensions = test_face_subset_embeddings(analysis_results)

    # Step 3: Test optimized comparison
    test_optimized_face_comparison(top_dimensions)

    # Step 4: Save optimal dimensions to file for use by other scripts
    optimal_dims_data = {
        "optimal_dimension_count": int(best_subset),
        "top_dimension_indices": top_dimensions.tolist(),
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "discrimination_ratio": float(subset_results[best_subset]["discrimination_ratio"]),
        "separation": float(subset_results[best_subset]["separation"]),
        "embedding_dimension": int(analysis_results['embedding_dim']),
        "modality": "face"
    }

    output_file = "optimal_face_dimensions.json"
    with open(output_file, 'w') as f:
        json.dump(optimal_dims_data, f, indent=2)

    print(f"\n💾 Optimal face dimensions saved to: {output_file}")
    print(f"   Best subset size: {best_subset} dimensions")
    print(f"   Discrimination ratio: {subset_results[best_subset]['discrimination_ratio']:.6f}")
    print(f"   Use this file with benchmark_optimized_face.py for automatic loading")

    return analysis_results, subset_results, top_dimensions

def test_face_comparison():
    """Simple face comparison test - calls the comprehensive test."""
    return test_multiple_face_methods()

def calculate_discrimination_metrics(embeddings: List[np.ndarray], labels: List[int]) -> Dict[str, float]:
    """Calculate discrimination metrics for face embeddings."""
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Calculate within-class and between-class distances
    within_class_distances = []
    between_class_distances = []

    unique_labels = np.unique(labels)

    for label in unique_labels:
        # Get embeddings for this class
        class_embeddings = embeddings[labels == label]

        # Calculate within-class distances
        if len(class_embeddings) > 1:
            for i in range(len(class_embeddings)):
                for j in range(i + 1, len(class_embeddings)):
                    dist = np.linalg.norm(class_embeddings[i] - class_embeddings[j])
                    within_class_distances.append(dist)

        # Calculate between-class distances (sample to avoid too many comparisons)
        other_embeddings = embeddings[labels != label]
        if len(other_embeddings) > 0:
            # Sample a subset to avoid computation explosion
            sample_size = min(100, len(other_embeddings))
            sampled_other = other_embeddings[np.random.choice(len(other_embeddings), sample_size, replace=False)]

            for class_emb in class_embeddings:
                for other_emb in sampled_other:
                    dist = np.linalg.norm(class_emb - other_emb)
                    between_class_distances.append(dist)

    # Calculate metrics
    within_mean = np.mean(within_class_distances) if within_class_distances else 0
    between_mean = np.mean(between_class_distances) if between_class_distances else 0

    separation = between_mean - within_mean
    discrimination_ratio = between_mean / within_mean if within_mean > 0 else 0

    return {
        "within_class_distance": within_mean,
        "between_class_distance": between_mean,
        "separation": separation,
        "discrimination_ratio": discrimination_ratio
    }

def analyze_embedding_dimensions(embeddings: List[np.ndarray], labels: List[int],
                                top_k: int = 150) -> Dict[str, Any]:
    """Analyze which embedding dimensions are most discriminative for face recognition."""
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    num_dimensions = embeddings.shape[1]
    dimension_scores = []

    logger.info(f"Analyzing {num_dimensions} embedding dimensions...")

    for dim in tqdm(range(num_dimensions), desc="Analyzing dimensions"):
        # Extract this dimension for all embeddings
        dim_values = embeddings[:, dim]

        # Calculate discrimination metrics for this dimension
        within_class_vars = []
        between_class_vars = []

        unique_labels = np.unique(labels)

        for label in unique_labels:
            class_values = dim_values[labels == label]
            if len(class_values) > 1:
                within_class_vars.append(np.var(class_values))

        # Calculate between-class variance
        class_means = []
        for label in unique_labels:
            class_values = dim_values[labels == label]
            if len(class_values) > 0:
                class_means.append(np.mean(class_values))

        within_class_var = np.mean(within_class_vars) if within_class_vars else 0
        between_class_var = np.var(class_means) if len(class_means) > 1 else 0

        # Fisher's discriminant ratio
        score = between_class_var / within_class_var if within_class_var > 0 else 0

        dimension_scores.append({
            "dimension": dim,
            "score": score,
            "within_class_variance": within_class_var,
            "between_class_variance": between_class_var
        })

    # Sort by score and get top-k
    dimension_scores.sort(key=lambda x: x["score"], reverse=True)
    top_dimensions = dimension_scores[:top_k]

    return {
        "top_dimensions": top_dimensions,
        "all_scores": dimension_scores,
        "summary": {
            "total_dimensions": num_dimensions,
            "top_k": top_k,
            "best_score": top_dimensions[0]["score"] if top_dimensions else 0,
            "worst_score": dimension_scores[-1]["score"] if dimension_scores else 0
        }
    }

def test_face_recognition_performance(sample_size: int = 1000, method: str = "mean_pooling"):
    """Test face recognition performance with different configurations."""
    logger.info("Loading LFW dataset...")
    dataset = LoadLFWDataset()

    # Get a balanced sample for testing
    test_pairs = dataset.get_balanced_sample(sample_size // 2, sample_size // 2)

    logger.info(f"Testing with {len(test_pairs)} face pairs...")

    # Extract all unique image paths
    image_paths = []
    for pair in test_pairs:
        img1_path, img2_path, label = pair
        image_paths.extend([str(img1_path), str(img2_path)])

    # Remove duplicates while preserving order
    unique_paths = list(dict.fromkeys(image_paths))

    # Extract embeddings
    logger.info("Extracting face embeddings...")
    embeddings = extract_face_embeddings(unique_paths, method=method)

    # Create mapping from path to embedding
    path_to_embedding = {path: emb for path, emb in zip(unique_paths, embeddings)}

    # Calculate similarities for pairs
    similarities = []
    labels = []

    for pair in test_pairs:
        img1_path, img2_path, label = pair

        if str(img1_path) in path_to_embedding and str(img2_path) in path_to_embedding:
            emb1 = path_to_embedding[str(img1_path)]
            emb2 = path_to_embedding[str(img2_path)]

            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(similarity)
            labels.append(label)

    # Calculate metrics
    similarities = np.array(similarities)
    labels = np.array(labels)

    # Find optimal threshold
    thresholds = np.linspace(similarities.min(), similarities.max(), 100)
    best_accuracy = 0
    best_threshold = 0

    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        accuracy = np.mean(predictions == labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    # Calculate final metrics
    predictions = (similarities > best_threshold).astype(int)
    accuracy = np.mean(predictions == labels)

    genuine_similarities = similarities[labels == 1]
    impostor_similarities = similarities[labels == 0]

    results = {
        "method": method,
        "sample_size": len(test_pairs),
        "accuracy": accuracy,
        "best_threshold": best_threshold,
        "genuine_similarity_mean": np.mean(genuine_similarities),
        "genuine_similarity_std": np.std(genuine_similarities),
        "impostor_similarity_mean": np.mean(impostor_similarities),
        "impostor_similarity_std": np.std(impostor_similarities),
        "separation": np.mean(genuine_similarities) - np.mean(impostor_similarities)
    }

    return results

def optimize_face_embeddings(sample_size: int = 500, top_k_dimensions: int = 150):
    """Optimize face embeddings by finding the most discriminative dimensions."""
    logger.info("Starting face embedding optimization...")

    # Load dataset
    dataset = LoadLFWDataset()

    # Get people with multiple images for analysis
    multi_image_people = dataset.get_people_with_multiple_images(min_images=2)

    # Sample people for analysis
    sampled_people = list(multi_image_people.keys())[:min(50, len(multi_image_people))]

    # Collect image paths and labels
    image_paths = []
    labels = []

    for person_id, person_name in enumerate(sampled_people):
        person_images = dataset.subject_images[person_name]
        # Take up to 10 images per person
        for image_path in person_images[:min(10, len(person_images))]:
            image_paths.append(str(image_path))
            labels.append(person_id)

    logger.info(f"Analyzing {len(image_paths)} face images from {len(sampled_people)} people...")

    # Extract embeddings
    embeddings = extract_face_embeddings(image_paths, method="mean_pooling")

    # Analyze dimensions
    analysis_results = analyze_embedding_dimensions(embeddings, labels, top_k_dimensions)

    # Calculate overall discrimination metrics
    discrimination_metrics = calculate_discrimination_metrics(embeddings, labels)

    # Save results
    results = {
        "optimization_results": analysis_results,
        "discrimination_metrics": discrimination_metrics,
        "dataset_info": {
            "total_images": len(image_paths),
            "total_people": len(sampled_people),
            "embedding_dimensions": len(embeddings[0]) if embeddings else 0
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save to file
    results_file = "optimal_face_dimensions.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {results_file}")

    return results

if __name__ == "__main__":
    # Test different pooling methods
    methods = ["mean_pooling", "max_pooling", "cls_token"]

    print("🔬 Testing face recognition performance with different pooling methods...")

    for method in methods:
        print(f"\n📊 Testing method: {method}")
        try:
            results = test_face_recognition_performance(sample_size=200, method=method)

            print(f"✅ Results for {method}:")
            print(f"   • Accuracy: {results['accuracy']:.3f}")
            print(f"   • Best threshold: {results['best_threshold']:.3f}")
            print(f"   • Genuine similarity: {results['genuine_similarity_mean']:.3f} ± {results['genuine_similarity_std']:.3f}")
            print(f"   • Impostor similarity: {results['impostor_similarity_mean']:.3f} ± {results['impostor_similarity_std']:.3f}")
            print(f"   • Separation: {results['separation']:.3f}")

        except Exception as e:
            print(f"❌ Error testing {method}: {e}")
            import traceback
            traceback.print_exc()

    # Optimize embeddings
    print("\n🔧 Optimizing face embeddings...")
    try:
        optimization_results = optimize_face_embeddings(sample_size=300, top_k_dimensions=150)

        print("✅ Optimization complete!")
        print(f"   • Top discriminative dimensions: {optimization_results['optimization_results']['summary']['top_k']}")
        print(f"   • Best dimension score: {optimization_results['optimization_results']['summary']['best_score']:.3f}")
        print(f"   • Discrimination ratio: {optimization_results['discrimination_metrics']['discrimination_ratio']:.3f}")

    except Exception as e:
        print(f"❌ Optimization error: {e}")
        import traceback
        traceback.print_exc()
