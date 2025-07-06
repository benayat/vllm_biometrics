#!/usr/bin/env python3
"""
Optimized Face Recognition Benchmark

This script uses optimized face embedding dimensions to benchmark
the LFW dataset for face recognition performance using Gemma Vision Model.
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

# ===============================================================================
# BENCHMARK CONFIGURATION
# ===============================================================================

# Model setup
MODEL_ID = "google/gemma-3-4b-it"

# Default optimal dimensions (will be loaded from analysis file if available)
DEFAULT_OPTIMAL_DIMENSIONS = 150  # Expected optimal range for faces

# Benchmark parameters
BATCH_SIZE = 50               # Reduced batch size for face processing
SIMILARITY_THRESHOLD = 0.85   # Threshold for same/different person (lower than iris)
MAX_PAIRS_TO_TEST = None      # Set to None for full dataset
SAVE_DETAILED_RESULTS = True  # Save individual pair results
OUTPUT_DIR = "face_benchmark_results"

# Performance tracking
performance_stats = {
    "total_pairs_processed": 0,
    "correct_predictions": 0,
    "genuine_pairs_tested": 0,
    "impostor_pairs_tested": 0,
    "true_positives": 0,
    "true_negatives": 0,
    "false_positives": 0,
    "false_negatives": 0,
    "processing_times": [],
    "similarity_scores": []
}

# ===============================================================================
# MODEL INITIALIZATION
# ===============================================================================

print("🔧 Loading Gemma Vision Model for face recognition...")
start_time = time.time()

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
).eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor2 = Gemma3ImageProcessor.from_pretrained(MODEL_ID)

load_time = time.time() - start_time
print(f"✅ Model loaded successfully in {load_time:.2f} seconds!")

# ===============================================================================
# OPTIMIZED FACE EMBEDDING FUNCTIONS
# ===============================================================================

def load_optimal_dimensions():
    """Load optimal dimensions from analysis file or use defaults."""
    global DEFAULT_OPTIMAL_DIMENSIONS
    
    optimal_file = "optimal_face_dimensions.json"
    
    if Path(optimal_file).exists():
        try:
            with open(optimal_file, 'r') as f:
                data = json.load(f)
            
            optimal_count = data.get("optimal_dimension_count", DEFAULT_OPTIMAL_DIMENSIONS)
            top_indices = data.get("top_dimension_indices", None)
            
            logger.info(f"Loaded optimal dimensions from {optimal_file}: {optimal_count} dimensions")
            return optimal_count, np.array(top_indices) if top_indices else None
            
        except Exception as e:
            logger.warning(f"Failed to load optimal dimensions: {e}")
    
    logger.info(f"Using default optimal dimensions: {DEFAULT_OPTIMAL_DIMENSIONS}")
    return DEFAULT_OPTIMAL_DIMENSIONS, None

def get_optimized_face_embedding(image_path, top_dimensions=None, method="mean_pooling"):
    """Extract optimized face embedding using only most discriminative dimensions."""
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        inputs = processor2(images=image, return_tensors="pt")
        
        # Move inputs to device and ensure consistent data types
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Use the same pipeline as the analysis phase to ensure dimension consistency
            image_outputs = model.vision_tower(inputs['pixel_values'])
            selected_image_feature = image_outputs.last_hidden_state

            # Convert to float32 if needed to avoid BFloat16 issues
            if selected_image_feature.dtype == torch.bfloat16:
                selected_image_feature = selected_image_feature.float()

            # Apply multi-modal projector to match analysis phase (2560 dimensions)
            image_embeddings = model.multi_modal_projector(selected_image_feature)

            # Convert to float32 if needed
            if image_embeddings.dtype == torch.bfloat16:
                image_embeddings = image_embeddings.float()

            # Apply pooling strategy (mean_pooling was used in analysis)
            if method == "mean_pooling":
                embedding = image_embeddings.mean(dim=1).squeeze(0)
            elif method == "max_pooling":
                embedding = image_embeddings.max(dim=1).values.squeeze(0)
            elif method == "cls_token":
                embedding = image_embeddings[:, 0, :].squeeze(0)
            else:
                embedding = image_embeddings.mean(dim=1).squeeze(0)

            # Convert to numpy
            full_embedding = embedding.cpu().numpy()
            
            # Extract optimal dimensions if specified
            if top_dimensions is not None:
                optimized_embedding = full_embedding[top_dimensions]
            else:
                optimized_embedding = full_embedding
            
            # Normalize the embedding
            norm = np.linalg.norm(optimized_embedding)
            if norm > 0:
                optimized_embedding = optimized_embedding / norm
            
            return optimized_embedding
            
    except Exception as e:
        logger.error(f"Error extracting embedding from {image_path}: {e}")
        return None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two face embeddings."""
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    # Cosine similarity (embeddings are already normalized)
    similarity = np.dot(embedding1, embedding2)
    return float(similarity)

def process_face_pairs_batch(pairs_batch, top_dimensions=None):
    """Process a batch of face pairs and return results."""
    batch_results = []
    batch_start_time = time.time()
    
    for i, (img1_path, img2_path, true_label) in enumerate(pairs_batch):
        try:
            # Extract embeddings
            emb1 = get_optimized_face_embedding(str(img1_path), top_dimensions)
            emb2 = get_optimized_face_embedding(str(img2_path), top_dimensions)
            
            if emb1 is None or emb2 is None:
                logger.warning(f"Failed to extract embeddings for pair {i}")
                continue
            
            # Calculate similarity
            similarity = calculate_similarity(emb1, emb2)
            
            # Make prediction
            predicted_label = 1 if similarity > SIMILARITY_THRESHOLD else 0
            is_correct = predicted_label == true_label
            
            # Store result
            result = {
                "image1": str(img1_path),
                "image2": str(img2_path),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "similarity": similarity,
                "is_correct": is_correct,
                "embedding_dim": len(emb1)
            }
            
            batch_results.append(result)
            
            # Update performance stats
            performance_stats["total_pairs_processed"] += 1
            performance_stats["similarity_scores"].append(similarity)
            
            if is_correct:
                performance_stats["correct_predictions"] += 1
            
            if true_label == 1:
                performance_stats["genuine_pairs_tested"] += 1
                if predicted_label == 1:
                    performance_stats["true_positives"] += 1
                else:
                    performance_stats["false_negatives"] += 1
            else:
                performance_stats["impostor_pairs_tested"] += 1
                if predicted_label == 0:
                    performance_stats["true_negatives"] += 1
                else:
                    performance_stats["false_positives"] += 1
            
        except Exception as e:
            logger.error(f"Error processing pair {i}: {e}")
            continue
    
    batch_time = time.time() - batch_start_time
    performance_stats["processing_times"].append(batch_time)
    
    return batch_results

def calculate_metrics():
    """Calculate comprehensive performance metrics."""
    tp = performance_stats["true_positives"]
    tn = performance_stats["true_negatives"]
    fp = performance_stats["false_positives"]
    fn = performance_stats["false_negatives"]
    
    total = tp + tn + fp + fn
    
    if total == 0:
        return {}
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate false acceptance rate (FAR) and false rejection rate (FRR)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Equal Error Rate (EER) approximation
    eer = (far + frr) / 2
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "false_acceptance_rate": far,
        "false_rejection_rate": frr,
        "equal_error_rate": eer,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total_pairs": total
    }

def run_face_recognition_benchmark(benchmark_size="medium"):
    """Run comprehensive face recognition benchmark."""
    
    print(f"\n🚀 STARTING FACE RECOGNITION BENCHMARK")
    print("=" * 60)
    
    # Load optimal dimensions
    optimal_dim_count, top_dimensions = load_optimal_dimensions()
    
    print(f"📊 Benchmark Configuration:")
    print(f"   • Model: {MODEL_ID}")
    print(f"   • Embedding dimensions: {optimal_dim_count}")
    print(f"   • Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Benchmark size: {benchmark_size}")
    
    # Load LFW dataset
    print(f"\n📂 Loading LFW dataset...")
    dataset = LoadLFWDataset()
    
    # Get benchmark configuration
    benchmark_configs = dataset.get_benchmark_subsets()
    
    if benchmark_size not in benchmark_configs:
        logger.error(f"Unknown benchmark size: {benchmark_size}")
        return None
    
    config = benchmark_configs[benchmark_size]
    
    # Get test pairs
    if benchmark_size == "official_only":
        test_pairs = dataset.get_official_pairs_only()
    else:
        test_pairs = dataset.get_balanced_sample(config["genuine"], config["impostor"])
    
    if MAX_PAIRS_TO_TEST:
        test_pairs = test_pairs[:MAX_PAIRS_TO_TEST]
    
    print(f"\n🧪 Test Configuration:")
    print(f"   • Total pairs: {len(test_pairs)}")
    print(f"   • Expected genuine pairs: {config['genuine']}")
    print(f"   • Expected impostor pairs: {config['impostor']}")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Start benchmarking
    benchmark_start_time = time.time()
    all_results = []
    
    print(f"\n⚡ Processing {len(test_pairs)} face pairs...")
    
    # Process in batches
    for i in tqdm(range(0, len(test_pairs), BATCH_SIZE), desc="Processing batches"):
        batch = test_pairs[i:i + BATCH_SIZE]
        batch_results = process_face_pairs_batch(batch, top_dimensions)
        all_results.extend(batch_results)
        
        # Print progress every 5 batches
        if (i // BATCH_SIZE + 1) % 5 == 0:
            current_accuracy = performance_stats["correct_predictions"] / max(performance_stats["total_pairs_processed"], 1)
            avg_time = np.mean(performance_stats["processing_times"]) if performance_stats["processing_times"] else 0
            
            print(f"   Batch {i // BATCH_SIZE + 1}: {current_accuracy:.3f} accuracy, {avg_time:.2f}s/batch")
    
    benchmark_time = time.time() - benchmark_start_time
    
    # Calculate final metrics
    metrics = calculate_metrics()
    
    # Create comprehensive results
    final_results = {
        "benchmark_info": {
            "model_id": MODEL_ID,
            "benchmark_size": benchmark_size,
            "dataset": "LFW",
            "total_pairs_tested": len(all_results),
            "embedding_dimensions": optimal_dim_count,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "batch_size": BATCH_SIZE,
            "benchmark_time_seconds": benchmark_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "performance_metrics": metrics,
        "detailed_stats": {
            "avg_processing_time_per_batch": np.mean(performance_stats["processing_times"]) if performance_stats["processing_times"] else 0,
            "total_processing_time": benchmark_time,
            "pairs_per_second": len(all_results) / benchmark_time if benchmark_time > 0 else 0,
            "avg_similarity_score": np.mean(performance_stats["similarity_scores"]) if performance_stats["similarity_scores"] else 0,
            "similarity_std": np.std(performance_stats["similarity_scores"]) if performance_stats["similarity_scores"] else 0
        },
        "configuration": {
            "optimal_dimensions_file": "optimal_face_dimensions.json" if top_dimensions is not None else "default",
            "using_optimized_dimensions": top_dimensions is not None,
            "embedding_method": "mean_pooling"
        }
    }
    
    # Add individual results if requested
    if SAVE_DETAILED_RESULTS:
        final_results["individual_results"] = all_results
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"face_benchmark_{benchmark_size}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n🎯 FACE RECOGNITION BENCHMARK RESULTS")
    print("=" * 60)
    print(f"📊 Performance Metrics:")
    print(f"   • Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   • Precision: {metrics['precision']:.4f}")
    print(f"   • Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"   • Specificity: {metrics['specificity']:.4f}")
    print(f"   • F1-Score: {metrics['f1_score']:.4f}")
    print(f"   • False Acceptance Rate: {metrics['false_acceptance_rate']:.4f}")
    print(f"   • False Rejection Rate: {metrics['false_rejection_rate']:.4f}")
    print(f"   • Equal Error Rate: {metrics['equal_error_rate']:.4f}")
    
    print(f"\n⚡ Performance Stats:")
    print(f"   • Total pairs processed: {len(all_results)}")
    print(f"   • Processing time: {benchmark_time:.2f} seconds")
    print(f"   • Pairs per second: {len(all_results)/benchmark_time:.2f}")
    print(f"   • Embedding dimensions: {optimal_dim_count}")
    
    print(f"\n📁 Results saved to: {results_file}")
    
    return final_results

def compare_benchmark_sizes():
    """Compare performance across different benchmark sizes."""
    
    print(f"\n🔬 COMPREHENSIVE FACE BENCHMARK COMPARISON")
    print("=" * 60)
    
    sizes_to_test = ["small", "medium", "large", "official_only"]
    all_results = {}
    
    for size in sizes_to_test:
        print(f"\n{'='*20} Testing {size.upper()} {'='*20}")
        
        # Reset performance stats for each test
        global performance_stats
        performance_stats = {
            "total_pairs_processed": 0,
            "correct_predictions": 0,
            "genuine_pairs_tested": 0,
            "impostor_pairs_tested": 0,
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "processing_times": [],
            "similarity_scores": []
        }
        
        try:
            results = run_face_recognition_benchmark(size)
            all_results[size] = results
            
        except Exception as e:
            print(f"❌ Error testing {size}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create comparison summary
    print(f"\n📊 FACE BENCHMARK COMPARISON SUMMARY")
    print("=" * 60)
    
    comparison_data = {
        "comparison_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": MODEL_ID,
        "results": {}
    }
    
    for size, results in all_results.items():
        if results and "performance_metrics" in results:
            metrics = results["performance_metrics"]
            comparison_data["results"][size] = {
                "accuracy": metrics["accuracy"],
                "pairs_tested": metrics["total_pairs"],
                "processing_time": results["benchmark_info"]["benchmark_time_seconds"],
                "f1_score": metrics["f1_score"],
                "equal_error_rate": metrics["equal_error_rate"]
            }
            
            print(f"{size:12s}: {metrics['accuracy']:.4f} accuracy, {metrics['total_pairs']:4d} pairs, {results['benchmark_info']['benchmark_time_seconds']:6.1f}s")
    
    # Save comparison
    comparison_file = Path(OUTPUT_DIR) / f"face_benchmark_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"\n📁 Comparison saved to: {comparison_file}")
    
    return all_results

def optimize_threshold():
    """Find optimal similarity threshold for face recognition."""
    
    print(f"\n🎛️ OPTIMIZING SIMILARITY THRESHOLD")
    print("=" * 60)
    
    # Load dataset and get sample for threshold optimization
    dataset = LoadLFWDataset()
    optimization_pairs = dataset.get_balanced_sample(200, 200)  # Smaller sample for optimization
    
    print(f"Testing {len(optimization_pairs)} pairs for threshold optimization...")
    
    # Load optimal dimensions
    optimal_dim_count, top_dimensions = load_optimal_dimensions()
    
    # Extract similarities and labels
    similarities = []
    labels = []
    
    for img1_path, img2_path, true_label in tqdm(optimization_pairs, desc="Extracting similarities"):
        emb1 = get_optimized_face_embedding(str(img1_path), top_dimensions)
        emb2 = get_optimized_face_embedding(str(img2_path), top_dimensions)
        
        if emb1 is not None and emb2 is not None:
            similarity = calculate_similarity(emb1, emb2)
            similarities.append(similarity)
            labels.append(true_label)
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # Test different thresholds
    thresholds = np.linspace(similarities.min(), similarities.max(), 100)
    best_accuracy = 0
    best_threshold = SIMILARITY_THRESHOLD
    
    threshold_results = []
    
    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        
        # Calculate other metrics
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        threshold_results.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"\n🎯 Threshold Optimization Results:")
    print(f"   • Best threshold: {best_threshold:.4f}")
    print(f"   • Best accuracy: {best_accuracy:.4f}")
    print(f"   • Current threshold: {SIMILARITY_THRESHOLD:.4f}")
    
    # Save threshold analysis
    threshold_file = Path(OUTPUT_DIR) / f"threshold_optimization_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(threshold_file, 'w') as f:
        json.dump({
            "optimal_threshold": best_threshold,
            "optimal_accuracy": best_accuracy,
            "current_threshold": SIMILARITY_THRESHOLD,
            "all_thresholds": threshold_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2, default=str)
    
    return best_threshold, threshold_results

if __name__ == "__main__":
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Run different benchmarks based on command line args or run all
    import sys
    
    if len(sys.argv) > 1:
        benchmark_type = sys.argv[1]
        
        if benchmark_type == "optimize_threshold":
            optimize_threshold()
        elif benchmark_type == "compare_all":
            compare_benchmark_sizes()
        elif benchmark_type in ["small", "medium", "large", "official_only", "full_dataset"]:
            run_face_recognition_benchmark(benchmark_type)
        else:
            print(f"Unknown benchmark type: {benchmark_type}")
            print("Available options: small, medium, large, official_only, full_dataset, compare_all, optimize_threshold")
    else:
        # Default: run medium benchmark
        print("Running default medium face recognition benchmark...")
        print("Use: python benchmark_optimized_face.py [small|medium|large|official_only|compare_all|optimize_threshold]")
        
        run_face_recognition_benchmark("medium")
