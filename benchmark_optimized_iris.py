#!/usr/bin/env python3
"""
Optimized Iris Recognition Benchmark

This script uses the optimized 90-dimension iris embedding approach to benchmark
the entire IITD dataset for iris recognition performance.
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
from dataloader.load_iitd_dataset import LoadIITDDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch._dynamo.config.disable = True

# ===============================================================================
# BENCHMARK CONFIGURATION
# ===============================================================================

# Model setup
MODEL_ID = "google/gemma-3-4b-it"

# Optimized dimensions (from previous analysis)
OPTIMAL_DIMENSIONS = 90

# Pre-computed top discriminative dimension indices (from dimension analysis)
# These are the specific 90 dimensions that showed best discrimination
TOP_DIMENSIONS = None  # Will be loaded from dimension analysis or computed

# Benchmark parameters
BATCH_SIZE = 100              # Number of pairs to process simultaneously (reduced for memory)
SIMILARITY_THRESHOLD = 0.96  # Threshold for same/different person decision
MAX_PAIRS_TO_TEST = None     # Set to None for full dataset, or number for subset
SAVE_DETAILED_RESULTS = True # Save individual pair results
OUTPUT_DIR = "benchmark_results"

# ===============================================================================


class OptimizedIrisRecognition:
    """
    Optimized iris recognition system using the best 90 dimensions identified
    through comprehensive dimension analysis.
    """
    
    def __init__(self, model_id: str = MODEL_ID, optimal_dims: int = OPTIMAL_DIMENSIONS):
        self.model_id = model_id
        self.optimal_dims = optimal_dims
        self.top_dimensions = None
        
        logger.info(f"🔧 Initializing Optimized Iris Recognition System...")
        logger.info(f"   Model: {model_id}")
        logger.info(f"   Optimal dimensions: {optimal_dims}")
        
        # Load model components
        self._load_model()
        
        # Load or compute optimal dimensions
        self._load_optimal_dimensions()
    
    def _load_model(self):
        """Load the Gemma model and processors with optimized memory usage."""
        logger.info("📥 Loading model and processors...")
        
        # Optimize model loading for memory efficiency
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # Use bfloat16 to reduce memory
            low_cpu_mem_usage=True,      # Reduce CPU memory during loading
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        ).eval()
        
        # Optimize model for inference only
        self.model.generation_config.do_sample = False
        self.model.generation_config.max_new_tokens = 1  # We don't need text generation

        # Clear unnecessary components for embedding-only usage
        if hasattr(self.model, 'lm_head'):
            # We don't need the language model head for embeddings
            self.model.lm_head = None

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.processor2 = Gemma3ImageProcessor.from_pretrained(self.model_id)
        
        # Print actual memory usage
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"   GPU Memory - Allocated: {allocated_gb:.2f}GB, Reserved: {reserved_gb:.2f}GB")
        else:
            logger.info("✅ Model loaded successfully!")

    def _load_optimal_dimensions(self):
        """Load the optimal dimension indices from saved analysis results."""
        logger.info("🔍 Loading optimal dimension indices...")
        
        optimal_dims_file = "optimal_iris_dimensions.json"

        try:
            # Try to load from saved analysis results
            with open(optimal_dims_file, 'r') as f:
                optimal_data = json.load(f)

            self.top_dimensions = np.array(optimal_data["top_dimension_indices"])
            self.optimal_dims = optimal_data["optimal_dimension_count"]

            logger.info(f"✅ Loaded optimal dimensions from {optimal_dims_file}")
            logger.info(f"   Using {self.optimal_dims} dimensions")
            logger.info(f"   Analysis timestamp: {optimal_data['analysis_timestamp']}")
            logger.info(f"   Discrimination ratio: {optimal_data['discrimination_ratio']:.6f}")
            logger.info(f"   Separation: {optimal_data['separation']:.6f}")

        except FileNotFoundError:
            logger.error(f"❌ {optimal_dims_file} not found!")
            logger.error("   Please run: python transformers_embeddings.py dimension")
            logger.error("   This will generate the optimal dimensions file automatically.")
            raise FileNotFoundError(
                f"Optimal dimensions file '{optimal_dims_file}' not found. "
                "Run 'python transformers_embeddings.py dimension' first to generate it."
            )

        except Exception as e:
            logger.error(f"❌ Error loading optimal dimensions: {e}")
            logger.warning("⚠️  Falling back to placeholder dimensions")
            # Use evenly spaced dimensions as fallback
            total_dims = 2560
            self.top_dimensions = np.linspace(0, total_dims-1, self.optimal_dims, dtype=int)
            logger.warning(f"   Using {len(self.top_dimensions)} evenly-spaced dimensions as fallback")

        logger.info(f"✅ Ready to use top {len(self.top_dimensions)} dimensions for iris recognition")

    def get_full_embedding(self, image_path: str) -> np.ndarray:
        """Extract full embedding from iris image."""
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor2(images=image, return_tensors="pt").to(
                self.model.device, dtype=torch.bfloat16
            )
            
            with torch.no_grad():
                image_outputs = self.model.vision_tower(inputs.pixel_values)
                selected_image_feature = image_outputs.last_hidden_state
                image_embeddings = self.model.multi_modal_projector(selected_image_feature)
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
    
    def get_full_embeddings_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        """Extract full embeddings from multiple iris images in a single batch."""
        try:
            # Load and preprocess all images
            images = []
            valid_indices = []

            for i, image_path in enumerate(image_paths):
                try:
                    image = Image.open(image_path).convert('RGB')
                    images.append(image)
                    valid_indices.append(i)
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {e}")

            if not images:
                return [None] * len(image_paths)

            # If too many images, process in smaller sub-batches to avoid OOM
            max_images_per_batch = 100  # Conservative limit for RTX 3090
            if len(images) > max_images_per_batch:
                logger.info(f"   Large batch ({len(images)} images), processing in sub-batches of {max_images_per_batch}")

                all_embeddings = []
                for i in range(0, len(images), max_images_per_batch):
                    sub_images = images[i:i + max_images_per_batch]
                    sub_paths = image_paths[i:i + max_images_per_batch]

                    # Process sub-batch
                    sub_embeddings = self._process_image_batch(sub_images)
                    all_embeddings.extend(sub_embeddings)

                    # Clear GPU cache between sub-batches
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Map results back to original indices
                results = [None] * len(image_paths)
                for i, valid_idx in enumerate(valid_indices):
                    if i < len(all_embeddings):
                        results[valid_idx] = all_embeddings[i]

                return results
            else:
                # Process normally for smaller batches
                return self._process_image_batch(images, image_paths, valid_indices)

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Fallback to individual processing
            logger.info("   Falling back to individual image processing...")
            return self._fallback_individual_processing(image_paths)

    def _process_image_batch(self, images: List, image_paths: List[str] = None, valid_indices: List[int] = None) -> List[np.ndarray]:
        """Process a batch of images through the model."""
        try:
            # Batch process all images at once
            inputs = self.processor2(images=images, return_tensors="pt")

            # Move to device and convert to the right dtype
            pixel_values = inputs.pixel_values.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                pixel_values = pixel_values.to(self.model.dtype)
            else:
                pixel_values = pixel_values.to(torch.bfloat16)

            with torch.no_grad():
                # Process entire batch through the model
                image_outputs = self.model.vision_tower(pixel_values)
                selected_image_features = image_outputs.last_hidden_state
                image_embeddings = self.model.multi_modal_projector(selected_image_features)
                image_embedding_vectors = image_embeddings.mean(dim=1)

            # Convert to numpy and normalize
            batch_embeddings = image_embedding_vectors.cpu().float().numpy()

            # Normalize each embedding to unit vector
            normalized_embeddings = []
            for embedding in batch_embeddings:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                normalized_embeddings.append(embedding)

            # Handle mapping back to original indices if provided
            if image_paths is not None and valid_indices is not None:
                results = [None] * len(image_paths)
                for i, valid_idx in enumerate(valid_indices):
                    results[valid_idx] = normalized_embeddings[i]
                return results
            else:
                return normalized_embeddings

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM in batch processing: {e}")
            # Clear cache and try again with smaller batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e

    def _fallback_individual_processing(self, image_paths: List[str]) -> List[np.ndarray]:
        """Fallback to processing images individually if batch processing fails."""
        results = []
        for image_path in image_paths:
            embedding = self.get_full_embedding(image_path)
            results.append(embedding)
        return results

    def get_optimized_embedding(self, image_path: str) -> np.ndarray:
        """Extract optimized embedding using only the most discriminative dimensions."""
        # Get full embedding first
        full_embedding = self.get_full_embedding(image_path)
        
        if full_embedding is None:
            return None
        
        # Extract only the most discriminative dimensions
        optimized_embedding = full_embedding[self.top_dimensions]
        
        # Renormalize
        norm = np.linalg.norm(optimized_embedding)
        if norm > 0:
            optimized_embedding = optimized_embedding / norm
        
        return optimized_embedding
    
    def get_optimized_embeddings_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        """Extract optimized embeddings from multiple iris images in a single batch."""
        # Get full embeddings for the batch
        full_embeddings = self.get_full_embeddings_batch(image_paths)

        # Extract optimized dimensions for each embedding
        optimized_embeddings = []
        for full_embedding in full_embeddings:
            if full_embedding is None:
                optimized_embeddings.append(None)
            else:
                # Extract only the most discriminative dimensions
                optimized_embedding = full_embedding[self.top_dimensions]

                # Renormalize
                norm = np.linalg.norm(optimized_embedding)
                if norm > 0:
                    optimized_embedding = optimized_embedding / norm

                optimized_embeddings.append(optimized_embedding)

        return optimized_embeddings

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Cosine similarity (embeddings are already normalized)
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def verify_iris_pair(self, image1_path: str, image2_path: str, 
                        threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, Any]:
        """Verify if two iris images belong to the same person."""
        # Extract optimized embeddings
        emb1 = self.get_optimized_embedding(image1_path)
        emb2 = self.get_optimized_embedding(image2_path)
        
        # Compute similarity
        similarity = self.compute_similarity(emb1, emb2)
        
        # Make decision
        is_same_person = similarity >= threshold
        confidence = "high" if abs(similarity - threshold) > 0.02 else "medium"
        
        return {
            "image1": str(image1_path),
            "image2": str(image2_path),
            "similarity": similarity,
            "threshold": threshold,
            "is_same_person": is_same_person,
            "confidence": confidence,
            "embedding_success": emb1 is not None and emb2 is not None
        }


class IrisRecognitionBenchmark:
    """
    Comprehensive benchmark for the optimized iris recognition system.
    """
    
    def __init__(self, iris_system: OptimizedIrisRecognition):
        self.iris_system = iris_system
        self.results = {}
        
        # Create output directory
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    def load_dataset(self, max_pairs: int = None, quick_test: bool = False) -> Tuple[List[Tuple], Dict[str, Any]]:
        """Load IITD dataset with optional limitation."""
        logger.info("📂 Loading IITD dataset...")
        dataset = LoadIITDDataset()
        
        # Get dataset statistics
        stats = {
            "total_subjects": len(dataset.subject_images),
            "total_images": sum(len(images) for images in dataset.subject_images.values()),
            "total_genuine_pairs": len(dataset.genuine_pairs),
            "total_impostor_pairs": len(dataset.impostor_pairs),
            "total_pairs": len(dataset.pairs)
        }
        
        # Handle quick test mode (2k pairs with ~100 genuine, rest impostor)
        if quick_test:
            logger.info("🚀 Quick test mode: Loading ~2000 pairs (100 genuine, 1900 impostor)")
            import random

            # Sample genuine pairs
            genuine_sample_size = 100
            genuine_sample = random.sample(dataset.genuine_pairs,
                                         min(genuine_sample_size, len(dataset.genuine_pairs)))

            # Sample impostor pairs
            impostor_sample_size = 1900
            impostor_sample = random.sample(dataset.impostor_pairs,
                                          min(impostor_sample_size, len(dataset.impostor_pairs)))

            # Combine and shuffle
            test_pairs = genuine_sample + impostor_sample
            random.shuffle(test_pairs)

            stats["test_pairs"] = len(test_pairs)
            stats["test_genuine"] = len(genuine_sample)
            stats["test_impostor"] = len(impostor_sample)

        # Limit pairs if requested (original logic)
        elif max_pairs:
            # Get balanced sample
            genuine_count = min(max_pairs // 2, len(dataset.genuine_pairs))
            impostor_count = max_pairs - genuine_count
            
            test_pairs = (dataset.genuine_pairs[:genuine_count] + 
                         dataset.impostor_pairs[:impostor_count])
            
            # Shuffle for randomness
            import random
            random.shuffle(test_pairs)
            
            stats["test_pairs"] = len(test_pairs)
            stats["test_genuine"] = genuine_count
            stats["test_impostor"] = impostor_count
        else:
            # Use full dataset
            test_pairs = dataset.pairs
            stats["test_pairs"] = len(test_pairs)
            stats["test_genuine"] = len(dataset.genuine_pairs)
            stats["test_impostor"] = len(dataset.impostor_pairs)
        
        logger.info(f"📊 Dataset loaded: {stats['test_pairs']:,} pairs to test")
        logger.info(f"   Genuine pairs: {stats['test_genuine']:,}")
        logger.info(f"   Impostor pairs: {stats['test_impostor']:,}")

        return test_pairs, stats
    
    def process_pairs_batch(self, pairs: List[Tuple], start_idx: int, batch_size: int) -> List[Dict]:
        """Process a batch of pairs using true GPU batching."""
        batch_pairs = pairs[start_idx:start_idx + batch_size]

        # Collect all unique images in the batch
        all_images = []
        image_to_idx = {}

        for img1, img2, label in batch_pairs:
            img1_str = str(img1)
            img2_str = str(img2)

            if img1_str not in image_to_idx:
                image_to_idx[img1_str] = len(all_images)
                all_images.append(img1_str)

            if img2_str not in image_to_idx:
                image_to_idx[img2_str] = len(all_images)
                all_images.append(img2_str)

        # Process all unique images in a single batch
        logger.info(f"   Processing {len(all_images)} unique images in GPU batch...")
        batch_embeddings = self.iris_system.get_optimized_embeddings_batch(all_images)

        # Create embedding lookup dictionary
        embedding_lookup = {}
        for i, img_path in enumerate(all_images):
            embedding_lookup[img_path] = batch_embeddings[i]

        # Process each pair using pre-computed embeddings
        batch_results = []
        for i, (img1, img2, label) in enumerate(batch_pairs):
            pair_idx = start_idx + i
            
            # Get embeddings from lookup
            emb1 = embedding_lookup[str(img1)]
            emb2 = embedding_lookup[str(img2)]

            # Compute similarity
            similarity = self.iris_system.compute_similarity(emb1, emb2)

            # Make decision
            is_same_person = similarity >= SIMILARITY_THRESHOLD
            confidence = "high" if abs(similarity - SIMILARITY_THRESHOLD) > 0.02 else "medium"

            # Create result
            result = {
                "image1": str(img1),
                "image2": str(img2),
                "similarity": similarity,
                "threshold": SIMILARITY_THRESHOLD,
                "is_same_person": is_same_person,
                "confidence": confidence,
                "embedding_success": emb1 is not None and emb2 is not None,
                "pair_id": pair_idx,
                "ground_truth": label,
                "pair_type": "genuine" if label == 1 else "impostor",
                "prediction_correct": is_same_person == (label == 1),
                "image1_name": img1.name,
                "image2_name": img2.name,
                "subject1": img1.parent.name,
                "subject2": img2.parent.name
            }

            batch_results.append(result)
        
        return batch_results
    
    def run_benchmark(self, max_pairs: int = MAX_PAIRS_TO_TEST, quick_test: bool = False) -> Dict[str, Any]:
        """Run comprehensive benchmark on the dataset."""
        logger.info("🚀 Starting Optimized Iris Recognition Benchmark")
        logger.info("=" * 60)
        
        # Load dataset
        test_pairs, dataset_stats = self.load_dataset(max_pairs, quick_test)

        # Initialize tracking variables
        all_results = []
        correct_predictions = 0
        total_pairs = len(test_pairs)
        failed_embeddings = 0
        
        # Timing
        start_time = time.time()
        
        # Process pairs in batches with tqdm progress bar
        logger.info(f"📊 Processing {total_pairs:,} pairs in batches of {BATCH_SIZE}...")
        
        # Create progress bar
        pbar = tqdm(range(0, total_pairs, BATCH_SIZE),
                   desc="Processing batches",
                   unit="batch",
                   ncols=100)

        for batch_start in pbar:
            batch_end = min(batch_start + BATCH_SIZE, total_pairs)
            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (total_pairs + BATCH_SIZE - 1) // BATCH_SIZE
            
            # Update progress bar description
            pbar.set_description(f"Batch {batch_num}/{total_batches}")

            # Process batch
            batch_results = self.process_pairs_batch(test_pairs, batch_start, BATCH_SIZE)
            all_results.extend(batch_results)
            
            # Update counters
            batch_correct = 0
            batch_failed = 0
            for result in batch_results:
                if result["prediction_correct"]:
                    correct_predictions += 1
                    batch_correct += 1
                if not result["embedding_success"]:
                    failed_embeddings += 1
                    batch_failed += 1

            # Update progress bar with current accuracy
            current_accuracy = correct_predictions / len(all_results) if all_results else 0
            pbar.set_postfix({
                'Accuracy': f'{current_accuracy:.3f}',
                'Processed': f'{len(all_results):,}',
                'Failed': batch_failed
            })

            # Clear GPU cache periodically
            if torch.cuda.is_available() and batch_num % 10 == 0:
                torch.cuda.empty_cache()
        
        pbar.close()

        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(all_results, total_time, dataset_stats)
        
        # Save results
        if SAVE_DETAILED_RESULTS:
            self._save_detailed_results(all_results, metrics)
        
        self._save_summary_results(metrics)
        
        # Print summary
        self._print_benchmark_summary(metrics)
        
        return {
            "metrics": metrics,
            "detailed_results": all_results if SAVE_DETAILED_RESULTS else None
        }
    
    def _calculate_metrics(self, results: List[Dict], total_time: float, 
                          dataset_stats: Dict) -> Dict[str, Any]:
        """Calculate comprehensive benchmark metrics."""
        total_pairs = len(results)
        correct_predictions = sum(1 for r in results if r["prediction_correct"])
        failed_embeddings = sum(1 for r in results if not r["embedding_success"])
        
        # Separate genuine and impostor results
        genuine_results = [r for r in results if r["pair_type"] == "genuine"]
        impostor_results = [r for r in results if r["pair_type"] == "impostor"]
        
        # Calculate confusion matrix
        tp = sum(1 for r in genuine_results if r["is_same_person"])  # True Positive
        fn = sum(1 for r in genuine_results if not r["is_same_person"])  # False Negative
        fp = sum(1 for r in impostor_results if r["is_same_person"])  # False Positive
        tn = sum(1 for r in impostor_results if not r["is_same_person"])  # True Negative
        
        # Calculate metrics
        accuracy = (tp + tn) / total_pairs if total_pairs > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Similarity statistics
        genuine_similarities = [r["similarity"] for r in genuine_results if r["embedding_success"]]
        impostor_similarities = [r["similarity"] for r in impostor_results if r["embedding_success"]]
        
        return {
            "dataset_stats": dataset_stats,
            "performance": {
                "total_pairs": total_pairs,
                "correct_predictions": correct_predictions,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "specificity": specificity
            },
            "confusion_matrix": {
                "true_positive": tp,
                "false_negative": fn,
                "false_positive": fp,
                "true_negative": tn
            },
            "similarity_stats": {
                "genuine": {
                    "count": len(genuine_similarities),
                    "mean": np.mean(genuine_similarities) if genuine_similarities else 0,
                    "std": np.std(genuine_similarities) if genuine_similarities else 0,
                    "min": np.min(genuine_similarities) if genuine_similarities else 0,
                    "max": np.max(genuine_similarities) if genuine_similarities else 0
                },
                "impostor": {
                    "count": len(impostor_similarities),
                    "mean": np.mean(impostor_similarities) if impostor_similarities else 0,
                    "std": np.std(impostor_similarities) if impostor_similarities else 0,
                    "min": np.min(impostor_similarities) if impostor_similarities else 0,
                    "max": np.max(impostor_similarities) if impostor_similarities else 0
                },
                "separation": (np.mean(genuine_similarities) - np.mean(impostor_similarities)) 
                             if genuine_similarities and impostor_similarities else 0
            },
            "timing": {
                "total_time_seconds": total_time,
                "avg_time_per_pair": total_time / total_pairs if total_pairs > 0 else 0,
                "pairs_per_second": total_pairs / total_time if total_time > 0 else 0
            },
            "system_info": {
                "model_id": self.iris_system.model_id,
                "optimal_dimensions": self.iris_system.optimal_dims,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "failed_embeddings": failed_embeddings,
                "embedding_success_rate": (total_pairs - failed_embeddings) / total_pairs if total_pairs > 0 else 0
            }
        }
    
    def _save_detailed_results(self, results: List[Dict], metrics: Dict):
        """Save detailed results to JSON file."""
        output_file = Path(OUTPUT_DIR) / "detailed_results.json"
        
        output_data = {
            "benchmark_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": self.iris_system.model_id,
                "optimal_dimensions": self.iris_system.optimal_dims,
                "threshold": SIMILARITY_THRESHOLD
            },
            "metrics": metrics,
            "detailed_results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"💾 Detailed results saved to: {output_file}")
    
    def _save_summary_results(self, metrics: Dict):
        """Save summary results to JSON file."""
        output_file = Path(OUTPUT_DIR) / "benchmark_summary.json"
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"💾 Summary results saved to: {output_file}")
    
    def _print_benchmark_summary(self, metrics: Dict):
        """Print comprehensive benchmark summary."""
        print("\n" + "=" * 80)
        print("🏆 OPTIMIZED IRIS RECOGNITION BENCHMARK RESULTS")
        print("=" * 80)
        
        # Dataset info
        ds = metrics["dataset_stats"]
        print(f"📊 Dataset Statistics:")
        print(f"   Total subjects: {ds['total_subjects']:,}")
        print(f"   Total images: {ds['total_images']:,}")
        print(f"   Pairs tested: {ds['test_pairs']:,}")
        print(f"   Genuine pairs: {ds['test_genuine']:,}")
        print(f"   Impostor pairs: {ds['test_impostor']:,}")
        
        # Performance metrics
        perf = metrics["performance"]
        print(f"\n🎯 Performance Metrics:")
        print(f"   Accuracy: {perf['accuracy']:.4f} ({perf['accuracy']*100:.2f}%)")
        print(f"   Precision: {perf['precision']:.4f}")
        print(f"   Recall: {perf['recall']:.4f}")
        print(f"   F1-Score: {perf['f1_score']:.4f}")
        print(f"   Specificity: {perf['specificity']:.4f}")
        
        # Confusion matrix
        cm = metrics["confusion_matrix"]
        print(f"\n📈 Confusion Matrix:")
        print(f"   True Positive:  {cm['true_positive']:,}")
        print(f"   False Negative: {cm['false_negative']:,}")
        print(f"   False Positive: {cm['false_positive']:,}")
        print(f"   True Negative:  {cm['true_negative']:,}")
        
        # Similarity statistics
        sim = metrics["similarity_stats"]
        print(f"\n📊 Similarity Statistics:")
        print(f"   Genuine pairs:  {sim['genuine']['mean']:.6f} ± {sim['genuine']['std']:.6f}")
        print(f"   Impostor pairs: {sim['impostor']['mean']:.6f} ± {sim['impostor']['std']:.6f}")
        print(f"   Separation:     {sim['separation']:.6f}")
        
        # Timing info
        timing = metrics["timing"]
        print(f"\n⏱️  Performance Timing:")
        print(f"   Total time: {timing['total_time_seconds']:.2f} seconds")
        print(f"   Avg per pair: {timing['avg_time_per_pair']:.4f} seconds")
        print(f"   Throughput: {timing['pairs_per_second']:.2f} pairs/second")
        
        # System info
        sys_info = metrics["system_info"]
        print(f"\n🔧 System Configuration:")
        print(f"   Model: {sys_info['model_id']}")
        print(f"   Dimensions: {sys_info['optimal_dimensions']} (96.5% reduction)")
        print(f"   Threshold: {sys_info['similarity_threshold']}")
        print(f"   Embedding success: {sys_info['embedding_success_rate']*100:.2f}%")


def main():
    """Main benchmark execution."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimized Iris Recognition Benchmark")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with ~2000 pairs (100 genuine, 1900 impostor)")
    parser.add_argument("--max-pairs", type=int, default=None,
                       help="Maximum number of pairs to test (balanced genuine/impostor)")
    args = parser.parse_args()

    logger.info("🔬 Initializing Optimized Iris Recognition Benchmark")
    
    if args.quick:
        logger.info("🚀 Running in QUICK TEST mode")
    elif args.max_pairs:
        logger.info(f"📊 Testing with maximum {args.max_pairs:,} pairs")
    else:
        logger.info("📈 Running FULL DATASET benchmark")

    # Initialize system
    iris_system = OptimizedIrisRecognition()
    
    # Initialize benchmark
    benchmark = IrisRecognitionBenchmark(iris_system)
    
    # Run benchmark with appropriate settings
    results = benchmark.run_benchmark(
        max_pairs=args.max_pairs,
        quick_test=args.quick
    )

    logger.info("✅ Benchmark completed successfully!")
    
    return results


if __name__ == "__main__":
    results = main()
