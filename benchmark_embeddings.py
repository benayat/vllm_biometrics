#!/usr/bin/env python3
"""
Iris Embedding Performance Benchmark Script

This script benchmarks embedding extraction performance for iris images,
testing different batch sizes and measuring timing per image.
"""

import asyncio
import sys
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from vlm_client.embedding_client import EmbeddingClient
from dataloader.load_iitd_dataset import LoadIITDDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===============================================================================
# BENCHMARK CONFIGURATION - EDIT THESE TO CUSTOMIZE YOUR TESTING
# ===============================================================================

# Number of images to test (set to None for all images)
NUM_IMAGES = 100  # Set to None to test all 2,180 images

# Batch sizes to test
BATCH_SIZES = [1, 2, 4, 8, 16, 32]

# Embedding method to use
EMBEDDING_METHOD = "chat_based"  # "chat_based" or "embeddings"

# Output results to file
SAVE_RESULTS = True
RESULTS_FILE = "embedding_benchmark_results.json"

# ===============================================================================


class EmbeddingBenchmark:
    """Class to benchmark iris embedding extraction performance."""
    
    def __init__(self, num_images: Optional[int] = None):
        self.num_images = num_images
        self.results = {}
        
    def load_test_images(self) -> List[str]:
        """Load test images from IITD dataset."""
        print("Loading IITD dataset...")
        dataset = LoadIITDDataset()
        
        # Collect all image paths
        all_images = []
        for subject_id, images in dataset.subject_images.items():
            for image_path in images:
                all_images.append(str(image_path))
        
        # Limit to specified number if requested
        if self.num_images and self.num_images < len(all_images):
            all_images = all_images[:self.num_images]
        
        print(f"Selected {len(all_images)} images for benchmarking")
        return all_images
    
    async def benchmark_single_image(self, image_path: str, client: EmbeddingClient) -> Dict[str, Any]:
        """Benchmark embedding extraction for a single image."""
        start_time = time.time()
        
        try:
            embedding = await client.get_iris_embedding(image_path, method=EMBEDDING_METHOD)
            end_time = time.time()
            
            return {
                "success": True,
                "duration": end_time - start_time,
                "embedding_shape": embedding.shape if embedding is not None else None,
                "error": None
            }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "duration": end_time - start_time,
                "embedding_shape": None,
                "error": str(e)
            }
    
    async def benchmark_batch_processing(self, image_paths: List[str], batch_size: int) -> Dict[str, Any]:
        """Benchmark batch processing with specified batch size."""
        print(f"\n🔄 Testing batch size: {batch_size}")
        print(f"Processing {len(image_paths)} images...")
        
        start_time = time.time()
        
        async with EmbeddingClient() as client:
            try:
                embeddings = await client.get_iris_embeddings_batch(
                    image_paths, 
                    batch_size=batch_size, 
                    method=EMBEDDING_METHOD
                )
                end_time = time.time()
                
                # Calculate statistics
                total_duration = end_time - start_time
                successful = sum(1 for emb in embeddings if emb is not None)
                failed = len(embeddings) - successful
                avg_time_per_image = total_duration / len(image_paths)
                throughput = len(image_paths) / total_duration  # images per second
                
                result = {
                    "batch_size": batch_size,
                    "total_images": len(image_paths),
                    "successful": successful,
                    "failed": failed,
                    "total_duration": total_duration,
                    "avg_time_per_image": avg_time_per_image,
                    "throughput_images_per_sec": throughput,
                    "success_rate": successful / len(image_paths) * 100,
                    "method": EMBEDDING_METHOD
                }
                
                print(f"   ✅ Completed in {total_duration:.2f}s")
                print(f"   📊 Avg time per image: {avg_time_per_image:.3f}s")
                print(f"   🚀 Throughput: {throughput:.2f} images/sec")
                print(f"   ✓ Success rate: {result['success_rate']:.1f}%")
                
                return result
                
            except Exception as e:
                end_time = time.time()
                print(f"   ❌ Batch failed: {e}")
                return {
                    "batch_size": batch_size,
                    "total_images": len(image_paths),
                    "error": str(e),
                    "duration": end_time - start_time
                }
    
    async def run_single_image_benchmark(self, image_paths: List[str]) -> Dict[str, Any]:
        """Run detailed single-image benchmark."""
        print(f"\n🔍 Single Image Benchmark")
        print("=" * 40)
        
        # Test first 10 images individually for detailed timing
        test_images = image_paths[:min(10, len(image_paths))]
        durations = []
        
        async with EmbeddingClient() as client:
            for i, image_path in enumerate(test_images):
                print(f"Processing image {i+1}/{len(test_images)}: {Path(image_path).name}")
                
                result = await self.benchmark_single_image(image_path, client)
                
                if result["success"]:
                    durations.append(result["duration"])
                    print(f"   ⏱️  Duration: {result['duration']:.3f}s")
                else:
                    print(f"   ❌ Failed: {result['error']}")
        
        if durations:
            stats = {
                "count": len(durations),
                "min_time": min(durations),
                "max_time": max(durations),
                "avg_time": statistics.mean(durations),
                "median_time": statistics.median(durations),
                "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0
            }
            
            print(f"\n📈 Single Image Statistics:")
            print(f"   Count: {stats['count']}")
            print(f"   Average: {stats['avg_time']:.3f}s")
            print(f"   Median: {stats['median_time']:.3f}s")
            print(f"   Min: {stats['min_time']:.3f}s")
            print(f"   Max: {stats['max_time']:.3f}s")
            print(f"   Std Dev: {stats['std_dev']:.3f}s")
            
            return stats
        
        return {"error": "No successful single image extractions"}
    
    async def run_batch_size_comparison(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Compare different batch sizes."""
        print(f"\n📦 Batch Size Comparison")
        print("=" * 40)
        
        batch_results = []
        
        for batch_size in BATCH_SIZES:
            # Limit images for smaller tests with larger batch sizes
            test_images = image_paths[:min(len(image_paths), batch_size * 10)]
            
            result = await self.benchmark_batch_processing(test_images, batch_size)
            batch_results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        return batch_results
    
    def analyze_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze batch size performance results."""
        print(f"\n📊 Batch Size Analysis")
        print("=" * 40)
        
        successful_results = [r for r in batch_results if "error" not in r]
        
        if not successful_results:
            print("❌ No successful batch results to analyze")
            return {"error": "No successful results"}
        
        # Find optimal batch size
        best_throughput = max(successful_results, key=lambda x: x.get("throughput_images_per_sec", 0))
        best_avg_time = min(successful_results, key=lambda x: x.get("avg_time_per_image", float('inf')))
        
        print(f"🏆 Best Throughput:")
        print(f"   Batch size: {best_throughput['batch_size']}")
        print(f"   Throughput: {best_throughput['throughput_images_per_sec']:.2f} images/sec")
        
        print(f"\n⚡ Best Average Time:")
        print(f"   Batch size: {best_avg_time['batch_size']}")
        print(f"   Avg time: {best_avg_time['avg_time_per_image']:.3f}s per image")
        
        # Detailed comparison table
        print(f"\n📋 Detailed Comparison:")
        print(f"{'Batch':<6} {'Images':<7} {'Total(s)':<9} {'Avg(s)':<8} {'Throughput':<11} {'Success%':<8}")
        print("-" * 60)
        
        for result in successful_results:
            print(f"{result['batch_size']:<6} {result['total_images']:<7} "
                  f"{result['total_duration']:<9.2f} {result['avg_time_per_image']:<8.3f} "
                  f"{result['throughput_images_per_sec']:<11.2f} {result['success_rate']:<8.1f}")
        
        return {
            "best_throughput": best_throughput,
            "best_avg_time": best_avg_time,
            "all_results": successful_results
        }
    
    def save_results(self, results: Dict[str, Any]):
        """Save benchmark results to JSON file."""
        if SAVE_RESULTS:
            import json
            
            # Add metadata
            results["metadata"] = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_images_tested": self.num_images,
                "embedding_method": EMBEDDING_METHOD,
                "batch_sizes_tested": BATCH_SIZES
            }
            
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {RESULTS_FILE}")
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("🚀 IRIS EMBEDDING PERFORMANCE BENCHMARK")
        print("=" * 50)
        print(f"Configuration:")
        print(f"   Images to test: {self.num_images or 'ALL'}")
        print(f"   Batch sizes: {BATCH_SIZES}")
        print(f"   Method: {EMBEDDING_METHOD}")
        print("=" * 50)
        
        # Load test images
        image_paths = self.load_test_images()
        
        # Run benchmarks
        results = {}
        
        # 1. Single image benchmark
        single_stats = await self.run_single_image_benchmark(image_paths)
        results["single_image_stats"] = single_stats
        
        # 2. Batch size comparison
        batch_results = await self.run_batch_size_comparison(image_paths)
        results["batch_results"] = batch_results
        
        # 3. Analysis
        analysis = self.analyze_batch_results(batch_results)
        results["analysis"] = analysis
        
        # Save results
        self.save_results(results)
        
        return results


async def main():
    """Main benchmark function."""
    benchmark = EmbeddingBenchmark(num_images=NUM_IMAGES)
    
    try:
        # Check server connection first
        async with EmbeddingClient() as client:
            print("🔍 Testing server connection...")
            
        results = await benchmark.run_full_benchmark()
        
        print(f"\n🎯 Benchmark completed successfully!")
        
        # Summary
        if "analysis" in results and "best_throughput" in results["analysis"]:
            best = results["analysis"]["best_throughput"]
            print(f"\n🏆 Optimal Configuration:")
            print(f"   Batch size: {best['batch_size']}")
            print(f"   Throughput: {best['throughput_images_per_sec']:.2f} images/sec")
            print(f"   Time per image: {best['avg_time_per_image']:.3f}s")
            
            # Estimate time for full dataset
            if NUM_IMAGES and NUM_IMAGES < 2180:
                full_dataset_time = 2180 * best['avg_time_per_image']
                print(f"\n⏰ Estimated time for full dataset (2,180 images):")
                print(f"   {full_dataset_time:.0f} seconds ({full_dataset_time/60:.1f} minutes)")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("Make sure the vLLM server is running: ./vllm_serve_gemma.sh")


if __name__ == "__main__":
    asyncio.run(main())
