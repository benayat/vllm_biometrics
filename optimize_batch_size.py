#!/usr/bin/env python3
"""
Batch Size Optimization for RTX 3090

This script finds the optimal batch size for your RTX 3090 by testing different values
and measuring GPU memory usage, throughput, and stability.
"""

import torch
import time
import json
import numpy as np
from pathlib import Path
import sys
import gc
import psutil
from typing import Dict, List, Tuple
import logging

# Add the project root to the path for imports
sys.path.append(str(Path(__file__).parent))
from benchmark_optimized_iris import OptimizedIrisRecognition
from dataloader.load_iitd_dataset import LoadIITDDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchSizeOptimizer:
    """
    Optimize batch size for RTX 3090 by testing different values
    and measuring performance metrics.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_memory_gb = self._get_gpu_memory()
        self.iris_system = None
        self.test_images = []
        
        logger.info(f"🔧 BatchSizeOptimizer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   GPU Memory: {self.gpu_memory_gb:.1f} GB")
    
    def _get_gpu_memory(self) -> float:
        """Get total GPU memory in GB."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0.0
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['gpu_cached'] = torch.cuda.memory_reserved() / (1024**3)
            memory_info['gpu_free'] = (torch.cuda.get_device_properties(0).total_memory - 
                                     torch.cuda.memory_allocated()) / (1024**3)
        
        memory_info['cpu_percent'] = psutil.virtual_memory().percent
        memory_info['cpu_available'] = psutil.virtual_memory().available / (1024**3)
        
        return memory_info
    
    def _clear_memory(self):
        """Clear GPU and CPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _load_test_images(self, num_images: int = 100) -> List[str]:
        """Load a subset of test images for batch size testing."""
        logger.info(f"📂 Loading {num_images} test images...")
        
        dataset = LoadIITDDataset()
        
        # Get diverse set of images from different subjects
        test_images = []
        images_per_subject = max(1, num_images // len(dataset.subject_images))
        
        for subject, images in dataset.subject_images.items():
            subject_images = images[:images_per_subject]
            test_images.extend([str(img) for img in subject_images])
            
            if len(test_images) >= num_images:
                break
        
        # Limit to requested number
        test_images = test_images[:num_images]
        
        logger.info(f"✅ Loaded {len(test_images)} test images")
        return test_images
    
    def _test_batch_size(self, batch_size: int, test_images: List[str], 
                        num_iterations: int = 3) -> Dict[str, float]:
        """Test a specific batch size and measure performance."""
        logger.info(f"🧪 Testing batch size: {batch_size}")
        
        # Clear memory before test
        self._clear_memory()
        
        # Initialize iris system if not already done
        if self.iris_system is None:
            self.iris_system = OptimizedIrisRecognition()
        
        results = {
            'batch_size': batch_size,
            'success': False,
            'avg_time_per_batch': 0.0,
            'avg_time_per_image': 0.0,
            'throughput_images_per_sec': 0.0,
            'peak_gpu_memory': 0.0,
            'avg_gpu_memory': 0.0,
            'memory_efficiency': 0.0,
            'error': None
        }
        
        try:
            iteration_times = []
            memory_measurements = []
            
            # Test multiple iterations for stability
            for iteration in range(num_iterations):
                # Select batch of images
                start_idx = (iteration * batch_size) % len(test_images)
                end_idx = min(start_idx + batch_size, len(test_images))
                batch_images = test_images[start_idx:end_idx]
                
                # Measure memory before processing
                mem_before = self._get_memory_usage()
                
                # Process batch
                start_time = time.time()
                
                batch_embeddings = []
                for img_path in batch_images:
                    embedding = self.iris_system.get_optimized_embedding(img_path)
                    if embedding is not None:
                        batch_embeddings.append(embedding)
                
                end_time = time.time()
                batch_time = end_time - start_time
                
                # Measure memory after processing
                mem_after = self._get_memory_usage()
                
                # Record metrics
                iteration_times.append(batch_time)
                memory_measurements.append(mem_after['gpu_allocated'])
                
                # Clear memory between iterations
                del batch_embeddings
                self._clear_memory()
                
                logger.info(f"   Iteration {iteration + 1}: {batch_time:.3f}s, "
                           f"GPU: {mem_after['gpu_allocated']:.2f}GB")
            
            # Calculate averages
            avg_batch_time = np.mean(iteration_times)
            avg_image_time = avg_batch_time / batch_size
            throughput = batch_size / avg_batch_time
            avg_gpu_memory = np.mean(memory_measurements)
            peak_gpu_memory = np.max(memory_measurements)
            
            # Memory efficiency (higher is better)
            memory_efficiency = throughput / peak_gpu_memory if peak_gpu_memory > 0 else 0
            
            results.update({
                'success': True,
                'avg_time_per_batch': avg_batch_time,
                'avg_time_per_image': avg_image_time,
                'throughput_images_per_sec': throughput,
                'peak_gpu_memory': peak_gpu_memory,
                'avg_gpu_memory': avg_gpu_memory,
                'memory_efficiency': memory_efficiency
            })
            
            logger.info(f"   ✅ Success: {throughput:.2f} img/s, "
                       f"Peak GPU: {peak_gpu_memory:.2f}GB")
            
        except torch.cuda.OutOfMemoryError as e:
            results['error'] = f"GPU Out of Memory: {str(e)}"
            logger.warning(f"   ❌ GPU OOM at batch size {batch_size}")
            self._clear_memory()
            
        except Exception as e:
            results['error'] = f"Error: {str(e)}"
            logger.error(f"   ❌ Error at batch size {batch_size}: {e}")
            self._clear_memory()
        
        return results
    
    def find_optimal_batch_size(self, test_batch_sizes: List[int] = None) -> Dict[str, any]:
        """Find optimal batch size through systematic testing."""
        logger.info("🔍 Finding optimal batch size for RTX 3090...")
        logger.info("=" * 60)
        
        # Default batch sizes to test (tailored for RTX 3090)
        if test_batch_sizes is None:
            test_batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256]
        
        logger.info(f"📊 Testing batch sizes: {test_batch_sizes}")
        
        # Load test images
        test_images = self._load_test_images(200)  # Use 200 images for robust testing
        
        # Test each batch size
        all_results = []
        successful_results = []
        
        for batch_size in test_batch_sizes:
            result = self._test_batch_size(batch_size, test_images)
            all_results.append(result)
            
            if result['success']:
                successful_results.append(result)
            else:
                logger.warning(f"❌ Batch size {batch_size} failed: {result['error']}")
                # If we hit OOM, no point testing larger batch sizes
                if "Out of Memory" in result['error']:
                    logger.info("🛑 Stopping tests due to GPU memory limit")
                    break
        
        if not successful_results:
            logger.error("❌ No successful batch sizes found!")
            return {"error": "No successful batch sizes", "results": all_results}
        
        # Analyze results
        analysis = self._analyze_results(successful_results)
        
        # Save results
        self._save_optimization_results(all_results, analysis)
        
        # Print recommendations
        self._print_recommendations(analysis)
        
        return {
            "analysis": analysis,
            "all_results": all_results,
            "successful_results": successful_results
        }
    
    def _analyze_results(self, results: List[Dict]) -> Dict[str, any]:
        """Analyze batch size test results and find optimal values."""
        logger.info("📈 Analyzing batch size results...")
        
        # Find best performers for different metrics
        best_throughput = max(results, key=lambda x: x['throughput_images_per_sec'])
        best_memory_efficiency = max(results, key=lambda x: x['memory_efficiency'])
        lowest_latency = min(results, key=lambda x: x['avg_time_per_image'])
        
        # Find largest successful batch size
        largest_batch = max(results, key=lambda x: x['batch_size'])
        
        # Find sweet spot (good balance of throughput and memory efficiency)
        # Score = throughput * memory_efficiency
        for result in results:
            result['composite_score'] = (result['throughput_images_per_sec'] * 
                                       result['memory_efficiency'])
        
        best_overall = max(results, key=lambda x: x['composite_score'])
        
        # Conservative recommendation (80% of max successful batch size)
        conservative_batch = max(1, int(largest_batch['batch_size'] * 0.8))
        conservative_result = min(results, key=lambda x: abs(x['batch_size'] - conservative_batch))
        
        analysis = {
            'recommendations': {
                'best_throughput': {
                    'batch_size': best_throughput['batch_size'],
                    'throughput': best_throughput['throughput_images_per_sec'],
                    'memory_usage': best_throughput['peak_gpu_memory'],
                    'reason': 'Highest throughput (images/second)'
                },
                'best_memory_efficiency': {
                    'batch_size': best_memory_efficiency['batch_size'],
                    'throughput': best_memory_efficiency['throughput_images_per_sec'],
                    'memory_usage': best_memory_efficiency['peak_gpu_memory'],
                    'efficiency': best_memory_efficiency['memory_efficiency'],
                    'reason': 'Best throughput per GB of GPU memory'
                },
                'lowest_latency': {
                    'batch_size': lowest_latency['batch_size'],
                    'latency': lowest_latency['avg_time_per_image'],
                    'memory_usage': lowest_latency['peak_gpu_memory'],
                    'reason': 'Fastest processing per image'
                },
                'best_overall': {
                    'batch_size': best_overall['batch_size'],
                    'throughput': best_overall['throughput_images_per_sec'],
                    'memory_usage': best_overall['peak_gpu_memory'],
                    'efficiency': best_overall['memory_efficiency'],
                    'composite_score': best_overall['composite_score'],
                    'reason': 'Best balance of throughput and memory efficiency'
                },
                'conservative': {
                    'batch_size': conservative_result['batch_size'],
                    'throughput': conservative_result['throughput_images_per_sec'],
                    'memory_usage': conservative_result['peak_gpu_memory'],
                    'reason': 'Safe choice with headroom for memory spikes'
                }
            },
            'system_limits': {
                'max_successful_batch': largest_batch['batch_size'],
                'max_gpu_memory_used': max(r['peak_gpu_memory'] for r in results),
                'gpu_memory_available': self.gpu_memory_gb
            },
            'performance_curve': results
        }
        
        return analysis
    
    def _save_optimization_results(self, results: List[Dict], analysis: Dict):
        """Save optimization results to JSON file."""
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_info": {
                "total_memory_gb": self.gpu_memory_gb,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            },
            "analysis": analysis,
            "detailed_results": results
        }
        
        output_file = "batch_size_optimization_results.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"💾 Optimization results saved to: {output_file}")
    
    def _print_recommendations(self, analysis: Dict):
        """Print batch size recommendations."""
        print("\n" + "=" * 80)
        print("🏆 BATCH SIZE OPTIMIZATION RESULTS - RTX 3090")
        print("=" * 80)
        
        recs = analysis['recommendations']
        limits = analysis['system_limits']
        
        print(f"🔧 System Information:")
        print(f"   GPU Memory: {self.gpu_memory_gb:.1f} GB")
        print(f"   Max Successful Batch: {limits['max_successful_batch']}")
        print(f"   Peak Memory Used: {limits['max_gpu_memory_used']:.2f} GB")
        
        print(f"\n🏆 RECOMMENDED BATCH SIZES:")
        
        print(f"\n1. 🚀 BEST OVERALL (Recommended): {recs['best_overall']['batch_size']}")
        print(f"   Throughput: {recs['best_overall']['throughput']:.2f} images/sec")
        print(f"   Memory Usage: {recs['best_overall']['memory_usage']:.2f} GB")
        print(f"   Efficiency Score: {recs['best_overall']['composite_score']:.2f}")
        print(f"   Reason: {recs['best_overall']['reason']}")
        
        print(f"\n2. 🛡️  CONSERVATIVE (Safe): {recs['conservative']['batch_size']}")
        print(f"   Throughput: {recs['conservative']['throughput']:.2f} images/sec")
        print(f"   Memory Usage: {recs['conservative']['memory_usage']:.2f} GB")
        print(f"   Reason: {recs['conservative']['reason']}")
        
        print(f"\n3. ⚡ MAXIMUM THROUGHPUT: {recs['best_throughput']['batch_size']}")
        print(f"   Throughput: {recs['best_throughput']['throughput']:.2f} images/sec")
        print(f"   Memory Usage: {recs['best_throughput']['memory_usage']:.2f} GB")
        print(f"   Reason: {recs['best_throughput']['reason']}")
        
        print(f"\n4. 🎯 LOWEST LATENCY: {recs['lowest_latency']['batch_size']}")
        print(f"   Latency: {recs['lowest_latency']['latency']:.4f} sec/image")
        print(f"   Memory Usage: {recs['lowest_latency']['memory_usage']:.2f} GB")
        print(f"   Reason: {recs['lowest_latency']['reason']}")
        
        print(f"\n📝 USAGE RECOMMENDATIONS:")
        print(f"   • For production: Use batch size {recs['best_overall']['batch_size']}")
        print(f"   • For development: Use batch size {recs['conservative']['batch_size']}")
        print(f"   • For real-time: Use batch size {recs['lowest_latency']['batch_size']}")
        print(f"   • For maximum throughput: Use batch size {recs['best_throughput']['batch_size']}")


def main():
    """Main optimization execution."""
    print("🔬 RTX 3090 Batch Size Optimizer")
    print("This will test different batch sizes to find the optimal configuration.")
    print("=" * 60)
    
    optimizer = BatchSizeOptimizer()
    
    # Run optimization
    results = optimizer.find_optimal_batch_size()
    
    if "error" in results:
        logger.error(f"❌ Optimization failed: {results['error']}")
        return
    
    logger.info("✅ Batch size optimization completed!")
    
    # Get the recommended batch size
    recommended_batch = results['analysis']['recommendations']['best_overall']['batch_size']
    
    print(f"\n🎯 QUICK RECOMMENDATION:")
    print(f"   Update BATCH_SIZE in benchmark_optimized_iris.py to: {recommended_batch}")
    print(f"   This should give you the best balance of speed and memory efficiency.")


if __name__ == "__main__":
    main()
