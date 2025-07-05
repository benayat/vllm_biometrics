from pathlib import Path
import os
import itertools
import random
from typing import List, Tuple


class LoadIITDDataset:
    """
    Class to load the IITD iris database and generate all verification pairs.
    Optimized for high-RAM systems - loads everything into memory for speed.
    """

    def __init__(self, database_path: str = None, load_all_pairs: bool = True):
        """
        Initialize the LoadIITDDataset class.
        
        Args:
            database_path (str): Path to the IITD database folder
            load_all_pairs (bool): If True, generates all possible pairs (uses more RAM but faster)
        """
        self.database_path = database_path or os.path.join(os.getcwd(), "data", "IITD_database")
        self.load_all_pairs = load_all_pairs

        print("Loading IITD database...")

        # Load all subject images
        self.subject_images = self._load_all_subject_images()

        # Generate all pairs
        print("Generating genuine pairs...")
        self.genuine_pairs = self._generate_all_genuine_pairs()

        print("Generating impostor pairs...")
        self.impostor_pairs = self._generate_all_impostor_pairs()

        # Combine all pairs
        self.pairs = self.genuine_pairs + self.impostor_pairs
        
        # Shuffle for random order
        random.shuffle(self.pairs)
        
        self.dataset_size = len(self.pairs)
        
        # Print statistics
        total_images = sum(len(images) for images in self.subject_images.values())
        print(f"\n📊 IITD Database Statistics:")
        print(f"   • Subjects: {len(self.subject_images)}")
        print(f"   • Total images: {total_images}")
        print(f"   • Genuine pairs: {len(self.genuine_pairs):,}")
        print(f"   • Impostor pairs: {len(self.impostor_pairs):,}")
        print(f"   • Total pairs: {self.dataset_size:,}")
        print(f"   • Memory usage: ~{self._estimate_memory_usage():.1f}MB")

    def _load_all_subject_images(self) -> dict:
        """Load all images from all subjects into memory."""
        database_path = Path(self.database_path)
        if not database_path.exists():
            raise FileNotFoundError(f"Database path {database_path} not found.")
        
        subject_images = {}
        
        # Get all numbered directories
        for folder in sorted(database_path.iterdir()):
            if folder.is_dir() and folder.name.isdigit():
                subject_id = folder.name
                images = []

                # Load all BMP files from this subject
                for image_file in sorted(folder.glob("*.bmp")):
                    images.append(image_file)

                if images:
                    subject_images[subject_id] = images

        return subject_images

    def _generate_all_genuine_pairs(self) -> List[Tuple[Path, Path, int]]:
        """Generate ALL genuine pairs (same subject, different images)."""
        genuine_pairs = []
        
        for subject_id, images in self.subject_images.items():
            # Generate all combinations of 2 images for this subject
            for img1, img2 in itertools.combinations(images, 2):
                genuine_pairs.append((img1, img2, 1))  # Label 1 for genuine
        
        return genuine_pairs

    def _generate_all_impostor_pairs(self) -> List[Tuple[Path, Path, int]]:
        """Generate ALL impostor pairs (different subjects)."""
        impostor_pairs = []
        subject_ids = list(self.subject_images.keys())
        
        # Generate pairs between all different subjects
        for i, subject1 in enumerate(subject_ids):
            for subject2 in subject_ids[i+1:]:
                images1 = self.subject_images[subject1]
                images2 = self.subject_images[subject2]
                
                # All combinations between these two subjects
                for img1 in images1:
                    for img2 in images2:
                        impostor_pairs.append((img1, img2, 0))  # Label 0 for impostor

        return impostor_pairs

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        total_images = sum(len(images) for images in self.subject_images.values())
        image_memory = total_images * 0.235  # 235KB per image
        pair_memory = len(self.pairs) * 0.000064  # ~64 bytes per pair
        return image_memory + pair_memory

    def get_benchmark_subsets(self) -> dict:
        """
        Get different sized subsets for benchmarking.

        Returns:
            Dictionary with different test configurations
        """
        return {
            "small": {
                "genuine": 100,
                "impostor": 100,
                "total": 200
            },
            "medium": {
                "genuine": 500,
                "impostor": 500,
                "total": 1000
            },
            "large": {
                "genuine": 1000,
                "impostor": 1000,
                "total": 2000
            },
            "xlarge": {
                "genuine": 2000,
                "impostor": 2000,
                "total": 4000
            },
            "full_genuine": {
                "genuine": len(self.genuine_pairs),
                "impostor": len(self.genuine_pairs),  # Match genuine count
                "total": len(self.genuine_pairs) * 2
            },
            "full_dataset": {
                "genuine": len(self.genuine_pairs),
                "impostor": len(self.impostor_pairs),
                "total": len(self.pairs)
            }
        }

    def get_balanced_sample(self, num_genuine: int, num_impostor: int) -> List[Tuple[Path, Path, int]]:
        """
        Get a balanced sample of pairs for testing.

        Args:
            num_genuine (int): Number of genuine pairs
            num_impostor (int): Number of impostor pairs

        Returns:
            List of sampled pairs
        """
        genuine_sample = random.sample(self.genuine_pairs, min(num_genuine, len(self.genuine_pairs)))
        impostor_sample = random.sample(self.impostor_pairs, min(num_impostor, len(self.impostor_pairs)))
        
        sample_pairs = genuine_sample + impostor_sample
        random.shuffle(sample_pairs)

        return sample_pairs

    def get_statistics(self) -> dict:
        """Get detailed dataset statistics."""
        total_images = sum(len(images) for images in self.subject_images.values())
        images_per_subject = [len(images) for images in self.subject_images.values()]
        
        return {
            "subjects": len(self.subject_images),
            "total_images": total_images,
            "images_per_subject": {
                "min": min(images_per_subject),
                "max": max(images_per_subject),
                "avg": sum(images_per_subject) / len(images_per_subject)
            },
            "pairs": {
                "genuine": len(self.genuine_pairs),
                "impostor": len(self.impostor_pairs),
                "total": len(self.pairs)
            },
            "memory_usage_mb": self._estimate_memory_usage()
        }


# Quick benchmark configuration generator
def create_benchmark_config(dataset: LoadIITDDataset, size: str = "medium") -> dict:
    """
    Create a benchmark configuration for testing.

    Args:
        dataset: LoadIITDDataset instance
        size: "small", "medium", "large", "xlarge", "full_genuine", or "full_dataset"

    Returns:
        Dictionary with benchmark configuration
    """
    benchmarks = dataset.get_benchmark_subsets()

    if size not in benchmarks:
        raise ValueError(f"Size must be one of: {list(benchmarks.keys())}")

    config = benchmarks[size]
    pairs = dataset.get_balanced_sample(config["genuine"], config["impostor"])

    return {
        "pairs": pairs,
        "config": config,
        "description": f"Benchmark with {config['total']} pairs ({config['genuine']} genuine, {config['impostor']} impostor)"
    }


async def demo_iris_comparison(dataset: LoadIITDDataset, num_samples: int = 5):
    """
    Demo function to test iris comparison using the VLM client.

    Args:
        dataset: LoadIITDDataset instance
        num_samples: Number of sample comparisons to run
    """
    from vlm_client.client import Client
    from constants.prompts import IRIS_COMPARISON_PROMPT

    print(f"\n🔍 IRIS COMPARISON DEMO")
    print("="*50)

    # Get a small balanced sample for demo
    sample_pairs = dataset.get_balanced_sample(num_samples//2, num_samples//2)

    async with Client() as client:
        print(f"Testing {len(sample_pairs)} iris pairs...")

        for i, (img1, img2, label) in enumerate(sample_pairs):
            pair_type = "Genuine" if label == 1 else "Impostor"
            subject1 = img1.parent.name
            subject2 = img2.parent.name

            print(f"\n--- Sample {i+1}/{len(sample_pairs)} ---")
            print(f"Type: {pair_type}")
            print(f"Images: {img1.name} (Subject {subject1}) vs {img2.name} (Subject {subject2})")

            try:
                # Make the iris comparison
                result = await client.is_same_person(
                    str(img1),
                    str(img2),
                    prompt=IRIS_COMPARISON_PROMPT
                )

                # Extract prediction
                prediction = "YES" if "YES" in result.upper() else "NO"
                expected = "YES" if label == 1 else "NO"
                correct = "✓" if prediction == expected else "✗"

                print(f"Expected: {expected}")
                print(f"Predicted: {prediction} {correct}")
                print(f"Model Response: {result}")

            except Exception as e:
                print(f"Error: {e}")

        print(f"\n🎯 Demo completed!")


# Example usage
if __name__ == "__main__":
    # Load the full dataset (uses ~1GB RAM)
    dataset = LoadIITDDataset()

    # Print detailed statistics
    stats = dataset.get_statistics()
    print(f"\n📈 Detailed Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for subkey, subvalue in value.items():
                print(f"     • {subkey}: {subvalue}")
        else:
            print(f"   • {key}: {value}")

    # Create benchmark configurations
    print(f"\n🏆 Available Benchmark Configurations:")
    for size in ["small", "medium", "large", "xlarge", "full_genuine"]:
        config = create_benchmark_config(dataset, size)
        print(f"   • {size}: {config['description']}")

    # Run iris comparison demo
    print(f"\n" + "="*60)
    print("RUNNING IRIS COMPARISON DEMO")
    print("="*60)

    import asyncio
    asyncio.run(demo_iris_comparison(dataset, num_samples=6))
