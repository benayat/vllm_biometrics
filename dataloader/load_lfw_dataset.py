#!/usr/bin/env python3
"""
LFW (Labeled Faces in the Wild) Dataset Loader

This module loads and manages the LFW dataset for face recognition benchmarking.
It handles pair generation for genuine and impostor comparisons.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from collections import defaultdict

class LoadLFWDataset:
    """
    Load and manage the LFW (Labeled Faces in the Wild) dataset.
    
    The LFW dataset contains face images of famous people collected from the web.
    Each person has multiple images, making it suitable for face verification tasks.
    """
    
    def __init__(self, data_dir: str = None, pairs_file: str = None):
        """
        Initialize LFW dataset loader.
        
        Args:
            data_dir: Path to LFW dataset directory (auto-detected if None)
            pairs_file: Path to pairs.txt file (auto-detected if None)
        """
        self.data_dir = self._find_data_directory(data_dir)
        self.pairs_file = self._find_pairs_file(pairs_file)
        
        print("Loading LFW database...")
        
        # Load all subject images
        self.subject_images = self._load_subject_images()
        
        # Load official pairs
        self.official_genuine_pairs, self.official_impostor_pairs = self._load_official_pairs()
        
        # Generate additional pairs if needed
        print("Generating additional pairs...")
        self.additional_genuine_pairs = self._generate_additional_genuine_pairs()
        self.additional_impostor_pairs = self._generate_additional_impostor_pairs()
        
        # Combine all pairs
        self.genuine_pairs = self.official_genuine_pairs + self.additional_genuine_pairs
        self.impostor_pairs = self.official_impostor_pairs + self.additional_impostor_pairs

        # Shuffle all pairs to ensure randomness
        random.shuffle(self.genuine_pairs)
        random.shuffle(self.impostor_pairs)

        self.pairs = self.genuine_pairs + self.impostor_pairs
        random.shuffle(self.pairs)  # Shuffle combined pairs too

        # Print statistics
        self._print_statistics()
    
    def _find_data_directory(self, data_dir: str = None) -> Path:
        """Find the LFW data directory."""
        if data_dir:
            return Path(data_dir)
        
        # Common LFW directory names to search for
        possible_dirs = [
            "data/lfw_funneled",
            "data/lfw-funneled", 
            "data/lfw",
            "lfwpeople/extracted/lfw_funneled",
            "lfwpeople/extracted/lfw-funneled",
            "extracted/lfw_funneled",
            "extracted/lfw-funneled",
            "lfw_funneled",
            "lfw-funneled",
            "lfw"
        ]
        
        base_path = Path(__file__).parent.parent
        
        for dir_name in possible_dirs:
            potential_path = base_path / dir_name
            if potential_path.exists() and potential_path.is_dir():
                # Check if it contains person directories
                subdirs = [d for d in potential_path.iterdir() if d.is_dir()]
                if len(subdirs) > 100:  # LFW has 5749+ people
                    print(f"Found LFW data directory: {potential_path}")
                    return potential_path
        
        raise FileNotFoundError(
            "LFW dataset directory not found. Please ensure the LFW dataset is extracted "
            "and available in one of the expected locations."
        )
    
    def _find_pairs_file(self, pairs_file: str = None) -> Path:
        """Find the pairs.txt file."""
        if pairs_file:
            return Path(pairs_file)
        
        # Search for pairs.txt in common locations
        possible_files = [
            "lfwpeople/pairs.txt",
            "data/pairs.txt",
            "pairs.txt"
        ]
        
        base_path = Path(__file__).parent.parent
        
        for file_name in possible_files:
            potential_path = base_path / file_name
            if potential_path.exists():
                print(f"Found pairs file: {potential_path}")
                return potential_path
        
        print("⚠️  No pairs.txt file found. Will use generated pairs only.")
        return None
    
    def _load_subject_images(self) -> Dict[str, List[Path]]:
        """Load all subject images from the dataset."""
        subject_images = defaultdict(list)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"LFW data directory not found: {self.data_dir}")
        
        # Each subdirectory represents a person
        for person_dir in self.data_dir.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            
            # Load all images for this person
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(person_dir.glob(ext))
            
            if image_files:
                subject_images[person_name] = sorted(image_files)
        
        print(f"Loaded {len(subject_images)} people with images")
        return dict(subject_images)
    
    def _load_official_pairs(self) -> Tuple[List[Tuple], List[Tuple]]:
        """Load official pairs from pairs.txt file."""
        if not self.pairs_file or not self.pairs_file.exists():
            return [], []
        
        genuine_pairs = []
        impostor_pairs = []
        
        try:
            with open(self.pairs_file, 'r') as f:
                lines = f.readlines()
            
            # Skip header line if present
            if lines[0].strip().isdigit():
                lines = lines[1:]
            
            for line in lines:
                parts = line.strip().split('\t')
                
                if len(parts) == 3:
                    # Genuine pair: person_name image1_num image2_num
                    person_name, img1_num, img2_num = parts
                    img1_num, img2_num = int(img1_num), int(img2_num)
                    
                    # Find the actual image files
                    if person_name in self.subject_images:
                        person_images = self.subject_images[person_name]
                        
                        # LFW naming convention: PersonName_####.jpg
                        img1_name = f"{person_name}_{img1_num:04d}.jpg"
                        img2_name = f"{person_name}_{img2_num:04d}.jpg"
                        
                        img1_path = self.data_dir / person_name / img1_name
                        img2_path = self.data_dir / person_name / img2_name
                        
                        if img1_path.exists() and img2_path.exists():
                            genuine_pairs.append((img1_path, img2_path, 1))
                
                elif len(parts) == 4:
                    # Impostor pair: person1_name image1_num person2_name image2_num
                    person1_name, img1_num, person2_name, img2_num = parts
                    img1_num, img2_num = int(img1_num), int(img2_num)
                    
                    # Find the actual image files
                    if person1_name in self.subject_images and person2_name in self.subject_images:
                        img1_name = f"{person1_name}_{img1_num:04d}.jpg"
                        img2_name = f"{person2_name}_{img2_num:04d}.jpg"
                        
                        img1_path = self.data_dir / person1_name / img1_name
                        img2_path = self.data_dir / person2_name / img2_name
                        
                        if img1_path.exists() and img2_path.exists():
                            impostor_pairs.append((img1_path, img2_path, 0))
        
        except Exception as e:
            print(f"Error loading official pairs: {e}")
            return [], []
        
        print(f"Loaded {len(genuine_pairs)} official genuine pairs")
        print(f"Loaded {len(impostor_pairs)} official impostor pairs")
        
        return genuine_pairs, impostor_pairs
    
    def _generate_additional_genuine_pairs(self, max_pairs: int = 3000) -> List[Tuple]:
        """Generate additional genuine pairs from people with multiple images."""
        genuine_pairs = []
        
        # Filter people with multiple images
        multi_image_people = {
            name: images for name, images in self.subject_images.items()
            if len(images) >= 2
        }
        
        print(f"Found {len(multi_image_people)} people with multiple images")
        
        # Generate pairs for each person
        pairs_per_person = max(1, max_pairs // len(multi_image_people))
        
        for person_name, images in multi_image_people.items():
            person_pairs = 0
            
            # Generate all possible pairs for this person
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    if person_pairs >= pairs_per_person:
                        break
                    
                    genuine_pairs.append((images[i], images[j], 1))
                    person_pairs += 1
                
                if person_pairs >= pairs_per_person:
                    break
        
        # Shuffle to randomize
        random.shuffle(genuine_pairs)
        
        # Limit to max_pairs
        genuine_pairs = genuine_pairs[:max_pairs]
        
        print(f"Generated {len(genuine_pairs)} additional genuine pairs")
        return genuine_pairs
    
    def _generate_additional_impostor_pairs(self, max_pairs: int = 10000) -> List[Tuple]:
        """Generate additional impostor pairs from different people."""
        impostor_pairs = []
        
        people_list = list(self.subject_images.keys())
        
        # Generate random impostor pairs
        for _ in range(max_pairs):
            # Select two different people
            person1, person2 = random.sample(people_list, 2)
            
            # Select random images from each person
            img1 = random.choice(self.subject_images[person1])
            img2 = random.choice(self.subject_images[person2])
            
            impostor_pairs.append((img1, img2, 0))
        
        print(f"Generated {len(impostor_pairs)} additional impostor pairs")
        return impostor_pairs
    
    def _print_statistics(self):
        """Print dataset statistics."""
        total_images = sum(len(images) for images in self.subject_images.values())
        images_per_person = [len(images) for images in self.subject_images.values()]

        print(f"\n📊 LFW Database Statistics:")
        print(f"   • People: {len(self.subject_images)}")
        print(f"   • Total images: {total_images}")
        print(f"   • Images per person: {min(images_per_person)}-{max(images_per_person)} (avg: {np.mean(images_per_person):.1f})")
        print(f"   • Official genuine pairs: {len(self.official_genuine_pairs)}")
        print(f"   • Official impostor pairs: {len(self.official_impostor_pairs)}")
        print(f"   • Additional genuine pairs: {len(self.additional_genuine_pairs)}")
        print(f"   • Additional impostor pairs: {len(self.additional_impostor_pairs)}")
        print(f"   • Total pairs: {len(self.pairs)}")
        print(f"   • Memory usage: ~{self._estimate_memory_usage():.1f}MB")

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        total_images = sum(len(images) for images in self.subject_images.values())
        # Estimate average image size for face images
        image_memory = total_images * 0.15  # ~150KB per face image
        pair_memory = len(self.pairs) * 0.000064  # ~64 bytes per pair
        return image_memory + pair_memory

    def get_balanced_sample(self, num_genuine: int, num_impostor: int) -> List[Tuple]:
        """
        Get a balanced sample of pairs for testing.

        Args:
            num_genuine: Number of genuine pairs
            num_impostor: Number of impostor pairs

        Returns:
            List of sampled pairs
        """
        # Sample genuine pairs
        genuine_sample = random.sample(
            self.genuine_pairs, 
            min(num_genuine, len(self.genuine_pairs))
        )
        
        # Sample impostor pairs
        impostor_sample = random.sample(
            self.impostor_pairs,
            min(num_impostor, len(self.impostor_pairs))
        )
        
        # Combine and shuffle
        sample_pairs = genuine_sample + impostor_sample
        random.shuffle(sample_pairs)

        return sample_pairs

    def get_official_pairs_only(self) -> List[Tuple]:
        """Get only the official LFW pairs for benchmarking."""
        official_pairs = self.official_genuine_pairs + self.official_impostor_pairs
        random.shuffle(official_pairs)
        return official_pairs

    def get_benchmark_subsets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get different sized subsets for benchmarking.

        Returns:
            Dictionary with different test configurations
        """
        return {
            "official_only": {
                "genuine": len(self.official_genuine_pairs),
                "impostor": len(self.official_impostor_pairs),
                "total": len(self.official_genuine_pairs) + len(self.official_impostor_pairs),
                "description": "Official LFW pairs only"
            },
            "small": {
                "genuine": 500,
                "impostor": 500,
                "total": 1000,
                "description": "Small balanced subset"
            },
            "medium": {
                "genuine": 1500,
                "impostor": 1500,
                "total": 3000,
                "description": "Medium balanced subset"
            },
            "large": {
                "genuine": 3000,
                "impostor": 3000,
                "total": 6000,
                "description": "Large balanced subset"
            },
            "full_dataset": {
                "genuine": len(self.genuine_pairs),
                "impostor": len(self.impostor_pairs),
                "total": len(self.pairs),
                "description": "Complete dataset with all pairs"
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed dataset statistics."""
        total_images = sum(len(images) for images in self.subject_images.values())
        images_per_person = [len(images) for images in self.subject_images.values()]

        return {
            "people": len(self.subject_images),
            "total_images": total_images,
            "images_per_person": {
                "min": min(images_per_person),
                "max": max(images_per_person),
                "mean": np.mean(images_per_person),
                "std": np.std(images_per_person)
            },
            "pairs": {
                "official_genuine": len(self.official_genuine_pairs),
                "official_impostor": len(self.official_impostor_pairs),
                "additional_genuine": len(self.additional_genuine_pairs),
                "additional_impostor": len(self.additional_impostor_pairs),
                "total_genuine": len(self.genuine_pairs),
                "total_impostor": len(self.impostor_pairs),
                "total": len(self.pairs)
            },
            "memory_usage_mb": self._estimate_memory_usage(),
            "data_directory": str(self.data_dir),
            "pairs_file": str(self.pairs_file) if self.pairs_file else None
        }

    def get_people_with_multiple_images(self, min_images: int = 2) -> Dict[str, int]:
        """
        Get people who have multiple images.

        Args:
            min_images: Minimum number of images required

        Returns:
            Dictionary mapping person name to number of images
        """
        return {
            name: len(images)
            for name, images in self.subject_images.items()
            if len(images) >= min_images
        }

    def get_person_pairs(self, person_name: str) -> List[Tuple]:
        """
        Get all pairs for a specific person.

        Args:
            person_name: Name of the person

        Returns:
            List of pairs involving this person
        """
        person_pairs = []

        for pair in self.pairs:
            img1_path, img2_path, label = pair

            # Check if either image belongs to this person
            if (person_name in str(img1_path) or person_name in str(img2_path)):
                person_pairs.append(pair)

        return person_pairs


def create_benchmark_config(dataset: LoadLFWDataset, size: str = "medium") -> Dict[str, Any]:
    """
    Create a benchmark configuration for testing.

    Args:
        dataset: LoadLFWDataset instance
        size: Size of the benchmark ('small', 'medium', 'large', 'official_only', 'full_dataset')

    Returns:
        Dictionary with benchmark configuration
    """
    subsets = dataset.get_benchmark_subsets()

    if size not in subsets:
        raise ValueError(f"Unknown size '{size}'. Available: {list(subsets.keys())}")

    config = subsets[size]

    # Get the actual pairs
    if size == "official_only":
        pairs = dataset.get_official_pairs_only()
    else:
        pairs = dataset.get_balanced_sample(config["genuine"], config["impostor"])

    return {
        "name": f"LFW_{size}",
        "pairs": pairs,
        "statistics": config,
        "dataset_info": dataset.get_statistics()
    }


# Quick usage example
if __name__ == "__main__":
    # Load the dataset
    dataset = LoadLFWDataset()

    # Get some statistics
    stats = dataset.get_statistics()
    print(f"\nDataset loaded with {stats['people']} people and {stats['total_images']} images")

    # Get a sample for testing
    sample_pairs = dataset.get_balanced_sample(100, 100)
    print(f"Sample contains {len(sample_pairs)} pairs")

    # Show benchmark configurations
    print("\nAvailable benchmark configurations:")
    for name, config in dataset.get_benchmark_subsets().items():
        print(f"  {name}: {config['description']} ({config['total']} pairs)")
