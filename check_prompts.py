from dataloader.load_dataset import LoadDataset
from vlm_client.client import Client
from constants.prompts import (
    SAME_PERSON_CONCISE_PROMPT,
    SAME_PERSON_ENHANCED_PROMPT,
    SAME_PERSON_SYSTEMATIC_PROMPT,
    SAME_PERSON_CONFIDENCE_PROMPT,
    SAME_PERSON_COT_PROMPT,
    PERSONA_EMOTIONLESS_AI_PROMPT,
    PERSONA_FORENSIC_EXPERT_PROMPT,
    PERSONA_SECURITY_SPECIALIST_PROMPT,
    PERSONA_BIOMETRIC_SCIENTIST_PROMPT,
    PERSONA_MEDICAL_EXAMINER_PROMPT,
    ANTI_FALSE_POSITIVE_PROMPT,
    ULTRA_CONSERVATIVE_PROMPT,
    PRECISION_MATCHING_PROMPT,
    DISCRIMINATIVE_ANALYSIS_PROMPT,
    IMPROVED_PROMPT_FROM_ANALYSIS
)
import asyncio
import json
from pathlib import Path
from tqdm import tqdm
import time
import logging
from typing import Dict, List, Tuple, Optional
import argparse


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTester:
    """Class to test different prompts for face verification."""

    def __init__(self, batch_size: int = 8, max_retries: int = 3, timeout: int = 30):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.client = Client(max_retries=self.max_retries, timeout=self.timeout)
        await self.client._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.close()

    def load_dataset(self, num_samples: Optional[int] = None) -> List[Tuple]:
        """Load dataset pairs."""
        loader = LoadDataset()
        pairs = loader.pairs

        if num_samples:
            pairs = pairs[:num_samples]

        logger.info(f"Loaded {len(pairs)} pairs from the dataset.")
        return pairs

    def get_prompt_configurations(self) -> Dict[str, str]:
        """Get all available prompt configurations."""
        return {
            "concise": SAME_PERSON_CONCISE_PROMPT,
            "enhanced": SAME_PERSON_ENHANCED_PROMPT,
            "systematic": SAME_PERSON_SYSTEMATIC_PROMPT,
            "confidence": SAME_PERSON_CONFIDENCE_PROMPT,
            "chain_of_thought": SAME_PERSON_COT_PROMPT,
            "emotionless_ai": PERSONA_EMOTIONLESS_AI_PROMPT,
            "forensic_expert": PERSONA_FORENSIC_EXPERT_PROMPT,
            "security_specialist": PERSONA_SECURITY_SPECIALIST_PROMPT,
            "biometric_scientist": PERSONA_BIOMETRIC_SCIENTIST_PROMPT,
            "medical_examiner": PERSONA_MEDICAL_EXAMINER_PROMPT,
            "anti_false_positive": ANTI_FALSE_POSITIVE_PROMPT,
            "ultra_conservative": ULTRA_CONSERVATIVE_PROMPT,
            "precision_matching": PRECISION_MATCHING_PROMPT,
            "discriminative_analysis": DISCRIMINATIVE_ANALYSIS_PROMPT,
            "improved_from_analysis": IMPROVED_PROMPT_FROM_ANALYSIS
        }

    async def test_single_prompt(self, pairs: List[Tuple], prompt_name: str, prompt_text: str) -> Dict:
        """Test a single prompt configuration."""
        logger.info(f"Testing prompt: {prompt_name}")

        # Extract just the image paths for batch processing
        pairs_for_batch = [(str(img1_path), str(img2_path)) for img1_path, img2_path, _ in pairs]

        # Process pairs in batches
        start_time = time.time()
        results = await self.client.is_same_person_batch(pairs_for_batch, batch_size=self.batch_size, prompt=prompt_text)
        end_time = time.time()

        # Calculate metrics
        metrics = self.calculate_metrics(pairs, results)
        metrics.update({
            "prompt_name": prompt_name,
            "processing_time": end_time - start_time,
            "total_pairs": len(pairs),
            "failed_requests": sum(1 for r in results if r.startswith("Error:"))
        })

        return metrics

    def calculate_metrics(self, pairs: List[Tuple], results: List[str]) -> Dict:
        """Calculate accuracy metrics for a set of results."""
        correct_predictions = 0
        total_predictions = len(pairs)
        false_positives = 0
        false_negatives = 0
        no_prediction = 0

        for (img1_path, img2_path, label), result in zip(pairs, results):
            # Skip failed requests
            if result.startswith("Error:"):
                no_prediction += 1
                continue

            # Simple heuristic to determine model prediction
            model_prediction = None
            result_upper = result.upper()

            if "YES" in result_upper and "NO" in result_upper:
                # Contradictory response - count as incorrect
                model_prediction = -1  # Invalid prediction
            elif "YES" in result_upper:
                model_prediction = 1
            elif "NO" in result_upper:
                model_prediction = 0
            else:
                no_prediction += 1
                continue

            # Calculate accuracy
            if model_prediction == label:
                correct_predictions += 1
            elif model_prediction == 1 and label == 0:
                false_positives += 1
            elif model_prediction == 0 and label == 1:
                false_negatives += 1

        valid_predictions = total_predictions - no_prediction
        accuracy = (correct_predictions / valid_predictions * 100) if valid_predictions > 0 else 0

        return {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "no_prediction": no_prediction,
            "valid_predictions": valid_predictions
        }

    async def test_multiple_prompts(self, pairs: List[Tuple], prompt_names: List[str] = None) -> Dict:
        """Test multiple prompt configurations."""
        prompts = self.get_prompt_configurations()

        if prompt_names:
            prompts = {name: prompts[name] for name in prompt_names if name in prompts}

        results = {}

        for prompt_name, prompt_text in prompts.items():
            try:
                result = await self.test_single_prompt(pairs, prompt_name, prompt_text)
                results[prompt_name] = result

                # Log progress
                logger.info(f"Completed {prompt_name}: {result['accuracy']:.2f}% accuracy")

            except Exception as e:
                logger.error(f"Failed to test prompt {prompt_name}: {e}")
                results[prompt_name] = {"error": str(e)}

        return results

    def save_results(self, results: Dict, output_file: str = "prompt_comparison_results.json"):
        """Save results to JSON file."""
        output_path = Path(output_file)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_path}")

    def print_summary(self, results: Dict):
        """Print a summary of the results."""
        print("\n" + "="*80)
        print("PROMPT COMPARISON SUMMARY")
        print("="*80)

        # Sort by accuracy
        sorted_results = sorted(
            [(name, data) for name, data in results.items() if isinstance(data, dict) and "accuracy" in data],
            key=lambda x: x[1]["accuracy"],
            reverse=True
        )

        print(f"{'Prompt Name':<30} {'Accuracy':<12} {'FP':<6} {'FN':<6} {'Time':<8}")
        print("-" * 80)

        for prompt_name, metrics in sorted_results:
            print(f"{prompt_name:<30} {metrics['accuracy']:<12.2f}% {metrics['false_positives']:<6} {metrics['false_negatives']:<6} {metrics['processing_time']:<8.1f}s")

        if sorted_results:
            best_prompt, best_metrics = sorted_results[0]
            print(f"\nBest performing prompt: {best_prompt} ({best_metrics['accuracy']:.2f}% accuracy)")


async def main():
    """Main function to run prompt testing."""
    parser = argparse.ArgumentParser(description="Test different prompts for face verification")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples to test (default: 200)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing (default: 8)")
    parser.add_argument("--prompts", nargs="+", help="Specific prompts to test (default: all)")
    parser.add_argument("--output", default="prompt_comparison_results.json", help="Output file for results")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for failed requests")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")

    args = parser.parse_args()

    # Initialize tester
    async with PromptTester(batch_size=args.batch_size, max_retries=args.max_retries, timeout=args.timeout) as tester:
        # Load dataset
        pairs = tester.load_dataset(num_samples=args.num_samples)

        # Test prompts
        results = await tester.test_multiple_prompts(pairs, prompt_names=args.prompts)

        # Save and display results
        tester.save_results(results, args.output)
        tester.print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
