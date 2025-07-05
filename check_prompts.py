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
from typing import Dict, List, Tuple


# ===============================================================================
# CONFIGURATION PARAMETERS - EDIT THESE TO CUSTOMIZE YOUR TESTING
# ===============================================================================

# Test parameters
NUM_SAMPLES = 6000           # Number of image pairs to test
BATCH_SIZE = 8              # Number of concurrent requests
MAX_RETRIES = 3             # Maximum retries for failed requests
TIMEOUT = 30                # Request timeout in seconds

# Output file
OUTPUT_FILE = "prompt_comparison_results.json"

# Select which prompts to test (comment out prompts you don't want to test)
PROMPTS_TO_TEST = {
    # "CONCISE": SAME_PERSON_CONCISE_PROMPT,
    # "ENHANCED": SAME_PERSON_ENHANCED_PROMPT,
    # "SYSTEMATIC": SAME_PERSON_SYSTEMATIC_PROMPT,
    # "CONFIDENCE": SAME_PERSON_CONFIDENCE_PROMPT,
    # "CHAIN_OF_THOUGHT": SAME_PERSON_COT_PROMPT,
    # "EMOTIONLESS_AI": PERSONA_EMOTIONLESS_AI_PROMPT,
    "FORENSIC_EXPERT": PERSONA_FORENSIC_EXPERT_PROMPT,
    "SECURITY_SPECIALIST": PERSONA_SECURITY_SPECIALIST_PROMPT,
    "BIOMETRIC_SCIENTIST": PERSONA_BIOMETRIC_SCIENTIST_PROMPT,
    "MEDICAL_EXAMINER": PERSONA_MEDICAL_EXAMINER_PROMPT,
    "ANTI_FALSE_POSITIVE": ANTI_FALSE_POSITIVE_PROMPT,
    # "ULTRA_CONSERVATIVE": ULTRA_CONSERVATIVE_PROMPT,
    "PRECISION_MATCHING": PRECISION_MATCHING_PROMPT,
    "DISCRIMINATIVE_ANALYSIS": DISCRIMINATIVE_ANALYSIS_PROMPT,
    # "IMPROVED_FROM_ANALYSIS": IMPROVED_PROMPT_FROM_ANALYSIS,  # Uncomment to test
}

# ===============================================================================


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_dataset(num_samples: int) -> List[Tuple]:
    """Load dataset pairs."""
    loader = LoadDataset()
    pairs = loader.pairs[:num_samples]
    logger.info(f"Loaded {len(pairs)} pairs from the dataset.")
    return pairs


async def test_single_prompt(client: Client, pairs: List[Tuple], prompt_name: str, prompt_text: str) -> Dict:
    """Test a single prompt configuration."""
    logger.info(f"Testing prompt: {prompt_name}")

    # Extract just the image paths for batch processing
    pairs_for_batch = [(str(img1_path), str(img2_path)) for img1_path, img2_path, _ in pairs]

    # Process pairs in batches
    start_time = time.time()
    results = await client.is_same_person_batch(pairs_for_batch, batch_size=BATCH_SIZE, prompt=prompt_text)
    end_time = time.time()

    # Calculate metrics
    metrics = calculate_metrics(pairs, results)
    metrics.update({
        "total_predictions": len(pairs),
        "failed_requests": sum(1 for r in results if r.startswith("Error:")),
        "error_rate": 0.0,
        "processing_time": end_time - start_time,
        "avg_time_per_request": (end_time - start_time) / len(pairs),
        "prompt_text": prompt_text
    })

    return metrics


def calculate_metrics(pairs: List[Tuple], results: List[str]) -> Dict:
    """Calculate accuracy metrics for a set of results."""
    correct_predictions = 0
    total_predictions = len(pairs)
    false_positives = 0
    false_negatives = 0
    no_predictions = 0

    for (img1_path, img2_path, label), result in zip(pairs, results):
        # Skip failed requests
        if result.startswith("Error:"):
            no_predictions += 1
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
            no_predictions += 1
            continue

        # Calculate accuracy
        if model_prediction == label:
            correct_predictions += 1
        elif model_prediction == 1 and label == 0:
            false_positives += 1
        elif model_prediction == 0 and label == 1:
            false_negatives += 1

    valid_predictions = total_predictions - no_predictions
    accuracy = (correct_predictions / valid_predictions * 100) if valid_predictions > 0 else 0

    return {
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "no_predictions": no_predictions,
        "valid_predictions": valid_predictions
    }


async def test_all_prompts(pairs: List[Tuple]) -> Dict:
    """Test all configured prompts."""
    results = {}

    async with Client(max_retries=MAX_RETRIES, timeout=TIMEOUT) as client:
        for prompt_name, prompt_text in PROMPTS_TO_TEST.items():
            try:
                result = await test_single_prompt(client, pairs, prompt_name, prompt_text)
                results[prompt_name] = result

                # Log progress
                logger.info(f"Completed {prompt_name}: {result['accuracy']:.2f}% accuracy")

            except Exception as e:
                logger.error(f"Failed to test prompt {prompt_name}: {e}")
                results[prompt_name] = {"error": str(e)}

    return results


def save_results(results: Dict, output_file: str):
    """Save results to JSON file."""
    output_path = Path(output_file)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_path}")


def print_summary(results: Dict):
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

    print(f"{'Prompt Name':<25} {'Accuracy':<10} {'FP':<6} {'FN':<6} {'Time':<8}")
    print("-" * 80)

    for prompt_name, metrics in sorted_results:
        print(f"{prompt_name:<25} {metrics['accuracy']:<10.1f}% {metrics['false_positives']:<6} {metrics['false_negatives']:<6} {metrics['processing_time']:<8.1f}s")

    if sorted_results:
        best_prompt, best_metrics = sorted_results[0]
        print(f"\nBest performing prompt: {best_prompt} ({best_metrics['accuracy']:.2f}% accuracy)")


async def main():
    """Main function to run prompt testing."""
    print("="*80)
    print("FACE VERIFICATION PROMPT TESTING")
    print("="*80)
    print(f"Configuration:")
    print(f"- Samples: {NUM_SAMPLES}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Max retries: {MAX_RETRIES}")
    print(f"- Timeout: {TIMEOUT}s")
    print(f"- Prompts to test: {len(PROMPTS_TO_TEST)}")
    print(f"- Output file: {OUTPUT_FILE}")
    print("="*80)

    # Load dataset
    pairs = await load_dataset(NUM_SAMPLES)

    # Test all prompts
    results = await test_all_prompts(pairs)

    # Save and display results
    save_results(results, OUTPUT_FILE)
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
