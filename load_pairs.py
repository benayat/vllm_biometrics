from dataloader.load_dataset import LoadDataset
from vlm_client.client import Client
import asyncio
import json
from pathlib import Path
from tqdm import tqdm

async def main():
    loader = LoadDataset()
    pairs = loader.pairs
    print(f"Loaded {len(pairs)} pairs from the dataset.")

    client = Client()

    # Extract just the image paths for batch processing
    pairs_for_batch = [(str(img1_path), str(img2_path)) for img1_path, img2_path, _ in pairs]

    # Process all pairs in batch with specified batch size
    print("Processing all pairs in batch...")
    results = await client.is_same_person_batch(pairs_for_batch, batch_size=10)

    # Prepare data for JSON output with progress bar
    output_data = []
    correct_predictions = 0
    total_predictions = len(pairs)

    for i, ((img1_path, img2_path, label), result) in enumerate(tqdm(zip(pairs, results), desc="Processing results", total=len(pairs))):
        pair_type = "positive" if label == 1 else "negative"

        # Simple heuristic to determine if model predicted correctly
        # Look for "YES" or "NO" in the response
        model_prediction = None
        if "YES" in result.upper():
            model_prediction = 1
        elif "NO" in result.upper():
            model_prediction = 0

        is_correct = model_prediction == label if model_prediction is not None else False
        if is_correct:
            correct_predictions += 1

        pair_data = {
            "pair_id": i + 1,
            "image1_path": str(img1_path),
            "image2_path": str(img2_path),
            "gold_standard": label,
            "pair_type": pair_type,
            "model_response": result,
            "model_prediction": model_prediction,
            "is_correct": is_correct
        }
        output_data.append(pair_data)

    # Save results to JSON
    output_file = Path("face_validation_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    wrong_predictions = total_predictions - correct_predictions
    accuracy = (correct_predictions / total_predictions) * 100

    print(f"\nResults saved to {output_file}")
    print(f"Wrong predictions: {wrong_predictions} out of {total_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())
