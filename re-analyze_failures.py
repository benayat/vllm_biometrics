import json
from pathlib import Path
from tqdm import tqdm
import asyncio
from vlm_client.client import Client

async def process_failed_pairs():
    # Load the JSON file
    input_file = Path("face_validation_results.json")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract failed pairs
    failed_pairs = [
        (item["image1_path"], item["image2_path"])
        for item in data if not item["is_correct"]
    ]
    print(f"Found {len(failed_pairs)} failed pairs.")

    # Process failed pairs
    async with Client() as client:
        print("Processing failed pairs...")
        results = await client.is_same_person_batch(
            failed_pairs, batch_size=10, prompt="is it the same person?output in YES or NO only"
        )

        # Prepare output data
        output_data = []
        for i, ((img1_path, img2_path), result) in enumerate(tqdm(zip(failed_pairs, results), desc="Processing failed pairs", total=len(failed_pairs))):
            pair_data = {
                "pair_id": i + 1,
                "image1_path": img1_path,
                "image2_path": img2_path,
                "model_response": result
            }
            output_data.append(pair_data)

        # Save results to JSON
        output_file = Path("failed_pairs_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nFailed pairs results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(process_failed_pairs())