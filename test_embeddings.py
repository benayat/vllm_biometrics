#!/usr/bin/env python3
"""
Test script for iris embedding extraction using the current vLLM server.
This script tests the EmbeddingClient with sample iris images from the IITD database.
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from vlm_client.embedding_client import EmbeddingClient
from dataloader.load_iitd_dataset import LoadIITDDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_server_connection():
    """Test if the vLLM server is running and accessible."""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    model_name = data["data"][0]["id"]
                    print(f"✅ Server is running with model: {model_name}")
                    return True
                else:
                    print(f"❌ Server returned status {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        print("Make sure to start the vLLM server with: ./vllm_serve_gemma.sh")
        return False


async def test_single_embedding():
    """Test extracting embedding from a single iris image."""
    print("\n🔬 Testing single iris embedding extraction...")
    
    # Load dataset to get sample images
    dataset = LoadIITDDataset()
    
    # Get first image from first subject
    first_subject = list(dataset.subject_images.keys())[0]
    sample_image = dataset.subject_images[first_subject][0]
    
    print(f"Sample image: {sample_image}")
    
    async with EmbeddingClient() as client:
        try:
            # Test chat-based embedding extraction
            embedding = await client.get_iris_embedding(str(sample_image), method="chat_based")
            
            print(f"✅ Embedding extracted successfully!")
            print(f"   Embedding shape: {embedding.shape}")
            print(f"   Embedding type: {type(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
            
            return embedding
        except Exception as e:
            print(f"❌ Embedding extraction failed: {e}")
            return None


async def test_iris_verification():
    """Test iris verification between genuine and impostor pairs."""
    print("\n🔍 Testing iris verification...")
    
    # Load dataset and get test pairs
    dataset = LoadIITDDataset()
    
    # Get one genuine pair (same person)
    genuine_pair = dataset.genuine_pairs[0]
    img1_gen, img2_gen, label_gen = genuine_pair
    
    # Get one impostor pair (different people)
    impostor_pair = dataset.impostor_pairs[0]
    img1_imp, img2_imp, label_imp = impostor_pair
    
    async with EmbeddingClient() as client:
        print(f"\n--- Testing Genuine Pair ---")
        print(f"Images: {img1_gen.name} vs {img2_gen.name}")
        print(f"Subjects: {img1_gen.parent.name} vs {img2_gen.parent.name}")
        
        try:
            result_gen = await client.verify_iris_pair(
                str(img1_gen), str(img2_gen), 
                threshold=0.6, method="chat_based"
            )
            
            expected_gen = "Same person"
            predicted_gen = "Same person" if result_gen["is_same_person"] else "Different people"
            correct_gen = "✅" if predicted_gen == expected_gen else "❌"
            
            print(f"Expected: {expected_gen}")
            print(f"Predicted: {predicted_gen} {correct_gen}")
            print(f"Similarity: {result_gen['similarity']:.3f}")
            print(f"Confidence: {result_gen['confidence']}")
            
        except Exception as e:
            print(f"❌ Genuine pair verification failed: {e}")
        
        print(f"\n--- Testing Impostor Pair ---")
        print(f"Images: {img1_imp.name} vs {img2_imp.name}")
        print(f"Subjects: {img1_imp.parent.name} vs {img2_imp.parent.name}")
        
        try:
            result_imp = await client.verify_iris_pair(
                str(img1_imp), str(img2_imp), 
                threshold=0.6, method="chat_based"
            )
            
            expected_imp = "Different people"
            predicted_imp = "Same person" if result_imp["is_same_person"] else "Different people"
            correct_imp = "✅" if predicted_imp == expected_imp else "❌"
            
            print(f"Expected: {expected_imp}")
            print(f"Predicted: {predicted_imp} {correct_imp}")
            print(f"Similarity: {result_imp['similarity']:.3f}")
            print(f"Confidence: {result_imp['confidence']}")
            
        except Exception as e:
            print(f"❌ Impostor pair verification failed: {e}")


async def test_batch_processing():
    """Test batch processing of multiple iris images."""
    print("\n📦 Testing batch embedding extraction...")
    
    # Load dataset and get sample images
    dataset = LoadIITDDataset()
    
    # Get 5 sample images from different subjects
    sample_images = []
    for i, (subject_id, images) in enumerate(list(dataset.subject_images.items())[:5]):
        sample_images.append(str(images[0]))  # First image from each subject
    
    print(f"Processing {len(sample_images)} images in batch...")
    
    async with EmbeddingClient() as client:
        try:
            embeddings = await client.get_iris_embeddings_batch(
                sample_images, batch_size=3, method="chat_based"
            )
            
            successful = sum(1 for emb in embeddings if emb is not None)
            print(f"✅ Batch processing completed!")
            print(f"   Successful extractions: {successful}/{len(sample_images)}")
            
            # Test similarity between first two embeddings
            if embeddings[0] is not None and embeddings[1] is not None:
                similarity = await client.compute_similarity(embeddings[0], embeddings[1])
                print(f"   Similarity between first two: {similarity:.3f}")
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")


async def main():
    """Main test function."""
    print("🧪 IRIS EMBEDDING CLIENT TEST SUITE")
    print("=" * 50)
    
    # Test 1: Server connection
    if not await test_server_connection():
        print("\n❌ Cannot proceed without server connection.")
        print("Please start the vLLM server first: ./vllm_serve_gemma.sh")
        return
    
    # Test 2: Single embedding extraction
    embedding = await test_single_embedding()
    if embedding is None:
        print("\n❌ Cannot proceed without basic embedding extraction.")
        return
    
    # Test 3: Iris verification
    await test_iris_verification()
    
    # Test 4: Batch processing
    await test_batch_processing()
    
    print("\n🎯 Test suite completed!")
    print("\nNext steps:")
    print("1. Run full embedding-based benchmarks")
    print("2. Compare with prompt-based results")
    print("3. Tune similarity thresholds for optimal accuracy")


if __name__ == "__main__":
    asyncio.run(main())
