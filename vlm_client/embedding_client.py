import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from dataloader.util import encode_image_to_base64
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Client for extracting embeddings from iris images using Gemma-3 via vLLM OpenAI-compatible API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", model_name: str = "gemma-3-27b-it",
                 max_retries: int = 3, retry_delay: float = 1.0, timeout: int = 30):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.embeddings_url = f"{self.base_url}/embeddings"
        self.chat_url = f"{self.base_url}/chat/completions"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_created = False

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _ensure_session(self):
        """Ensure session is created if not using context manager"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
            )
            self._session_created = True

    async def close(self):
        """Close the session manually if not using context manager"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            self._session_created = False

    async def get_iris_embedding(self, image_path: str, method: str = "chat_based") -> np.ndarray:
        """
        Get embedding for a single iris image.
        
        Args:
            image_path (str): Path to the iris image
            method (str): "embeddings" for direct embedding API or "chat_based" for chat completion
            
        Returns:
            np.ndarray: Embedding vector for the iris image
        """
        if method == "embeddings":
            return await self._get_embedding_direct(image_path)
        else:
            return await self._get_embedding_via_chat(image_path)

    async def get_iris_embeddings_batch(self, image_paths: List[str], batch_size: int = 8, 
                                      method: str = "chat_based") -> List[np.ndarray]:
        """
        Get embeddings for multiple iris images in batches.
        
        Args:
            image_paths (List[str]): List of image paths
            batch_size (int): Number of concurrent requests
            method (str): Embedding extraction method
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        await self._ensure_session()
        
        # Create all requests up front
        tasks = []
        for image_path in image_paths:
            task = self.get_iris_embedding(image_path, method)
            tasks.append(task)

        # Process tasks in batches
        results = []
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Request failed: {result}")
                    results.append(None)
                else:
                    results.append(result)

        return results

    async def _get_embedding_direct(self, image_path: str) -> np.ndarray:
        """
        Get embedding using the direct embeddings API endpoint.
        
        Args:
            image_path (str): Path to the iris image
            
        Returns:
            np.ndarray: Embedding vector
        """
        await self._ensure_session()
        
        # Encode image to base64
        image_base64 = encode_image_to_base64(image_path)
        
        payload = {
            "model": self.model_name,
            "input": f"data:image/png;base64,{image_base64}",
            "encoding_format": "float"
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.post(self.embeddings_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        embedding = data["data"][0]["embedding"]
                        return np.array(embedding, dtype=np.float32)
                    else:
                        error_text = await response.text()
                        logger.warning(f"Embedding request failed with status {response.status}: {error_text}")

                        if attempt < self.max_retries:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        else:
                            raise aiohttp.ClientError(f"Embedding request failed after {self.max_retries} retries")

            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"Embedding request error (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise

    async def _get_embedding_via_chat(self, image_path: str) -> np.ndarray:
        """
        Get embedding by asking the model to describe iris features via chat completion.
        This extracts the hidden state as a proxy for embeddings.
        
        Args:
            image_path (str): Path to the iris image
            
        Returns:
            np.ndarray: Feature vector based on model response
        """
        await self._ensure_session()
        
        # Use the specialized iris comparison prompt from your prompts.py
        from constants.prompts import IRIS_COMPARISON_PROMPT

        # Enhanced iris analysis prompt for better feature extraction
        iris_analysis_prompt = """
        Analyze this iris image in extreme detail for biometric identification:
        
        STRUCTURAL ANALYSIS:
        - Radial furrows: direction, depth, frequency
        - Crypts: size, location, distribution pattern
        - Lacunae: shape, position, density
        - Corona structure and boundaries
        - Pupillary border characteristics
        
        TEXTURE ANALYSIS:
        - Fine texture patterns and grain
        - Collagen fiber orientation
        - Surface irregularities and bumps
        - Pigmentation distribution and intensity
        
        DISTINCTIVE FEATURES:
        - Unique identifying marks or patterns
        - Asymmetries and irregularities
        - Color variations and transitions
        - Specific geometric relationships
        
        Provide a comprehensive technical description focusing on quantifiable biometric features that would distinguish this iris from others.
        """
        
        payload = self._build_chat_payload(iris_analysis_prompt, image_path)

        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.post(self.chat_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_text = data["choices"][0]["message"]["content"]
                        
                        # Convert text response to feature vector
                        embedding = self._text_to_embedding(response_text)
                        return embedding
                    else:
                        error_text = await response.text()
                        logger.warning(f"Chat request failed with status {response.status}: {error_text}")

                        if attempt < self.max_retries:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        else:
                            raise aiohttp.ClientError(f"Chat request failed after {self.max_retries} retries")

            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"Chat request error (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise

    def _build_chat_payload(self, prompt: str, image_path: str) -> Dict[str, Any]:
        """Build the JSON payload for chat completion."""
        image_base64 = encode_image_to_base64(image_path)

        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent feature extraction
            "max_tokens": 512    # Longer response for detailed features
        }

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text description to a numerical embedding vector.
        This is a simple approach - could be enhanced with proper text embeddings.
        
        Args:
            text (str): Text description of iris features
            
        Returns:
            np.ndarray: Feature vector
        """
        # Simple text-to-vector conversion
        # This could be improved with proper text embedding models
        
        # Extract key iris features and create a feature vector
        features = {
            'radial': text.lower().count('radial') + text.lower().count('spoke'),
            'furrow': text.lower().count('furrow') + text.lower().count('groove'),
            'texture': text.lower().count('texture') + text.lower().count('pattern'),
            'color': text.lower().count('color') + text.lower().count('pigment'),
            'crypt': text.lower().count('crypt') + text.lower().count('pit'),
            'lacuna': text.lower().count('lacuna') + text.lower().count('hole'),
            'corona': text.lower().count('corona') + text.lower().count('ring'),
            'border': text.lower().count('border') + text.lower().count('edge'),
            'dark': text.lower().count('dark') + text.lower().count('black'),
            'light': text.lower().count('light') + text.lower().count('bright'),
            'dense': text.lower().count('dense') + text.lower().count('thick'),
            'fine': text.lower().count('fine') + text.lower().count('thin'),
            'circular': text.lower().count('circular') + text.lower().count('round'),
            'linear': text.lower().count('linear') + text.lower().count('straight'),
            'complex': text.lower().count('complex') + text.lower().count('intricate'),
            'unique': text.lower().count('unique') + text.lower().count('distinctive')
        }
        
        # Create a longer feature vector with text statistics
        text_stats = [
            len(text),                    # Text length
            len(text.split()),           # Word count
            text.count('.'),             # Sentence count
            text.count(','),             # Comma count
            sum(1 for c in text if c.isupper()),  # Uppercase count
            sum(1 for c in text if c.isdigit()),  # Digit count
        ]
        
        # Combine all features
        feature_vector = list(features.values()) + text_stats
        
        # Normalize to unit vector
        embedding = np.array(feature_vector, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding

    async def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    async def verify_iris_pair(self, image1_path: str, image2_path: str, 
                              threshold: float = 0.8, method: str = "chat_based") -> Dict[str, Any]:
        """
        Verify if two iris images belong to the same person using embeddings.
        
        Args:
            image1_path (str): Path to first iris image
            image2_path (str): Path to second iris image
            threshold (float): Similarity threshold for same person decision
            method (str): Embedding extraction method
            
        Returns:
            Dict with verification result, similarity score, and decision
        """
        # Get embeddings for both images
        embedding1 = await self.get_iris_embedding(image1_path, method)
        embedding2 = await self.get_iris_embedding(image2_path, method)
        
        # Compute similarity
        similarity = await self.compute_similarity(embedding1, embedding2)
        
        # Make decision based on threshold
        is_same_person = similarity >= threshold
        
        return {
            "image1": str(image1_path),
            "image2": str(image2_path),
            "similarity": similarity,
            "threshold": threshold,
            "is_same_person": is_same_person,
            "confidence": "high" if abs(similarity - threshold) > 0.2 else "medium",
            "method": method
        }


# Demo function for testing
async def demo_iris_embeddings():
    """Demo function to test iris embedding extraction."""
    from dataloader.load_iitd_dataset import LoadIITDDataset
    
    print("🔬 IRIS EMBEDDING EXTRACTION DEMO")
    print("=" * 50)
    
    # Load dataset and get sample pairs
    dataset = LoadIITDDataset()
    sample_pairs = dataset.get_balanced_sample(3, 3)  # 3 genuine, 3 impostor
    
    async with EmbeddingClient() as client:
        print(f"Testing embedding extraction on {len(sample_pairs)} pairs...")
        
        for i, (img1, img2, label) in enumerate(sample_pairs):
            pair_type = "Genuine" if label == 1 else "Impostor"
            subject1 = img1.parent.name
            subject2 = img2.parent.name
            
            print(f"\n--- Pair {i+1}/{len(sample_pairs)} ---")
            print(f"Type: {pair_type}")
            print(f"Images: {img1.name} (Subject {subject1}) vs {img2.name} (Subject {subject2})")
            
            try:
                # Test embedding-based verification
                result = await client.verify_iris_pair(str(img1), str(img2), threshold=0.6)
                
                expected = "Same" if label == 1 else "Different"
                predicted = "Same" if result["is_same_person"] else "Different"
                correct = "✓" if (predicted == expected) else "✗"
                
                print(f"Expected: {expected}")
                print(f"Predicted: {predicted} {correct}")
                print(f"Similarity: {result['similarity']:.3f}")
                print(f"Confidence: {result['confidence']}")
                
            except Exception as e:
                print(f"Error: {e}")
        
        print(f"\n🎯 Embedding demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_iris_embeddings())
