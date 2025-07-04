import aiohttp
import asyncio
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataloader.util import encode_image_to_base64
from constants.prompts import FAMILIAL_RELATIONSHIP_PROMPT, SAME_PERSON_CONCISE_PROMPT
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Client:
    def __init__(self, base_url: str = "http://localhost:8000/v1", model_name: str = "gemma-3-27b-it",
                 max_retries: int = 3, retry_delay: float = 1.0, timeout: int = 30):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.chat_url = f"{self.base_url}/chat/completions"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _ensure_session(self):
        """Ensure session is created if not using context manager"""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self):
        """Close the session manually if not using context manager"""
        if self.session:
            await self.session.close()
            self.session = None

    async def is_same_person(self, face1_path: str, face2_path: str) -> str:
        """
        Determine if two faces belong to the same person using the vision model.
        Args:
            face1_path (str): Path to the first face image.
            face2_path (str): Path to the second face image.
        Returns:
            str: The model's response indicating if the faces are of the same person.
        """
        return await self._compare_two_faces(SAME_PERSON_CONCISE_PROMPT, face1_path, face2_path)

    async def is_same_person_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 8) -> List[str]:
        """
        Determine if multiple pairs of faces belong to the same person using asyncio batch processing.
        Args:
            pairs (List[Tuple[str, str]]): List of tuples containing paths to face image pairs.
            batch_size (int): Number of concurrent requests per batch.
        Returns:
            List[str]: List of model responses for each pair.
        """
        # Create all requests up front
        tasks = []
        for face1_path, face2_path in pairs:
            task = self._make_request(SAME_PERSON_CONCISE_PROMPT, face1_path, face2_path)
            tasks.append(task)

        # Process tasks in batches using asyncio.gather with progress bar
        results = []
        num_batches = (len(tasks) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(tasks), batch_size), desc="Processing batches", total=num_batches):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions and convert to strings
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Request failed: {result}")
                    results.append(f"Error: {str(result)}")
                else:
                    results.append(result)

        return results

    async def are_faces_related(self, face1_path: str, face2_path: str) -> str:
        """
        Determine if two faces are related using the vision model.
        Args:
            face1_path (str): Path to the first face image.
            face2_path (str): Path to the second face image.
        Returns:
            str: The model's response indicating if the faces are related.
        """
        return await self._compare_two_faces(FAMILIAL_RELATIONSHIP_PROMPT, face1_path, face2_path)

    async def are_faces_related_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 5) -> List[str]:
        """
        Determine if multiple pairs of faces are related using asyncio batch processing.
        Args:
            pairs (List[Tuple[str, str]]): List of tuples containing paths to face image pairs.
            batch_size (int): Number of concurrent requests per batch.
        Returns:
            List[str]: List of model responses for each pair.
        """
        # Create all requests up front
        tasks = []
        for face1_path, face2_path in pairs:
            task = self._make_request(FAMILIAL_RELATIONSHIP_PROMPT, face1_path, face2_path)
            tasks.append(task)

        # Process tasks in batches using asyncio.gather with progress bar
        results = []
        num_batches = (len(tasks) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(tasks), batch_size), desc="Processing batches", total=num_batches):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions and convert to strings
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Request failed: {result}")
                    results.append(f"Error: {str(result)}")
                else:
                    results.append(result)

        return results

    async def _make_request(self, prompt: str, face1_path: str, face2_path: str) -> str:
        """
        Make a single async request to the vision model API with retry mechanism.
        Args:
            prompt (str): The prompt to use for the comparison.
            face1_path (str): Path to the first face image.
            face2_path (str): Path to the second face image.
        Returns:
            str: The model's response to the comparison prompt.
        """
        await self._ensure_session()

        payload = self._build_payload(prompt, face1_path, face2_path)

        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.post(self.chat_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.warning(f"Request failed with status {response.status}: {error_text}")

                        if attempt < self.max_retries:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                            continue
                        else:
                            raise aiohttp.ClientError(f"Request failed after {self.max_retries} retries. Status: {response.status}")

            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise

            except aiohttp.ClientError as e:
                logger.warning(f"Client error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def _compare_two_faces(self, prompt: str, face1_path: str, face2_path: str) -> str:
        """
        Compare two faces using the vision model and a given prompt.
        Args:
            prompt (str): The prompt to use for the comparison.
            face1_path (str): Path to the first face image.
            face2_path (str): Path to the second face image.
        Returns:
            str: The model's response to the comparison prompt.
        """
        return await self._make_request(prompt, face1_path, face2_path)

    def _build_payload(self, prompt: str, image1_path: str, image2_path: str) -> Dict[str, Any]:
        """
        Build the JSON payload for the vision model API request.
        Args:
            prompt (str): The prompt text.
            image1_path (str): Path to the first image.
            image2_path (str): Path to the second image.
        Returns:
            Dict[str, Any]: The JSON payload for the API request.
        """
        image1_base64 = encode_image_to_base64(image1_path)
        image2_base64 = encode_image_to_base64(image2_path)

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
                                "url": f"data:image/png;base64,{image1_base64}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image2_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 64
        }
