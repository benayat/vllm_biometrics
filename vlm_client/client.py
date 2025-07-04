from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionContentPartTextParam, \
    ChatCompletionContentPartImageParam
from dataloader.util import encode_image_to_base64
from constants.prompts import FAMILIAL_RELATIONSHIP_PROMPT, SAME_PERSON_PROMPT, SAME_PERSON_CONCISE_PROMPT
from typing import List, Tuple
import asyncio
from tqdm import tqdm


class Client:
    def __init__(self, base_url: str = "http://10.100.102.24:8000/v1", model_name: str = "gemma-3-27b-it"):
        self.model_name = model_name
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="none"
        )

    async def is_same_person(self, face1_path: str, face2_path: str) -> str:
        """
        Determine if two faces belong to the same person using the OpenAI client.
        Args:
            face1_path (str): Path to the first face image.
            face2_path (str): Path to the second face image.
        Returns:
            str: The model's response indicating if the faces are of the same person.
        """
        return await self._compare_two_faces(SAME_PERSON_CONCISE_PROMPT, face1_path, face2_path)

    async def is_same_person_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 5) -> List[str]:
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
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

        return results

    async def are_faces_related(self, face1_path: str, face2_path: str) -> str:
        """
        Determine if two faces are related using the OpenAI client.
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
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

        return results

    async def _make_request(self, prompt: str, face1_path: str, face2_path: str) -> str:
        """
        Make a single async request to the OpenAI API.
        Args:
            prompt (str): The prompt to use for the comparison.
            face1_path (str): Path to the first face image.
            face2_path (str): Path to the second face image.
        Returns:
            str: The model's response to the comparison prompt.
        """
        user_message = self._get_user_message(prompt, face1_path, face2_path)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[user_message],
            temperature=0.2,
            max_tokens=64
        )

        return response.choices[0].message.content

    async def _compare_two_faces(self, prompt: str, face1_path: str, face2_path: str) -> str:
        """
        Compare two faces using the OpenAI client and a given prompt.
        Args:
            prompt (str): The prompt to use for the comparison.
            face1_path (str): Path to the first face image.
            face2_path (str): Path to the second face image.
        Returns:
            str: The model's response to the comparison prompt.
        """
        user_message = self._get_user_message(prompt, face1_path, face2_path)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[user_message],
            temperature=0.2,
            max_tokens=64
        )

        return response.choices[0].message.content

    @staticmethod
    def _get_user_message(prompt: str, image1_path: str, image2_path: str) -> ChatCompletionUserMessageParam:
        """
        Construct a user message for the OpenAI chat API with a prompt and two images.
        Args:
            prompt (str): The prompt text.
            image1_path (str): Path to the first image.
            image2_path (str): Path to the second image.
        Returns:
            ChatCompletionUserMessageParam: The user message with text and image content.
        """
        image1_base64 = encode_image_to_base64(image1_path)
        image2_base64 = encode_image_to_base64(image2_path)
        return {
            "role": "user",
            "content": [
                ChatCompletionContentPartTextParam(
                    type="text",
                    text=prompt
                ),
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={
                        "url": f"data:image/png;base64,{image1_base64}"
                    }
                ),
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={
                        "url": f"data:image/png;base64,{image2_base64}"
                    }
                )
            ]
        }
