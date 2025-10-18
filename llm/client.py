import os
import requests
from pydantic import BaseModel
from typing import Optional, List
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenRouterResponse(BaseModel):
    summary: str
    title: Optional[str] = None
    topics: List[str]
    sentiment: str


class OpenRouterError(Exception):
    """Custom exception for OpenRouter related errors"""

    pass


class OpenRouterClient:
    def __init__(self):
        # Get OpenRouter API key from environment
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning(
                "OPENROUTER_API_KEY environment variable is not set. Some features may not work properly."
            )

        # OpenRouter API endpoint
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
        }

    def analyze_text(self, text: str) -> OpenRouterResponse:
        """
        Analyze text using OpenRouter to generate summary, title, topics, and sentiment
        """
        if not text or len(text.strip()) == 0:
            raise OpenRouterError("Input text cannot be empty")

        prompt = f"""
        Analyze the following text and provide:
        1. A 1-2 sentence summary
        2. A title (if relevant)
        3. 3 main topics
        4. Sentiment (positive, neutral, or negative)
        
        Text: {text}
        
        Respond in the following JSON format:
        {{
            "summary": "your summary here",
            "title": "your title here or null if not appropriate",
            "topics": ["topic1", "topic2", "topic3"],
            "sentiment": "positive/neutral/negative"
        }}
        """

        try:
            # Make a request to the OpenRouter API
            payload = {
                "model": "deepseek/deepseek-chat-v3.1:free",  # Using the free DeepSeek model
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 512,
            }

            response = requests.post(self.api_url, headers=self.headers, json=payload)

            if response.status_code != 200:
                logger.error(
                    f"OpenRouter API error: {response.status_code} - {response.text}"
                )
                raise OpenRouterError(f"OpenRouter API error: {response.status_code}")

            # Parse the response
            response_json = response.json()

            if (
                not response_json
                or "choices" not in response_json
                or len(response_json["choices"]) == 0
            ):
                raise OpenRouterError("Empty response from OpenRouter API")

            response_text = response_json["choices"][0]["message"]["content"]

            # Find JSON in the response (sometimes there might be extra text)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)

                # Ensure topics is a list of 3 strings
                topics = data.get("topics", [])
                if len(topics) < 3:
                    topics.extend([""] * (3 - len(topics)))
                elif len(topics) > 3:
                    topics = topics[:3]

                return OpenRouterResponse(
                    summary=data.get("summary", ""),
                    title=data.get("title", None),
                    topics=topics,
                    sentiment=data.get("sentiment", "neutral"),
                )
            else:
                logger.warning(f"No JSON found in OpenRouter response: {response_text}")
                # If no JSON found, return default response with proper structure
                return OpenRouterResponse(
                    summary="Summary could not be generated",
                    title=None,
                    topics=["topic1", "topic2", "topic3"],
                    sentiment="neutral",
                )

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling OpenRouter API: {str(e)}")
            raise OpenRouterError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing OpenRouter response as JSON: {str(e)}")
            raise OpenRouterError("Error parsing OpenRouter response.")
        except Exception as e:
            logger.error(f"Unexpected error calling OpenRouter API: {str(e)}")
            raise OpenRouterError(f"Unexpected error: {str(e)}")
