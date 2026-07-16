"""OpenAI-compatible embedding client with dimension probing."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Wrapper around OpenAI-compatible embedding API."""

    def __init__(self, config: dict) -> None:
        self._base_url = config["EMBEDDING_BASE_URL"]
        self._api_key = config["EMBEDDING_API_KEY"]
        self._model = config["EMBEDDING_MODEL"]
        self._verify_ssl = config["LLM_VERIFY_SSL"]
        self._client: Any | None = None
        self._available = False

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx
            from openai import OpenAI

            extra_headers = {"X-Api-Key": self._api_key}
            http_client = httpx.Client(
                headers=extra_headers,
                verify=self._verify_ssl,
            )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                http_client=http_client,
            )
        return self._client

    def probe_dim(self) -> int:
        """Probe embedding dimension by embedding a short text."""
        try:
            emb = self.embed("dimension probe")
            dim = len(emb)
            logger.info("Detected embedding dimension: %s (model: %s)", dim, self._model)
            self._available = True
            return dim
        except Exception as e:
            logger.error("Failed to probe embedding dimension: %s", e)
            self._available = False
            raise

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            client = self._get_client()
            response = client.embeddings.create(input=texts, model=self._model)
            self._available = True
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error("Embedding API error: %s", e)
            self._available = False
            raise

    @property
    def available(self) -> bool:
        return self._available
