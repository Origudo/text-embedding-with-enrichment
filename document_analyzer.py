"""
Document Analyzer

Uses a local LLM (LM Studio or Ollama) to extract summary, tags,
categories, and keywords from a file.
"""

import json
import logging
from pathlib import Path

import requests

from settings import PROVIDERS, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """Analyze documents using a local LLM backend."""

    def __init__(self, provider: str | None = None, model: str | None = None, timeout: int | None = None):
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self.timeout = timeout or settings.llm_timeout
        self._prompt_template = settings.analysis_prompt

        if self.provider not in PROVIDERS:
            raise ValueError(f"Provider must be one of {list(PROVIDERS)}")

        self._running = self._check_health()

    def _check_health(self) -> bool:
        """Check whether the LLM backend is reachable."""
        url = PROVIDERS[self.provider]["health_url"]
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                logger.info(f"{self.provider} is running at {url}")
                return True
        except requests.ConnectionError:
            pass
        logger.warning(f"{self.provider} is not reachable at {url}")
        return False

    def is_running(self) -> bool:
        """True if the LLM backend was reachable at init."""
        return self._running

    def analyze(self, file_path: str) -> dict:
        """
        Read a file and return its document topic, type, and keywords.

        Args:
            file_path: Path to a text file to analyze.

        Returns:
            A dict with keys: document_topic, document_type, keywords.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        text = path.read_text(encoding="utf-8")

        prompt = self._prompt_template.replace("{text}", text[:30_000])

        if self._running:
            result = self._call_llm(prompt)
            if result:
                return result

        raise RuntimeError(
            f"{self.provider} is not available. Start the server and try again."
        )

    def _call_llm(self, prompt: str) -> dict | None:
        """Send prompt to the LLM and parse the JSON response."""
        cfg = PROVIDERS[self.provider]

        if self.provider == "lmstudio":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1024,
            }
        else:  # ollama
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3},
            }

        try:
            r = requests.post(cfg["chat_url"], json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

        raw = ""
        if self.provider == "lmstudio":
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            raw = data.get("message", {}).get("content", "")

        # Strip markdown fences if present
        raw = raw.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(raw)
            return {
                "document_topic": parsed.get("document_topic", ""),
                "document_type": parsed.get("document_type", ""),
                "keywords": parsed.get("keywords", []),
            }
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_analyzer.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyzer = DocumentAnalyzer()

    if not analyzer.is_running():
        print(f"Error: {analyzer.provider} is not available. Start the server and try again.")
        sys.exit(1)

    result = analyzer.analyze(file_path)
    print(f"\nDocument Topic: {result['document_topic']}")
    print(f"Document Type:  {result['document_type']}")
    print(f"Keywords:       {', '.join(result['keywords'])}")
