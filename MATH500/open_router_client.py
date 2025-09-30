#!/usr/bin/env python3
"""
OpenRouter client wrapper for chat completions with basic retries and
safe reasoning metadata extraction. Designed to be used by
`MATH500/math500_evaluator.py`.
"""
from __future__ import annotations

import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model_id: str = "",
        reasoning_effort: Optional[str] = None,
        reasoning_max_tokens: Optional[int] = None,
        reasoning_exclude: Optional[bool] = None,
        reasoning_enabled: Optional[bool] = None,
        save_reasoning_summary: bool = False,
        request_timeout: int = 120,
        max_retries: int = 2,
        backoff_seconds: float = 1.5,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.reasoning_max_tokens = reasoning_max_tokens
        self.reasoning_exclude = reasoning_exclude
        self.reasoning_enabled = reasoning_enabled
        self.save_reasoning_summary = save_reasoning_summary
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, Dict[str, Any]]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Optional reasoning configuration
        reasoning_cfg: Dict[str, Any] = {}
        if self.reasoning_effort is not None:
            reasoning_cfg["effort"] = self.reasoning_effort
        if self.reasoning_max_tokens is not None:
            reasoning_cfg["max_tokens"] = self.reasoning_max_tokens
        if self.reasoning_exclude is not None:
            reasoning_cfg["exclude"] = self.reasoning_exclude
        if self.reasoning_enabled is not None:
            reasoning_cfg["enabled"] = self.reasoning_enabled
        if reasoning_cfg:
            payload["reasoning"] = reasoning_cfg

        # Basic retry loop on network/server issues
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                if resp.status_code != 200:
                    # Retry on 5xx; otherwise raise
                    if 500 <= resp.status_code < 600 and attempt < self.max_retries:
                        logger.warning(
                            f"OpenRouter API {resp.status_code} on attempt {attempt+1}; retrying..."
                        )
                        time.sleep(self.backoff_seconds * (attempt + 1))
                        continue
                    raise RuntimeError(
                        f"OpenRouter API error {resp.status_code}: {resp.text[:500]}"
                    )

                data = resp.json()
                content = self._extract_content(data)
                meta = self._extract_reasoning_meta(data)

                # If the API reported completion tokens but content is empty,
                # persist diagnostics to meta for visibility
                if not content or not content.strip():
                    choice0 = (data.get("choices") or [{}])[0]
                    meta.setdefault("diagnostics", {})
                    meta["diagnostics"].update(
                        {
                            "empty_content": True,
                            "finish_reason": choice0.get("finish_reason"),
                            "has_message": bool(choice0.get("message")),
                            "message_keys": list((choice0.get("message") or {}).keys()),
                            "raw_choice_type": choice0.get("type"),
                        }
                    )
                    logger.warning(
                        "OpenRouter returned empty content. finish_reason=%s",
                        choice0.get("finish_reason"),
                    )

                return content or "", meta

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "OpenRouter request failed on attempt %d/%d: %s; retrying...",
                        attempt + 1,
                        self.max_retries + 1,
                        str(e),
                    )
                    time.sleep(self.backoff_seconds * (attempt + 1))
                    continue
                else:
                    logger.error("OpenRouter request failed: %s", str(e))
                    raise

        # Unreachable due to raise above, but satisfy type checker
        if last_error:
            raise last_error
        return "", {}

    def _extract_content(self, data: Dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"].get("content", "")
        except Exception:
            return ""

    def _extract_reasoning_meta(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract safe reasoning metadata from the OpenRouter response.
        Matches the schema expected by the evaluator.
        """
        meta: Dict[str, Any] = {
            "reasoning_present": False,
            "counts": {"summary": 0, "text": 0, "encrypted": 0},
            "total_text_chars": 0,
        }
        try:
            msg = (data.get("choices") or [{}])[0].get("message", {})
            details = msg.get("reasoning_details") or []
            if details:
                meta["reasoning_present"] = True
                summaries: List[str] = []
                for d in details:
                    t = d.get("type")
                    if t == "reasoning.summary":
                        meta["counts"]["summary"] += 1
                        if isinstance(d.get("summary"), str):
                            summaries.append(d.get("summary"))
                    elif t == "reasoning.text":
                        meta["counts"]["text"] += 1
                        txt = d.get("text")
                        if isinstance(txt, str):
                            meta["total_text_chars"] += len(txt)
                    elif t == "reasoning.encrypted":
                        meta["counts"]["encrypted"] += 1
                if summaries:
                    # This field will be consumed by the evaluator iff it asked to save summaries
                    meta["summaries"] = summaries
        except Exception:
            pass
        if isinstance(data.get("usage"), dict):
            meta["usage"] = data["usage"]
        return meta
