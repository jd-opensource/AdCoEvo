"""OpenAI-compatible chat client and JSON response extraction."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


def _estimate(*texts: str) -> int:
    """Approximate total token use when an API omits usage metadata."""
    total = 0.0
    for value in texts:
        for char in value:
            total += 0.6 if ord(char) > 127 else 0.25
    return int(total) + 1


@dataclass
class LLMResponse:
    text: str
    tokens: int           # Prompt and completion tokens; estimated when usage is omitted.
    ok: bool = True
    error: str = ""
    completion: str = ""  # Generated reply retained with its collection metadata.
    token_logprobs: Optional[List[float]] = None  # Behavior-policy log probabilities.
    completion_token_ids: Optional[List[int]] = None  # Token IDs aligned with log probabilities.
    latency_s: float = 0.0


class LLMClient:
    """OpenAI-compatible chat client with lazy SDK initialization."""

    def __init__(self, cfg: Dict[str, Any]):
        self.base_url = cfg.get("base_url", "http://localhost:8000/v1")
        self.model = cfg.get("model", "model")
        self.api_key = cfg.get("api_key", "EMPTY")
        self.temperature = float(cfg.get("temperature", 0.7))
        self.constrained_json = bool(cfg.get("constrained_json", True))
        self.top_p = cfg.get("top_p", None)  # None leaves the service default unchanged.
        self.min_p = cfg.get("min_p", None)  # Optional vLLM sampling parameter.
        self.max_tokens = int(cfg.get("max_tokens", 1024))
        self.timeout = float(cfg.get("timeout", 60))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.enable_thinking = cfg.get("enable_thinking", None)
        self.want_logprobs = bool(cfg.get("logprobs", False))
        self.seed = cfg.get("seed", None)
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                  timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def complete(self, system: str, user: str,
                 max_tokens: Optional[int] = None,
                 guided_json: Optional[Dict[str, Any]] = None) -> LLMResponse:
        started = time.perf_counter()
        try:
            client = self._ensure()
            extra: Dict[str, Any] = {}
            eb: Dict[str, Any] = {}
            if self.enable_thinking is not None:
                eb["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
            if self.min_p is not None:
                eb["min_p"] = float(self.min_p)
            if eb:
                extra["extra_body"] = eb
            if guided_json is not None and self.constrained_json:
                extra["response_format"] = {"type": "json_schema",
                                            "json_schema": {"name": "action", "schema": guided_json,
                                                            "strict": True}}
            if self.want_logprobs:
                extra["logprobs"] = True
            if self.top_p is not None:
                extra["top_p"] = float(self.top_p)
            if self.seed is not None:
                extra["seed"] = int(self.seed)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **extra,
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            tokens = getattr(usage, "total_tokens", 0) or _estimate(system, user, text)
            tlp = None
            ctids = None
            if self.want_logprobs:
                tlp = _extract_token_logprobs(resp.choices[0])
                ctids = _extract_token_ids(resp.choices[0])
            return LLMResponse(text=text, tokens=int(tokens), ok=True,
                               completion=text, token_logprobs=tlp,
                               completion_token_ids=ctids,
                               latency_s=time.perf_counter() - started)
        except Exception as e:  # The caller applies the role's execution fallback.
            return LLMResponse(text="", tokens=0, ok=False, error=f"{type(e).__name__}: {e}",
                               latency_s=time.perf_counter() - started)


def _extract_token_logprobs(choice: Any) -> Optional[List[float]]:
    """Read generated-token behavior log probabilities, or None if unavailable."""
    try:
        lp = getattr(choice, "logprobs", None)
        content = getattr(lp, "content", None) if lp is not None else None
        if not content:
            return None
        return [float(getattr(tok, "logprob", 0.0)) for tok in content]
    except Exception:
        return None


def _extract_token_ids(choice: Any) -> Optional[List[int]]:
    """Read exact generated token IDs from vLLM's token_id:NNNN representation.

    The server must use --return-tokens-as-token-ids. Missing or malformed IDs
    return None so the reply cannot be used for a policy update.
    """
    try:
        lp = getattr(choice, "logprobs", None)
        content = getattr(lp, "content", None) if lp is not None else None
        if not content:
            return None
        ids: List[int] = []
        for tok in content:
            s = getattr(tok, "token", "")
            if not isinstance(s, str) or not s.startswith("token_id:"):
                return None
            ids.append(int(s.split(":", 1)[1]))
        return ids
    except Exception:
        return None


# ---------------- 结构化输出解析 ----------------
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _balance(s: str) -> Optional[str]:
    """从首个 { 或 [ 起做括号配平，返回首个完整 JSON 子串。"""
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return s[start:j + 1]
    return None


def extract_json(text: str) -> Optional[Union[Dict, List]]:
    """从 LLM 输出抽取 JSON（优先 ```json 块，否则括号配平）。"""
    if not text:
        return None
    candidates: List[str] = []
    for m in _JSON_BLOCK.finditer(text):
        candidates.append(m.group(1).strip())
    bal = _balance(text)
    if bal:
        candidates.append(bal)
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            b = _balance(c)
            if b and b != c:
                try:
                    return json.loads(b)
                except json.JSONDecodeError:
                    continue
    return None
