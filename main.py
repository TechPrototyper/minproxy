#!/usr/bin/env python3
"""
MinProxy - Fehlertoleranter OpenAI-Format-Proxy

Sitzt zwischen LLM-Endpunkt (z.B. Qwen) und Clients,
normalisiert Responses auf 100% OpenAI-konformes Format,
besonders bei Tool-Calling.
"""

import os
import json
import time
import logging
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from normalizer import ToolCallNormalizer

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minproxy")

app = FastAPI(title="MinProxy", description="OpenAI Format Normalizer Proxy")

# Konfiguration aus Umgebungsvariablen
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://localhost:8080/v1")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
TIMEOUT = float(os.getenv("TIMEOUT", "120"))
DEFAULT_TOOL_MAX_TOKENS = int(os.getenv("DEFAULT_TOOL_MAX_TOKENS", "8192"))

normalizer = ToolCallNormalizer()
EMPTY_ASSISTANT_MARKERS = {"", "(empty)"}


def _sse_json(payload: dict) -> bytes:
    serialized = json.dumps(payload, ensure_ascii=False)
    return f"data: {serialized}\n\n".encode("utf-8")


def _tool_call_signature(
    message: dict[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    signature = []
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
            )
        signature.append((
            tool_call.get("id") or "",
            function.get("name") or "",
            arguments.strip(),
        ))
    return tuple(signature)


def _has_visible_content(content: Any) -> bool:
    return (
        isinstance(content, str)
        and content.strip() not in EMPTY_ASSISTANT_MARKERS
    )


def _normalize_request_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)

    for field in ("finish_reason", "reasoning", "_thinking_prefill"):
        normalized.pop(field, None)

    if normalized.get("role") != "assistant":
        return normalized

    tool_calls = normalized.get("tool_calls") or []
    if tool_calls:
        normalized["content"] = None

    content = normalized.get("content")
    if isinstance(content, str) and content.strip() in EMPTY_ASSISTANT_MARKERS:
        normalized["content"] = None

    return normalized


def _is_duplicate_empty_tool_turn(
    previous_messages: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> bool:
    if candidate.get("role") != "assistant" or not candidate.get("tool_calls"):
        return False
    if _has_visible_content(candidate.get("content")):
        return False
    if len(previous_messages) < 2:
        return False

    previous_tool = previous_messages[-1]
    previous_assistant = previous_messages[-2]
    if (
        previous_tool.get("role") != "tool"
        or previous_assistant.get("role") != "assistant"
    ):
        return False

    candidate_signature = _tool_call_signature(candidate)
    if (
        not candidate_signature
        or _tool_call_signature(previous_assistant) != candidate_signature
    ):
        return False

    candidate_ids = {
        tool_call[0]
        for tool_call in candidate_signature
        if tool_call[0]
    }
    tool_call_id = previous_tool.get("tool_call_id")
    return bool(tool_call_id and tool_call_id in candidate_ids)


def sanitize_request_body(body: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(body)

    raw_messages = body.get("messages") or []
    sanitized_messages: list[dict[str, Any]] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            sanitized_messages.append(raw_message)
            continue

        normalized_message = _normalize_request_message(raw_message)
        if _is_duplicate_empty_tool_turn(
            sanitized_messages,
            normalized_message,
        ):
            logger.info(
                "Dropping duplicate empty assistant tool turn for %s",
                [
                    tc.get("id")
                    for tc in normalized_message.get("tool_calls") or []
                ],
            )
            continue

        sanitized_messages.append(normalized_message)

    sanitized["messages"] = sanitized_messages

    has_tools = bool(sanitized.get("tools") or sanitized.get("functions"))
    has_output_limit = any(
        key in sanitized
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
    )
    if has_tools and not has_output_limit and DEFAULT_TOOL_MAX_TOKENS > 0:
        sanitized["max_tokens"] = DEFAULT_TOOL_MAX_TOKENS

    return sanitized


def build_stream_chunks_from_response(response: dict) -> list[bytes]:
    """Serialisiert eine normale Chat-Completion-Response in OpenAI-SSE-Chunks."""

    base = {
        "id": response.get("id"),
        "object": "chat.completion.chunk",
        "created": response.get("created", int(time.time())),
        "model": response.get("model", "unknown"),
    }
    chunks: list[bytes] = []

    for choice in response.get("choices", []):
        index = choice.get("index", 0)
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        chunks.append(_sse_json({
            **base,
            "choices": [{
                "index": index,
                "delta": {"role": message.get("role", "assistant")},
                "finish_reason": None,
            }],
        }))

        content = message.get("content")
        tool_calls = message.get("tool_calls") or []

        if content and not tool_calls:
            chunks.append(_sse_json({
                **base,
                "choices": [{
                    "index": index,
                    "delta": {"content": content},
                    "finish_reason": None,
                }],
            }))

        if tool_calls:
            serialized_tool_calls = []
            for tool_index, tool_call in enumerate(tool_calls):
                function = tool_call.get("function", {})
                serialized_tool_calls.append({
                    "index": tool_index,
                    "id": tool_call.get("id") or normalizer.generate_tool_call_id(),
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}"),
                    },
                })

            chunks.append(_sse_json({
                **base,
                "choices": [{
                    "index": index,
                    "delta": {"tool_calls": serialized_tool_calls},
                    "finish_reason": None,
                }],
            }))

        chunks.append(_sse_json({
            **base,
            "choices": [{
                "index": index,
                "delta": {},
                "finish_reason": finish_reason,
            }],
        }))

    chunks.append(b"data: [DONE]\n\n")
    return chunks


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy für Chat Completions mit Format-Normalisierung."""
    
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON body")

    body = sanitize_request_body(body)
    
    is_streaming = body.get("stream", False)
    
    # Headers für upstream
    headers = {
        "Content-Type": "application/json",
    }
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    
    # Original Auth-Header durchreichen falls kein eigener Key
    if not UPSTREAM_API_KEY:
        auth = request.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
    
    upstream_endpoint = f"{UPSTREAM_URL.rstrip('/')}/chat/completions"
    
    # Timeout-Konfiguration: längere Timeouts für LLMs
    timeout_config = httpx.Timeout(
        connect=30.0,
        read=TIMEOUT,
        write=30.0,
        pool=None
    )
    
    if is_streaming and (body.get("tools") or body.get("functions")):
        return await handle_buffered_stream(upstream_endpoint, headers, body, timeout_config)
    if is_streaming:
        # WICHTIG: Client NICHT im context manager für Streaming!
        # StreamingResponse braucht den Client am Leben
        return await handle_streaming(upstream_endpoint, headers, body, timeout_config)
    else:
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            return await handle_non_streaming(client, upstream_endpoint, headers, body)


async def handle_buffered_stream(
    url: str,
    headers: dict,
    body: dict,
    timeout_config: httpx.Timeout
) -> StreamingResponse:
    """Puffert Tool-Responses und streamt sie dann strikt im OpenAI-Format aus.

    Hintergrund: Manche Modelle senden in Streams erst reasoning/content und erst am Ende
    den eigentlichen Tool-Call. Viele Agenten behandeln das nicht mehr als ausführbaren
    Tool-Call. Für Requests mit Tools erzwingen wir deshalb saubere SSE-Chunks aus der
    finalen, normalisierten Response.
    """

    upstream_body = dict(body)
    upstream_body["stream"] = False

    async def stream_generator() -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                response = await client.post(url, headers=headers, json=upstream_body)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(f"Upstream buffered stream error: {e.response.status_code} - {e.response.text}")
                yield _sse_json({"error": e.response.text})
                yield b"data: [DONE]\n\n"
                return
            except httpx.RequestError as e:
                logger.error(f"Buffered stream request error: {e}")
                yield _sse_json({"error": str(e)})
                yield b"data: [DONE]\n\n"
                return

            try:
                data = response.json()
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from upstream buffered stream: {response.text[:500]}")
                yield _sse_json({"error": "Invalid JSON from upstream"})
                yield b"data: [DONE]\n\n"
                return

            normalized = normalizer.normalize_response(data)
            for chunk in build_stream_chunks_from_response(normalized):
                yield chunk

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


async def handle_non_streaming(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    body: dict
) -> JSONResponse:
    """Nicht-Streaming-Anfrage verarbeiten."""
    
    try:
        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error(f"Upstream error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(e.response.status_code, e.response.text)
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(502, f"Upstream connection error: {e}")
    
    try:
        data = response.json()
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from upstream: {response.text[:500]}")
        raise HTTPException(502, "Invalid JSON from upstream")
    
    # Normalisieren
    normalized = normalizer.normalize_response(data)
    
    logger.info(f"Normalized response: {json.dumps(normalized, indent=2)[:500]}")
    
    return JSONResponse(content=normalized)


async def handle_streaming(
    url: str,
    headers: dict,
    body: dict,
    timeout_config: httpx.Timeout
) -> StreamingResponse:
    """Streaming-Anfrage verarbeiten."""
    
    async def stream_generator() -> AsyncIterator[bytes]:
        # Client HIER erstellen - lebt so lange wie der Generator
        client = httpx.AsyncClient(timeout=timeout_config)
        
        try:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"Upstream stream error: {response.status_code} - {error_text}")
                    yield f"data: {json.dumps({'error': error_text.decode()})}\n\n".encode()
                    return
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:]
                        
                        if data_str.strip() == "[DONE]":
                            yield b"data: [DONE]\n\n"
                            continue
                        
                        try:
                            chunk = json.loads(data_str)
                            normalized_chunk = normalizer.normalize_stream_chunk(chunk)
                            yield f"data: {json.dumps(normalized_chunk)}\n\n".encode()
                        except json.JSONDecodeError:
                            # Versuche Tool-Calls aus Text zu extrahieren
                            logger.warning(f"Malformed chunk: {data_str[:200]}")
                            continue
                    else:
                        # Keine data: Zeile, trotzdem weiterleiten
                        yield f"{line}\n".encode()
                        
        except httpx.ReadTimeout:
            logger.error("Upstream read timeout")
            yield f"data: {json.dumps({'error': 'Upstream timeout'})}\n\n".encode()
        except httpx.RequestError as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()
        finally:
            await client.aclose()
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/v1/models")
async def list_models(request: Request):
    """Models-Endpunkt durchreichen."""
    headers = {}
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(f"{UPSTREAM_URL.rstrip('/')}/models", headers=headers)
            return JSONResponse(content=response.json())
        except Exception:
            return JSONResponse(content={
                "object": "list",
                "data": [{"id": "qwen", "object": "model", "owned_by": "local"}]
            })


@app.get("/health")
async def health():
    """Health-Check."""
    return {"status": "ok", "upstream": UPSTREAM_URL}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
