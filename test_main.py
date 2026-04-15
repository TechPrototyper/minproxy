import json

from main import build_stream_chunks_from_response, sanitize_request_body


def _decode_json_chunks(chunks: list[bytes]) -> list[dict]:
    payloads = []
    for chunk in chunks:
        text = chunk.decode("utf-8")
        if text == "data: [DONE]\n\n":
            continue
        assert text.startswith("data: ")
        payloads.append(json.loads(text[6:].strip()))
    return payloads


def test_build_stream_chunks_for_tool_calls_has_no_content_chunk():
    response = {
        "id": "chatcmpl-test",
        "created": 1,
        "model": "qwen",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "arguments": '{"timezone":"Europe/Berlin"}',
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }

    chunks = build_stream_chunks_from_response(response)
    payloads = _decode_json_chunks(chunks)

    assert len(payloads) == 3
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert (
        payloads[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
        == "get_time"
    )
    assert payloads[2]["choices"][0]["finish_reason"] == "tool_calls"
    assert all("content" not in payload["choices"][0]["delta"] for payload in payloads)
    assert chunks[-1] == b"data: [DONE]\n\n"


def test_build_stream_chunks_ignores_content_when_tool_calls_exist():
    response = {
        "id": "chatcmpl-test",
        "created": 1,
        "model": "qwen",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Ich rufe das Tool auf:",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "arguments": '{"timezone":"Europe/Berlin"}',
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }

    chunks = build_stream_chunks_from_response(response)
    payloads = _decode_json_chunks(chunks)

    assert len(payloads) == 3
    assert all("content" not in payload["choices"][0]["delta"] for payload in payloads)


def test_build_stream_chunks_for_content_streams_content_once():
    response = {
        "id": "chatcmpl-test",
        "created": 1,
        "model": "qwen",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hallo Welt",
            },
            "finish_reason": "stop",
        }],
    }

    chunks = build_stream_chunks_from_response(response)
    payloads = _decode_json_chunks(chunks)

    assert len(payloads) == 3
    assert payloads[1]["choices"][0]["delta"] == {"content": "Hallo Welt"}
    assert payloads[2]["choices"][0]["finish_reason"] == "stop"


def test_sanitize_request_body_nulls_empty_assistant_tool_content():
    body = {
        "model": "qwen",
        "messages": [{
            "role": "assistant",
            "content": "(empty)",
            "finish_reason": "tool_calls",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": '{"command":"pwd"}',
                },
            }],
        }],
        "tools": [{"type": "function"}],
    }

    sanitized = sanitize_request_body(body)

    assert sanitized["messages"][0]["content"] is None
    assert "finish_reason" not in sanitized["messages"][0]


def test_sanitize_request_body_drops_duplicate_empty_tool_turn_after_tool_result():
    body = {
        "model": "qwen",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"ls -la"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"output":"ok"}',
            },
            {
                "role": "assistant",
                "content": "(empty)",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"ls -la"}',
                    },
                }],
            },
            {
                "role": "user",
                "content": "Bitte weitermachen.",
            },
        ],
        "tools": [{"type": "function"}],
    }

    sanitized = sanitize_request_body(body)

    assert len(sanitized["messages"]) == 3
    assert [message["role"] for message in sanitized["messages"]] == [
        "assistant",
        "tool",
        "user",
    ]


def test_sanitize_request_body_adds_default_max_tokens_for_tool_requests(monkeypatch):
    monkeypatch.setattr("main.DEFAULT_TOOL_MAX_TOKENS", 4096)

    body = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"type": "function"}],
    }

    sanitized = sanitize_request_body(body)

    assert sanitized["max_tokens"] == 4096


def test_sanitize_request_body_keeps_explicit_max_tokens(monkeypatch):
    monkeypatch.setattr("main.DEFAULT_TOOL_MAX_TOKENS", 4096)

    body = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"type": "function"}],
        "max_tokens": 1024,
    }

    sanitized = sanitize_request_body(body)

    assert sanitized["max_tokens"] == 1024