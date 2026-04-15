import json

from main import build_stream_chunks_from_response


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