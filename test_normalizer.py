"""
Tests für den Tool-Call-Normalizer.
"""

import json

import pytest

from normalizer import ToolCallNormalizer


@pytest.fixture
def normalizer():
    return ToolCallNormalizer()


class TestArgumentNormalization:
    """Tests für die Arguments-Normalisierung."""

    def test_dict_to_string(self, normalizer):
        result = normalizer._normalize_arguments({"key": "value"})
        assert result == '{"key": "value"}'

    def test_string_passthrough(self, normalizer):
        result = normalizer._normalize_arguments('{"key": "value"}')
        assert json.loads(result) == {"key": "value"}

    def test_malformed_json_single_quotes(self, normalizer):
        result = normalizer._normalize_arguments("{'key': 'value'}")
        assert json.loads(result) == {"key": "value"}

    def test_none_to_empty_object(self, normalizer):
        result = normalizer._normalize_arguments(None)
        assert result == "{}"

    def test_python_constants(self, normalizer):
        result = normalizer._normalize_arguments('{"active": True, "data": None}')
        parsed = json.loads(result)
        assert parsed == {"active": True, "data": None}


class TestToolCallNormalization:
    """Tests für Tool-Call-Normalisierung."""

    def test_complete_tool_call(self, normalizer):
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Berlin"}',
            },
        }
        result = normalizer._normalize_tool_call(tool_call)
        assert result["id"] == "call_123"
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"

    def test_missing_id_generated(self, normalizer):
        tool_call = {
            "function": {
                "name": "test_func",
                "arguments": "{}",
            }
        }
        result = normalizer._normalize_tool_call(tool_call)
        assert result["id"].startswith("call_")
        assert len(result["id"]) > 10

    def test_flat_structure(self, normalizer):
        tool_call = {
            "name": "search",
            "arguments": '{"query": "test"}',
        }
        result = normalizer._normalize_tool_call(tool_call)
        assert result["function"]["name"] == "search"
        assert result["type"] == "function"

    def test_dict_arguments(self, normalizer):
        tool_call = {
            "id": "call_abc",
            "function": {
                "name": "calc",
                "arguments": {"x": 1, "y": 2},
            },
        }
        result = normalizer._normalize_tool_call(tool_call)
        assert isinstance(result["function"]["arguments"], str)
        assert json.loads(result["function"]["arguments"]) == {"x": 1, "y": 2}


class TestContentExtraction:
    """Tests für Tool-Call-Extraktion aus Content."""

    def test_xml_function_call(self, normalizer):
        content = (
            "Ich werde das Wetter abrufen.\n"
            '<function_call>{"name": "get_weather", "arguments": '
            '{"city": "Berlin"}}</function_call>\n'
        )
        calls = normalizer._extract_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"

    def test_tool_call_xml(self, normalizer):
        content = '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>'
        calls = normalizer._extract_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "search"

    def test_action_input_format(self, normalizer):
        content = (
            "Ich analysiere die Anfrage.\n"
            "Action: get_data\n"
            'Action Input: {"id": 123}\n\n'
            "Das Ergebnis..."
        )
        calls = normalizer._extract_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_data"

    def test_hermes_format(self, normalizer):
        content = '<tool_call>{"name": "calculate", "arguments": {"expression": "2+2"}}</tool_call>'
        calls = normalizer._extract_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "calculate"

    def test_no_tool_calls(self, normalizer):
        content = "Dies ist eine normale Antwort ohne Tool-Calls."
        calls = normalizer._extract_tool_calls_from_content(content)
        assert len(calls) == 0

    def test_xml_parameter_tool_call(self, normalizer):
        content = """<tool_use>
<name>terminal</name>
<parameters>
<parameter name="command">echo \"=== TEST 1: Einfache Command ===\" && date && whoami</parameter>
</parameters>
</tool_use>"""
        calls = normalizer._extract_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "terminal"
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "command": 'echo "=== TEST 1: Einfache Command ===" && date && whoami'
        }

    def test_malformed_xml_wrapper_tool_call(self, normalizer):
        content = """<function_call>
<tool_use>
<name>terminal</name>
<parameters>
<parameter name="command">echo \"=== TEST 1: Einfache Command ===\" && date && whoami</parameter>
</parameters>
</tool_use>
</parameters>
</function_call>"""
        calls = normalizer._extract_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "terminal"


class TestResponseNormalization:
    """Tests fuer komplette Response-Normalisierung."""

    def test_minimal_response(self, normalizer):
        response = {
            "choices": [{
                "message": {
                    "content": "Hello!",
                }
            }]
        }
        result = normalizer.normalize_response(response)

        assert "id" in result
        assert result["object"] == "chat.completion"
        assert "created" in result
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_tool_call_response(self, normalizer):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_xyz",
                        "function": {
                            "name": "test",
                            "arguments": "{}",
                        },
                    }],
                }
            }]
        }
        result = normalizer.normalize_response(response)

        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert result["choices"][0]["message"]["content"] is None
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1

    def test_tool_call_response_drops_assistant_text(self, normalizer):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Ich rufe jetzt das Tool auf.",
                    "tool_calls": [{
                        "id": "call_xyz",
                        "function": {
                            "name": "test",
                            "arguments": "{}",
                        },
                    }],
                }
            }]
        }
        result = normalizer.normalize_response(response)

        assert result["choices"][0]["message"]["content"] is None

    def test_embedded_tool_call(self, normalizer):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        "Ich rufe die Funktion auf.\n"
                        '<function_call>{"name": "greet", "arguments": '
                        '{"name": "Tim"}}</function_call>'
                    ),
                }
            }]
        }
        result = normalizer.normalize_response(response)

        message = result["choices"][0]["message"]
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] == "greet"

    def test_finish_reason_mapping(self, normalizer):
        for old, expected in [("eos", "stop"), ("tool_call", "tool_calls")]:
            response = {
                "choices": [{
                    "message": {"content": "test"},
                    "finish_reason": old,
                }]
            }
            result = normalizer.normalize_response(response)
            assert result["choices"][0]["finish_reason"] == expected

    def test_function_call_to_tool_call(self, normalizer):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "old_style_func",
                        "arguments": '{"param": "value"}',
                    },
                }
            }]
        }
        result = normalizer.normalize_response(response)

        message = result["choices"][0]["message"]
        assert "tool_calls" in message
        assert message["tool_calls"][0]["function"]["name"] == "old_style_func"

    def test_xml_content_is_converted_to_tool_calls(self, normalizer):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": """<tool_use>
<name>terminal</name>
<parameters>
<parameter name="command">echo \"=== TEST 1: Einfache Command ===\" && date && whoami</parameter>
</parameters>
</tool_use>""",
                }
            }]
        }

        result = normalizer.normalize_response(response)
        message = result["choices"][0]["message"]

        assert message["content"] is None
        assert message["tool_calls"][0]["function"]["name"] == "terminal"
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
            "command": 'echo "=== TEST 1: Einfache Command ===" && date && whoami'
        }
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_explicit_stop_is_overridden_for_tool_calls(self, normalizer):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": """<tool_use>
<name>terminal</name>
<parameters>
<parameter name="command">date</parameter>
</parameters>
</tool_use>""",
                },
                "finish_reason": "stop",
            }]
        }

        result = normalizer.normalize_response(response)

        assert result["choices"][0]["finish_reason"] == "tool_calls"


class TestStreamChunkNormalization:
    """Tests fuer Streaming-Chunk-Normalisierung."""

    def test_basic_chunk(self, normalizer):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None,
            }]
        }
        result = normalizer.normalize_stream_chunk(chunk)

        assert result["object"] == "chat.completion.chunk"
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_tool_call_chunk(self, normalizer):
        chunk = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "test"},
                    }]
                }
            }]
        }
        result = normalizer.normalize_stream_chunk(chunk)

        tool_call = result["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "test"
    
    def test_dict_to_string(self, normalizer):
        result = normalizer._normalize_arguments({"key": "value"})
        assert result == '{"key": "value"}'
    
    def test_string_passthrough(self, normalizer):
        result = normalizer._normalize_arguments('{"key": "value"}')
        assert json.loads(result) == {"key": "value"}
    
    def test_malformed_json_single_quotes(self, normalizer):
        result = normalizer._normalize_arguments("{'key': 'value'}")
        assert json.loads(result) == {"key": "value"}
    
    def test_none_to_empty_object(self, normalizer):
        result = normalizer._normalize_arguments(None)
        assert result == "{}"
    
    def test_python_constants(self, normalizer):
        result = normalizer._normalize_arguments('{"active": True, "data": None}')
        parsed = json.loads(result)
        assert parsed == {"active": True, "data": None}


class TestToolCallNormalization:
    """Tests für Tool-Call-Normalisierung."""
    
    def test_complete_tool_call(self, normalizer):
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Berlin"}'
            }
        }
        result = normalizer._normalize_tool_call(tool_call)
        assert result["id"] == "call_123"
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
    
    def test_missing_id_generated(self, normalizer):
        tool_call = {
            "function": {
                "name": "test_func",
                "arguments": "{}"
            }
        }
        result = normalizer._normalize_tool_call(tool_call)
        assert result["id"].startswith("call_")
        assert len(result["id"]) > 10
    
    def test_flat_structure(self, normalizer):
        """Tool-Call ohne verschachtelte function-Struktur."""
        tool_call = {
            "name": "search",
            "arguments": '{"query": "test"}'
        }
        result = normalizer._normalize_tool_call(tool_call)
        assert result["function"]["name"] == "search"
        assert result["type"] == "function"
    
    def test_dict_arguments(self, normalizer):
        """Arguments als Dict statt String."""
        tool_call = {
            "id": "call_abc",
            "function": {
                "name": "calc",
                "arguments": {"x": 1, "y": 2}
            }
        }
        result = normalizer._normalize_tool_call(tool_call)
        assert isinstance(result["function"]["arguments"], str)
        assert json.loads(result["function"]["arguments"]) == {"x": 1, "y": 2}


class TestContentExtraction:
    """Tests für Tool-Call-Extraktion aus Content."""
    
    def test_xml_function_call(self, normalizer):
        content = '''Ich werde das Wetter abrufen.
<function_call>{"name": "get_weather", "arguments": {"city": "Berlin"}}</function_call>
'''
        calls = normalizer._extract_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
    
    def test_tool_call_xml(self, normalizer):
        content = '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>'
        calls = normalizer._extract_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "search"
    
    def test_action_input_format(self, normalizer):
        content = '''Ich analysiere die Anfrage.
Action: get_data
Action Input: {"id": 123}

Das Ergebnis...'''
        calls = normalizer._extract_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_data"

    def test_hermes_format(self, normalizer):
        content = '<tool_call>{"name": "calculate", "arguments": {"expression": "2+2"}}</tool_call>'
        calls = normalizer._extract_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "calculate"
    
    def test_no_tool_calls(self, normalizer):
        content = "Dies ist eine normale Antwort ohne Tool-Calls."
        calls = normalizer._extract_tool_calls_from_content(content)
        assert len(calls) == 0


class TestResponseNormalization:
    """Tests für komplette Response-Normalisierung."""
    
    def test_minimal_response(self, normalizer):
        response = {
            "choices": [{
                "message": {
                    "content": "Hello!"
                }
            }]
        }
        result = normalizer.normalize_response(response)
        
        assert "id" in result
        assert result["object"] == "chat.completion"
        assert "created" in result
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
    
    def test_tool_call_response(self, normalizer):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_xyz",
                        "function": {
                            "name": "test",
                            "arguments": "{}"
                        }
                    }]
                }
            }]
        }
        result = normalizer.normalize_response(response)
        
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert result["choices"][0]["message"]["content"] is None
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1

    def test_tool_call_response_drops_assistant_text(self, normalizer):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Ich rufe jetzt das Tool auf.",
                    "tool_calls": [{
                        "id": "call_xyz",
                        "function": {
                            "name": "test",
                            "arguments": "{}"
                        }
                    }]
                }
            }]
        }
        result = normalizer.normalize_response(response)

        assert result["choices"][0]["message"]["content"] is None
    
    def test_embedded_tool_call(self, normalizer):
        """Tool-Call im Content statt im tool_calls-Feld."""
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": 'Ich rufe die Funktion auf.\n<function_call>{"name": "greet", "arguments": {"name": "Tim"}}</function_call>'
                }
            }]
        }
        result = normalizer.normalize_response(response)
        
        message = result["choices"][0]["message"]
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] == "greet"
    
    def test_finish_reason_mapping(self, normalizer):
        """Verschiedene finish_reason-Werte korrekt mappen."""
        for old, expected in [("eos", "stop"), ("tool_call", "tool_calls")]:
            response = {
                "choices": [{
                    "message": {"content": "test"},
                    "finish_reason": old
                }]
            }
            result = normalizer.normalize_response(response)
            assert result["choices"][0]["finish_reason"] == expected
    
    def test_function_call_to_tool_call(self, normalizer):
        """Deprecated function_call zu tool_calls konvertieren."""
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "old_style_func",
                        "arguments": '{"param": "value"}'
                    }
                }
            }]
        }
        result = normalizer.normalize_response(response)
        
        message = result["choices"][0]["message"]
        assert "tool_calls" in message
        assert message["tool_calls"][0]["function"]["name"] == "old_style_func"

    def test_xml_content_is_converted_to_tool_calls(self, normalizer):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": """<tool_use>
<name>terminal</name>
<parameters>
<parameter name=\"command\">echo \"=== TEST 1: Einfache Command ===\" && date && whoami</parameter>
</parameters>
</tool_use>"""
                }
            }]
        }

        result = normalizer.normalize_response(response)
        message = result["choices"][0]["message"]

        assert message["content"] is None
        assert message["tool_calls"][0]["function"]["name"] == "terminal"
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
            "command": 'echo "=== TEST 1: Einfache Command ===" && date && whoami'
        }
        assert result["choices"][0]["finish_reason"] == "tool_calls"


class TestStreamChunkNormalization:
    """Tests für Streaming-Chunk-Normalisierung."""
    
    def test_basic_chunk(self, normalizer):
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        }
        result = normalizer.normalize_stream_chunk(chunk)
        
        assert result["object"] == "chat.completion.chunk"
        assert result["choices"][0]["delta"]["content"] == "Hello"
    
    def test_tool_call_chunk(self, normalizer):
        chunk = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "test"}
                    }]
                }
            }]
        }
        result = normalizer.normalize_stream_chunk(chunk)
        
        tc = result["choices"][0]["delta"]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
