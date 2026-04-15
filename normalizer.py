"""
Tool-Call Normalizer

Repariert und normalisiert LLM-Responses auf OpenAI-konformes Format.
Besonders für Tool-Calling mit fehlertoleranter Extraktion.
"""

import json
import re
import uuid
import time
from typing import Any, Optional
import html
import logging

logger = logging.getLogger("minproxy.normalizer")


class ToolCallNormalizer:
    """Normalisiert LLM-Responses auf OpenAI-Format."""
    
    # Regex-Patterns für Tool-Call-Extraktion aus Text
    PATTERNS = {
        # OpenAI-Style function_call in Text
        "function_call_json": re.compile(
            r'<function_call>\s*(\{.*?\})\s*</function_call>',
            re.DOTALL
        ),
        # Tool-Call mit Name und Arguments
        "tool_call_xml": re.compile(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
            re.DOTALL
        ),
        # Qwen-Style Action/Action Input
        "action_input": re.compile(
            r'Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})',
            re.DOTALL
        ),
        # Generisches JSON-Objekt mit "name" und "arguments"
        "generic_tool": re.compile(
            r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\}|\[[^\[\]]*\]|"[^"]*")[^{}]*\}',
            re.DOTALL
        ),
        # Function-name mit parameters
        "function_params": re.compile(
            r'\{[^{}]*"function"\s*:\s*\{[^{}]*"name"\s*:\s*"([^"]+)".*?"arguments"\s*:\s*("[^"]*"|\{[^{}]*\})',
            re.DOTALL
        ),
        # Anthropic-style Tool-Use
        "anthropic_tool": re.compile(
            r'<tool_use>\s*<name>([^<]+)</name>\s*<input>(\{.*?\})</input>\s*</tool_use>',
            re.DOTALL
        ),
        # Hermes/Nous-style
        "hermes_tool": re.compile(
            r'<tool_call>\s*\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{.*?\})\s*\}\s*</tool_call>',
            re.DOTALL
        ),
    }
    
    def __init__(self):
        self._call_counter = 0
    
    def generate_tool_call_id(self) -> str:
        """Generiert eine eindeutige Tool-Call-ID im OpenAI-Stil."""
        self._call_counter += 1
        return f"call_{uuid.uuid4().hex[:24]}"
    
    def normalize_response(self, response: dict) -> dict:
        """Normalisiert eine komplette Chat-Completion-Response."""
        
        if not isinstance(response, dict):
            logger.error(f"Response is not a dict: {type(response)}")
            return self._create_error_response("Invalid response format")
        
        # Grundstruktur sicherstellen
        normalized = {
            "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex[:29]}"),
            "object": "chat.completion",
            "created": response.get("created", int(time.time())),
            "model": response.get("model", "unknown"),
            "choices": [],
            "usage": response.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
        }
        
        # Choices verarbeiten
        choices = response.get("choices", [])
        if not choices:
            # Fallback: Versuche aus anderen Feldern zu extrahieren
            if "message" in response:
                choices = [{"message": response["message"], "index": 0}]
            elif "content" in response:
                choices = [{"message": {"content": response["content"]}, "index": 0}]
        
        for i, choice in enumerate(choices):
            normalized_choice = self._normalize_choice(choice, i)
            normalized["choices"].append(normalized_choice)
        
        return normalized
    
    def _normalize_choice(self, choice: dict, index: int) -> dict:
        """Normalisiert eine einzelne Choice."""
        
        message = choice.get("message", {})
        if not isinstance(message, dict):
            message = {"content": str(message)}
        
        # Message normalisieren
        normalized_message = self._normalize_message(message)
        
        # Finish reason bestimmen
        finish_reason = choice.get("finish_reason")
        if normalized_message.get("tool_calls"):
            finish_reason = "tool_calls"
        elif normalized_message.get("function_call"):
            finish_reason = "function_call"
        elif not finish_reason:
            finish_reason = "stop"
        
        # Alte Werte auf OpenAI mappens
        finish_reason_map = {
            "eos": "stop",
            "length": "length",
            "tool_call": "tool_calls",
            "function": "function_call",
        }
        finish_reason = finish_reason_map.get(finish_reason, finish_reason)
        
        return {
            "index": index,
            "message": normalized_message,
            "finish_reason": finish_reason,
            "logprobs": choice.get("logprobs"),
        }
    
    def _normalize_message(self, message: dict) -> dict:
        """Normalisiert eine Message inkl. Tool-Calls."""
        
        role = message.get("role", "assistant")
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        function_call = message.get("function_call")
        
        # Wenn tool_calls als String kommen (manche Modelle...)
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except json.JSONDecodeError:
                tool_calls = []
        
        # Tool-Calls normalisieren falls vorhanden
        normalized_tool_calls = []
        if tool_calls:
            for tc in tool_calls:
                normalized_tc = self._normalize_tool_call(tc)
                if normalized_tc:
                    normalized_tool_calls.append(normalized_tc)
        
        # Versuche Tool-Calls aus Content zu extrahieren
        if not normalized_tool_calls and content:
            extracted = self._extract_tool_calls_from_content(content)
            if extracted:
                normalized_tool_calls = extracted
                # Content bereinigen wenn Tool-Calls extrahiert
                content = self._clean_content_after_extraction(content)
        
        # Deprecated function_call zu tool_calls konvertieren
        if function_call and not normalized_tool_calls:
            converted = self._convert_function_call_to_tool_call(function_call)
            if converted:
                normalized_tool_calls = [converted]
        
        result = {
            "role": role,
            "content": content if content else None,
        }
        
        if normalized_tool_calls:
            result["content"] = None
            result["tool_calls"] = normalized_tool_calls
        
        return result
    
    def _normalize_tool_call(self, tool_call: dict) -> Optional[dict]:
        """Normalisiert einen einzelnen Tool-Call."""
        
        if not isinstance(tool_call, dict):
            return None
        
        # ID sicherstellen
        tc_id = tool_call.get("id")
        if not tc_id:
            tc_id = self.generate_tool_call_id()
        
        # Function-Daten extrahieren
        function_data = tool_call.get("function", {})
        if not function_data:
            # Flache Struktur (name/arguments direkt im tool_call)
            function_data = {
                "name": tool_call.get("name"),
                "arguments": tool_call.get("arguments"),
            }
        
        name = function_data.get("name")
        arguments = function_data.get("arguments", "{}")
        
        if not name:
            logger.warning(f"Tool call without name: {tool_call}")
            return None
        
        # Arguments normalisieren
        arguments = self._normalize_arguments(arguments)
        
        return {
            "id": tc_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments
            }
        }
    
    def _normalize_arguments(self, arguments: Any) -> str:
        """Stellt sicher, dass arguments ein JSON-String ist."""
        
        if arguments is None:
            return "{}"
        
        if isinstance(arguments, str):
            # Versuche zu parsen und neu zu serialisieren für konsistente Formatierung
            try:
                parsed = json.loads(arguments)
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                # Versuche repariertes JSON
                repaired = self._repair_json(arguments)
                if repaired:
                    return repaired
                # Fallback: Als leeres Objekt
                logger.warning(f"Could not parse arguments: {arguments[:100]}")
                return "{}"
        
        if isinstance(arguments, dict):
            return json.dumps(arguments, ensure_ascii=False)
        
        if isinstance(arguments, list):
            return json.dumps(arguments, ensure_ascii=False)
        
        # Andere Typen als String einpacken
        return json.dumps({"value": str(arguments)})
    
    def _repair_json(self, text: str) -> Optional[str]:
        """Versucht kaputtes JSON zu reparieren."""
        
        # Entferne Whitespace
        text = text.strip()
        
        # Häufige Fehler korrigieren
        repairs = [
            # Single quotes zu double quotes
            (r"'([^']*)'(?=\s*:)", r'"\1"'),
            (r":\s*'([^']*)'", r': "\1"'),
            # Trailing commas entfernen
            (r',\s*}', '}'),
            (r',\s*]', ']'),
            # Fehlende Quotes bei Keys
            (r'(\{|,)\s*(\w+)\s*:', r'\1"\2":'),
            # Python None/True/False zu JSON null/true/false
            (r'\bNone\b', 'null'),
            (r'\bTrue\b', 'true'),
            (r'\bFalse\b', 'false'),
        ]
        
        repaired = text
        for pattern, replacement in repairs:
            repaired = re.sub(pattern, replacement, repaired)
        
        try:
            parsed = json.loads(repaired)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        
        # Versuche nur den ersten JSON-Block zu extrahieren
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                parsed = json.loads(match.group())
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _extract_tool_calls_from_content(self, content: str) -> list:
        """Extrahiert Tool-Calls aus dem Content-Text."""
        
        if not content:
            return []
        
        tool_calls = self._extract_xml_parameter_tool_calls(content)
        
        # Durchsuche alle Pattern
        for pattern_name, pattern in self.PATTERNS.items():
            matches = pattern.findall(content)
            
            for match in matches:
                tool_call = self._parse_pattern_match(pattern_name, match)
                if tool_call:
                    tool_calls.append(tool_call)
        
        # Deduplizieren nach Name+Arguments
        seen = set()
        unique_calls = []
        for tc in tool_calls:
            key = (tc["function"]["name"], tc["function"]["arguments"])
            if key not in seen:
                seen.add(key)
                unique_calls.append(tc)
        
        return unique_calls
    
    def _extract_xml_parameter_tool_calls(self, content: str) -> list:
        """Extrahiert Tool-Calls aus tolerantem XML-/Tag-Markup.

        Beispiele:
        - <tool_use><name>terminal</name><parameters>...</parameters></tool_use>
        - <function_call> ... <parameter name="command">...</parameter> ...
        """

        if "<name>" not in content or "<parameter" not in content:
            return []

        block_pattern = re.compile(
            r"<(tool_use|function_call)>\s*(.*?)\s*</\1>",
            re.DOTALL | re.IGNORECASE,
        )
        name_pattern = re.compile(
            r"<name>\s*([^<]+?)\s*</name>",
            re.DOTALL | re.IGNORECASE,
        )
        parameter_pattern = re.compile(
            r'<parameter\s+name="([^"]+)">\s*(.*?)\s*</parameter>',
            re.DOTALL | re.IGNORECASE,
        )

        blocks = [block for _, block in block_pattern.findall(content)]
        if not blocks:
            blocks = [content]

        tool_calls = []
        for block in blocks:
            name_match = name_pattern.search(block)
            if not name_match:
                continue

            name = name_match.group(1).strip()
            parameters = {}
            for param_name, raw_value in parameter_pattern.findall(block):
                value = html.unescape(raw_value).strip()
                value = re.sub(r"\s+", " ", value)
                parameters[param_name] = value

            if not name or not parameters:
                continue

            tool_calls.append({
                "id": self.generate_tool_call_id(),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(parameters, ensure_ascii=False),
                },
            })

        return tool_calls

    def _parse_pattern_match(self, pattern_name: str, match) -> Optional[dict]:
        """Parst einen Pattern-Match zu einem Tool-Call."""
        
        try:
            if pattern_name in ("function_call_json", "tool_call_xml"):
                # Match ist ein JSON-String
                data = json.loads(match)
                name = data.get("name") or data.get("function", {}).get("name")
                args = data.get("arguments") or data.get("parameters") or {}
                
            elif pattern_name == "action_input":
                # Match ist (action_name, action_input)
                name, args_str = match
                args = json.loads(args_str)
                
            elif pattern_name == "generic_tool":
                # Match ist (name, arguments_str)
                name, args_str = match
                args = json.loads(args_str) if args_str.startswith('{') else args_str
                
            elif pattern_name == "function_params":
                # Match ist (name, arguments)  
                name, args_str = match
                args = json.loads(args_str.strip('"'))
                
            elif pattern_name in ("anthropic_tool", "hermes_tool"):
                # Match ist (name, input_json)
                name, args_str = match
                args = json.loads(args_str)
                
            else:
                return None
            
            if not name:
                return None
            
            return {
                "id": self.generate_tool_call_id(),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
                }
            }
            
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.debug(f"Failed to parse {pattern_name} match: {e}")
            return None
    
    def _clean_content_after_extraction(self, content: str) -> str:
        """Bereinigt Content nachdem Tool-Calls extrahiert wurden."""
        
        if not content:
            return ""
        
        cleaned = content
        
        # Entferne alle bekannten Tool-Call-Patterns
        for pattern in self.PATTERNS.values():
            cleaned = pattern.sub("", cleaned)
        
        # Entferne Action/Action Input Blöcke
        cleaned = re.sub(r'Action:\s*\w+\s*\nAction Input:\s*\{[^}]*\}', '', cleaned)
        
        # Bereinige Whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned if cleaned else None
    
    def _convert_function_call_to_tool_call(self, function_call: dict) -> Optional[dict]:
        """Konvertiert deprecated function_call zu tool_call."""
        
        if not isinstance(function_call, dict):
            return None
        
        name = function_call.get("name")
        arguments = function_call.get("arguments", "{}")
        
        if not name:
            return None
        
        return {
            "id": self.generate_tool_call_id(),
            "type": "function",
            "function": {
                "name": name,
                "arguments": self._normalize_arguments(arguments)
            }
        }
    
    def normalize_stream_chunk(self, chunk: dict) -> dict:
        """Normalisiert einen Streaming-Chunk."""
        
        if not isinstance(chunk, dict):
            return chunk
        
        normalized = {
            "id": chunk.get("id", f"chatcmpl-{uuid.uuid4().hex[:29]}"),
            "object": "chat.completion.chunk",
            "created": chunk.get("created", int(time.time())),
            "model": chunk.get("model", "unknown"),
            "choices": [],
        }
        
        choices = chunk.get("choices", [])
        for i, choice in enumerate(choices):
            delta = dict(choice.get("delta", {}))
            
            # Qwen reasoning_content → ignorieren (oder separat behandeln)
            # OpenAI-konforme Clients erwarten nur "content"
            if "reasoning_content" in delta:
                # Optional: Als separates Feld behalten für Clients die es verstehen
                # Aber nicht als "content" durchreichen - das verwirrt Agents
                del delta["reasoning_content"]
            
            # Tool-Calls in Delta normalisieren
            if "tool_calls" in delta:
                delta["tool_calls"] = [
                    self._normalize_stream_tool_call(tc) 
                    for tc in delta["tool_calls"]
                ]
            
            normalized_choice = {
                "index": choice.get("index", i),
                "delta": delta,
                "finish_reason": choice.get("finish_reason"),
            }
            normalized["choices"].append(normalized_choice)
        
        return normalized
    
    def _normalize_stream_tool_call(self, tool_call: dict) -> dict:
        """Normalisiert einen Tool-Call in einem Stream-Delta."""
        
        result = {
            "index": tool_call.get("index", 0),
        }
        
        if "id" in tool_call:
            result["id"] = tool_call["id"] or self.generate_tool_call_id()
        
        if "type" in tool_call:
            result["type"] = "function"
        
        if "function" in tool_call:
            func = tool_call["function"]
            result["function"] = {}
            if "name" in func:
                result["function"]["name"] = func["name"]
            if "arguments" in func:
                result["function"]["arguments"] = func["arguments"]
        
        return result
    
    def _create_error_response(self, error_message: str) -> dict:
        """Erstellt eine Fehler-Response im OpenAI-Format."""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "error",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Error: {error_message}"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
