# MinProxy - OpenAI Format Normalizer

Fehlertoleranter Proxy, der LLM-Responses (z.B. von Qwen, Llama, etc.) auf 100% OpenAI-konformes Format normalisiert.

## Features

- **Tool-Call-Normalisierung**: Repariert fehlende IDs, falsches Format, Dict statt JSON-String
- **Content-Extraktion**: Erkennt Tool-Calls in verschiedenen Formaten im Content-Text:
  - `<function_call>{...}</function_call>`
  - `<tool_call>{...}</tool_call>`
  - `Action: name\nAction Input: {...}`
  - Anthropic-Style `<tool_use>...</tool_use>`
- **JSON-Reparatur**: Korrigiert häufige JSON-Fehler (Single-Quotes, Trailing-Commas, Python-Konstanten)
- **Streaming-Support**: Normalisiert auch SSE-Stream-Chunks
- **Tool-Stream-Härtung**: Requests mit `tools` werden upstream gepuffert und als saubere OpenAI-SSE-Chunks re-emittiert, damit Agenten Tool-Calls zuverlässig ausführen
- **Request-Härtung**: Bereinigt Assistant-History mit leeren Tool-Call-Turns (`""`, `(empty)`) und entfernt offensichtliche Duplikate aus Retry-Schleifen, bevor sie das Modell erneut verwirren
- **Token-Default für Tool-Requests**: Setzt bei fehlendem Output-Limit automatisch ein sinnvolles `max_tokens`, damit Tool-Calls und Abschlussantworten seltener auf `finish_reason="length"` laufen
- **Finish-Reason-Mapping**: Konvertiert nicht-standard Werte (`eos` → `stop`, etc.)

## Installation

```bash
cd minproxy
pip install -r requirements.txt
```

## Verwendung

```bash
# Upstream-URL setzen (dein Qwen-Endpunkt)
export UPSTREAM_URL="http://localhost:8080/v1"
export UPSTREAM_API_KEY=""  # Falls nötig

# Proxy starten
python main.py

# Oder mit uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

Dann Agenten auf `http://localhost:8000/v1` zeigen statt direkt auf den LLM.

## Beispiel

**Input vom LLM (fehlerhaft):**
```json
{
  "choices": [{
    "message": {
      "content": "Ich rufe die Funktion auf.\n<function_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\"}}</function_call>"
    },
    "finish_reason": "eos"
  }]
}
```

**Output vom Proxy (OpenAI-konform):**
```json
{
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1699999999,
  "model": "qwen",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Ich rufe die Funktion auf.",
      "tool_calls": [{
        "id": "call_7f8a9b2c3d4e5f6a7b8c9d0e",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"Berlin\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

## Tests

```bash
pytest test_normalizer.py -v
```

## Umgebungsvariablen

| Variable | Default | Beschreibung |
|----------|---------|--------------|
| `HOST` | `0.0.0.0` | Bind-Adresse (z.B. `127.0.0.1` für nur lokal) |
| `PORT` | `8000` | Port für den Proxy |
| `UPSTREAM_URL` | `http://localhost:8080/v1` | URL des LLM-Endpunkts |
| `UPSTREAM_API_KEY` | `""` | API-Key für Upstream (optional) |
| `DEFAULT_TOOL_MAX_TOKENS` | `8192` | Wird für Requests mit `tools` gesetzt, wenn der Client kein eigenes Output-Limit mitsendet |
| `TIMEOUT` | `120` | Request-Timeout in Sekunden |

## Endpoints

- `POST /v1/chat/completions` - Chat Completions (Hauptendpunkt)
- `GET /v1/models` - Models auflisten
- `GET /health` - Health-Check

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t minproxy .
docker run -p 8000:8000 -e UPSTREAM_URL=http://host.docker.internal:8080/v1 minproxy
```
