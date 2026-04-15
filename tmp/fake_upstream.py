import json
from http.server import BaseHTTPRequestHandler, HTTPServer


RESPONSE = {
    "id": "chatcmpl-fake",
    "object": "chat.completion",
    "created": 1776273200,
    "model": "fake-upstream",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": (
                "<function_call>\n"
                "<tool_use>\n"
                "<name>terminal</name>\n"
                "<parameters>\n"
                "<parameter name=\"command\">"
                "echo \"=== TEST 1: Einfache Command ===\" && date && whoami"
                "</parameter>\n"
                "</parameters>\n"
                "</tool_use>\n"
                "</parameters>\n"
                "</function_call>"
            ),
        },
        "finish_reason": "stop",
    }],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    },
}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        data = json.dumps(RESPONSE).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8099), Handler).serve_forever()