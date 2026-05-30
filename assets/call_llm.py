#!/usr/bin/env python3
"""External LLM operator — reads a JSON context payload from stdin, calls the
browser-router's OpenAI-compatible endpoint, and writes the response to stdout.

Exit codes:
  0   success (response text on stdout)
  1   API / HTTP error
  127 connection failure (browser-router not reachable)
"""

import sys
import json
import argparse

BROWSER_ROUTER_URL = "http://127.0.0.1:8082/v1/chat/completions"
MODEL = "chatgpt-cdp"

PROMPT_TEMPLATES = {
    "plan": (
        "You are a neuro-symbolic planning engine embedded in a quantale-matrix agent loop.\n"
        "Given the following execution context:\n\n{context}\n\n"
        "Generate a concise, structured execution plan as JSON with keys: "
        "'steps' (list of action strings) and 'goal' (single sentence)."
    ),
    "repair": (
        "You are a repair subsystem for a quantale-matrix agent loop.\n"
        "The system encountered an error. Execution context:\n\n{context}\n\n"
        "Provide a structural repair directive as JSON with keys: "
        "'action' (one of: retry, rollback, halt) and 'reason' (single sentence)."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantale LLM operator")
    parser.add_argument("--template", required=True, choices=list(PROMPT_TEMPLATES))
    args = parser.parse_args()

    try:
        input_data = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"stdin JSON parse error: {exc}\n")
        sys.exit(1)

    context = input_data.get("context", "")
    prompt = PROMPT_TEMPLATES[args.template].format(context=context)

    try:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(
            BROWSER_ROUTER_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            body = json.loads(response.read().decode())
            content = body["choices"][0]["message"]["content"]
            print(content)
            sys.exit(0)

    except urllib.error.HTTPError as exc:
        sys.stderr.write(f"LLM API HTTP error {exc.code}: {exc.reason}\n")
        sys.exit(1)
    except urllib.error.URLError as exc:
        sys.stderr.write(f"browser-router connection failed: {exc.reason}\n")
        sys.exit(127)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"unexpected response shape: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
