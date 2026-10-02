"""Bounded synthetic inference through the installed admission gate; no user evidence."""
import base64
import json
import struct
import urllib.error
import urllib.request
import zlib

from hermes_memory.config import load_settings


def synthetic_red_image():
    def chunk(kind, data):
        return (struct.pack("!I", len(data)) + kind + data
                + struct.pack("!I", zlib.crc32(kind + data)))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack("!2I5B", 64, 64, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 64) * 64))
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main():
    settings = load_settings()
    results = []
    for name, route in (("foreground", settings.text_route),
                        ("vision", settings.vision_route),
                        ("embeddings", settings.embeddings_route)):
        if route is None or name not in settings.route_credentials:
            results.append({"route": name, "ok": False, "error": "route not configured"})
            print(json.dumps(results[-1]), flush=True)
            continue
        embedding = name == "embeddings"
        payload = {"model": route.model}
        if embedding:
            payload["input"] = "Synthetic Hermes memory connectivity test."
        else:
            payload.update(messages=[{"role": "user", "content": "Reply with OK only."}],
                           max_tokens=64, stream=False,
                           chat_template_kwargs={"enable_thinking": False})
            if name == "vision":
                payload["messages"][0]["content"] = [
                    {"type": "text", "text": "What color is this square? Answer one color word."},
                    {"type": "image_url", "image_url": {"url": synthetic_red_image()}},
                ]
        path = "/v1/embeddings" if embedding else "/v1/chat/completions"
        request = urllib.request.Request(
            settings.admission_url.rstrip("/") + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + settings.route_credentials[name]})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.load(response)
            if embedding:
                vectors = body.get("data", [])
                ok = bool(vectors and vectors[0].get("embedding"))
                detail = {"dimensions": len(vectors[0]["embedding"]) if ok else 0}
            else:
                choices = body.get("choices", [])
                message = choices[0].get("message", {}) if choices else {}
                ok = bool(message.get("content"))
                detail = {"output": str(message.get("content") or "")[:120]}
                if name == "vision":
                    ok = "red" in detail["output"].lower()
            results.append({"route": name, "ok": ok, **detail})
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            detail = error.read().decode()[:300] if isinstance(error, urllib.error.HTTPError) else str(error)
            results.append({"route": name, "ok": False, "error": detail})
        print(json.dumps(results[-1]), flush=True)
    return 0 if all(item["ok"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
