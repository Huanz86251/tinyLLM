"""Local-only chat server. No cloud inference or external UI resources."""
from pathlib import Path
import argparse, base64, binascii, io, json, logging, queue, re, threading, uuid
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory
from PIL import Image, UnidentifiedImageError
from inference.chat_engine import ChatEngine, BusyError, MODES
from project_paths import ROOT, path

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 16*1024*1024
engine = ChatEngine()
PORT = 8501

@app.before_request
def local_only():
    valid = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
    if request.host not in valid:
        abort(403)
    if request.method == "POST":
        origin = request.headers.get("Origin")
        if origin and origin not in {f"http://{h}" for h in valid}:
            abort(403)
        if not request.is_json:
            abort(415)

@app.after_request
def headers(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'"
    return response

@app.errorhandler(413)
def too_large(exc):
    return jsonify(error="图片或消息太大，请选择小于 10 MB 的图片"), 413

@app.route("/")
def index():
    return send_from_directory(path("ui"), "index.html")

@app.route("/assets/<name>")
def assets(name):
    if name not in {"app.js", "app.css", "logo.svg", "rendering.js"}:
        abort(404)
    return send_from_directory(path("ui"), name)

@app.route("/assets/vendor/katex/<path:filename>")
def katex_assets(filename):
    if filename not in {"katex.min.js","katex.min.css","LICENSE"} and not re.fullmatch(r"fonts/[A-Za-z0-9_-]+\.(?:woff2?|ttf)", filename):
        abort(404)
    return send_from_directory(path("ui/vendor/katex"), filename)

@app.route("/api/health")
@app.route("/_stcore/health")
def health():
    return jsonify(ok=True, app="tinyllm-chat", version=2)

@app.route("/api/status")
def status():
    return jsonify(engine.status())

@app.route("/api/model", methods=["POST"])
def choose_model():
    data = request.get_json()
    mode = data.get("mode")
    if mode not in MODES:
        return jsonify(error="未知模式"), 400
    try:
        engine.select(mode)
        return jsonify(engine.status())
    except BusyError as exc:
        return jsonify(error=str(exc)), 409
    except Exception as exc:
        logging.exception("Model selection failed")
        return jsonify(error=str(exc)), 500

@app.route("/api/unload", methods=["POST"])
def unload():
    try:
        engine.unload()
        return jsonify(engine.status())
    except BusyError as exc:
        return jsonify(error=str(exc)), 409

def image_path(image_id):
    if not isinstance(image_id, str) or not re.fullmatch(r"[0-9a-f]{32}\.(?:png|jpg|webp)", image_id):
        raise ValueError("图片编号无效")
    result = path("runs/chat_uploads") / image_id
    if not result.is_file():
        raise ValueError("原图片已不存在，请重新上传")
    return result

def save_image(raw):
    if len(raw) > 10*1024*1024:
        raise ValueError("图片请小于 10 MB")
    with Image.open(io.BytesIO(raw)) as im:
        if im.width*im.height > 20_000_000:
            raise ValueError("图片像素过多，请缩小后上传")
        extension = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}.get(im.format)
        if extension is None:
            raise ValueError("仅支持 PNG、JPEG 和 WebP 图片")
        im.verify()
    folder = path("runs/chat_uploads")
    folder.mkdir(parents=True, exist_ok=True)
    image_id = uuid.uuid4().hex + "." + extension
    (folder/image_id).write_bytes(raw)
    return {"image_id": image_id, "url": "/api/images/"+image_id}

@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json()
        encoded = data.get("data", "")
        if not isinstance(encoded, str) or len(encoded) > 14*1024*1024:
            raise ValueError("图片数据过大")
        raw = base64.b64decode(encoded, validate=True)
        return jsonify(save_image(raw))
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        return jsonify(error=str(exc) or "无法读取图片"), 400

@app.route("/api/example", methods=["POST"])
def example():
    name = request.get_json().get("name")
    file = {"cat": "chelsea.png", "astronaut": "astronaut.png"}.get(name)
    if not file:
        return jsonify(error="未知示例图片"), 400
    return jsonify(save_image(path("examples/images").joinpath(file).read_bytes()))

@app.route("/api/images/<image_id>")
def images(image_id):
    try:
        return send_file(image_path(image_id))
    except ValueError:
        abort(404)

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        request_id = str(uuid.UUID(data.get("request_id", "")))
        mode = data.get("mode", "chat")
        if mode not in MODES:
            raise ValueError("未知模式")
        limit = data.get("max_tokens", 2048)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 16 <= limit <= 2048:
            raise ValueError("生成长度需在 16–2048 之间")
        messages = data.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 40:
            raise ValueError("会话最多提交最近 40 条消息")
        clean = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
                raise ValueError("消息格式不正确")
            content = message.get("content")
            if not isinstance(content, str) or len(content) > 16000 or not content.strip():
                raise ValueError("消息为空或过长")
            clean.append({"role": message["role"], "content": content})
        if clean[-1]["role"] != "user":
            raise ValueError("最后一条消息必须是用户问题")
        image = image_path(data["image_id"]) if data.get("image_id") else None
        events = engine.start(request_id, mode, clean, image, limit)
    except BusyError as exc:
        return jsonify(error=str(exc)), 409
    except (ValueError, TypeError, AttributeError) as exc:
        return jsonify(error=str(exc)), 400

    def stream():
        completed = False
        try:
            while True:
                try:
                    event = events.get(timeout=1)
                except queue.Empty:
                    event = {"type": "heartbeat"}
                yield json.dumps(event, ensure_ascii=False)+"\n"
                if event["type"] in ("done", "error"):
                    completed = True
                    break
        finally:
            if not completed:
                engine.cancel(request_id)
    return Response(stream(), mimetype="application/x-ndjson", headers={"X-Accel-Buffering": "no"})

@app.route("/api/cancel", methods=["POST"])
def cancel():
    return jsonify(cancelled=engine.cancel(request.get_json().get("request_id")))

def main():
    global PORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()
    PORT = args.port
    if not args.no_preload:
        def warm():
            try:
                engine.select("chat")
            except Exception:
                logging.exception("SFT preload failed")
        threading.Thread(target=warm, daemon=True).start()
    # Local, single-user service with concurrent status/cancel endpoints.
    app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()

