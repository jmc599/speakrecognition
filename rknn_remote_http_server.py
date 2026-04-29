import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from utils.rknn_audio import cosine_score, create_runtime, extract_embedding


RUNNER_VERSION = "remote_rknn_http_v2"
DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 8765


def parse_args():
    parser = argparse.ArgumentParser(
        description="Persistent HTTP service for board-side RKNN audio verification."
    )
    parser.add_argument("--model", required=True, help="Path to the RKNN model.")
    parser.add_argument("--workdir", required=True, help="Persistent workdir on the board.")
    parser.add_argument("--host", default=DEFAULT_HTTP_HOST, help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT, help="HTTP bind port.")
    parser.add_argument("--target", default="", help="Optional RKNN target.")
    parser.add_argument(
        "--force-stdlib",
        action="store_true",
        help="Force stdlib http.server even if Flask is available.",
    )
    return parser.parse_args()


def _model_fingerprint(model_path):
    model_path = Path(model_path)
    stat = model_path.stat()
    signature = (
        f"path={model_path.resolve()}|size={int(stat.st_size)}|"
        f"mtime_ns={int(getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9)))}"
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def _bool_from_disable_flag(disable_flag):
    return not bool(disable_flag)


class RemoteRknnHttpService:
    def __init__(self, model_path, workdir, target=""):
        self.model_path = Path(model_path).resolve()
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.active_profile_path = self.workdir / "active_profile.npy"
        self.tmp_profile_path = self.workdir / "active_profile.tmp.npy"
        self.model_fingerprint = _model_fingerprint(self.model_path)
        self._lock = threading.Lock()
        self._runtime = create_runtime(
            self.model_path,
            verbose=False,
            target=target or None,
        )

    def close(self):
        if self._runtime is not None:
            try:
                self._runtime.release()
            finally:
                self._runtime = None

    @contextmanager
    def request_dir(self, prefix):
        path = Path(
            tempfile.mkdtemp(
                prefix=f"{prefix}_",
                dir=str(self.workdir),
            )
        )
        try:
            yield path
        finally:
            try:
                for child in path.iterdir():
                    child.unlink()
                path.rmdir()
            except Exception:
                pass

    def health_payload(self):
        return {
            "status": "ok",
            "python_version": sys.version.split()[0],
            "runner_version": RUNNER_VERSION,
            "model_path": str(self.model_path),
            "model_fingerprint": self.model_fingerprint,
            "runtime_ready": self._runtime is not None,
            "workdir": str(self.workdir),
            "runner_script": str(Path(__file__).resolve()),
            "service_framework": "",
        }

    def _embed_from_audio_path(self, audio_path, *, num_eval=5, preprocess_for_inference=True):
        with self._lock:
            embedding = extract_embedding(
                self._runtime,
                str(audio_path),
                num_eval=int(num_eval),
                preprocess_for_inference=bool(preprocess_for_inference),
            )
        return np.asarray(embedding, dtype=np.float32).reshape(1, -1)

    def embed_audio_file(self, audio_path, *, num_eval=5, preprocess_for_inference=True):
        embedding = self._embed_from_audio_path(
            audio_path,
            num_eval=num_eval,
            preprocess_for_inference=preprocess_for_inference,
        )
        return {
            "embedding": embedding.reshape(-1).tolist(),
            "shape": list(embedding.shape),
            "preprocess_for_inference": bool(preprocess_for_inference),
            "num_eval": int(num_eval),
        }

    def verify_audio_file(
        self,
        audio_path,
        *,
        threshold,
        profile_path=None,
        num_eval=5,
        preprocess_for_inference=True,
    ):
        profile_candidate = Path(profile_path).resolve() if profile_path else self.active_profile_path
        if not profile_candidate.is_file():
            raise FileNotFoundError(f"Profile not found: {profile_candidate}")

        profile = np.load(profile_candidate).astype(np.float32).reshape(1, -1)
        embedding = self._embed_from_audio_path(
            audio_path,
            num_eval=num_eval,
            preprocess_for_inference=preprocess_for_inference,
        )
        score = cosine_score(embedding, profile)
        decision = "accept" if float(score) >= float(threshold) else "reject"
        return {
            "score": float(score),
            "threshold": float(threshold),
            "passed": decision == "accept",
            "decision": decision,
            "preprocess_for_inference": bool(preprocess_for_inference),
            "num_eval": int(num_eval),
        }

    def publish_profile_file(self, profile_file):
        with self._lock:
            profile = np.load(profile_file).astype(np.float32).reshape(1, -1)
            np.save(self.tmp_profile_path, profile.astype(np.float32))
            np.load(self.tmp_profile_path)
            self.tmp_profile_path.replace(self.active_profile_path)
        return {"active_profile_path": str(self.active_profile_path)}

    def clear_profile(self):
        with self._lock:
            if self.active_profile_path.exists():
                self.active_profile_path.unlink()
            if self.tmp_profile_path.exists():
                self.tmp_profile_path.unlink()
        return {"active_profile_path": str(self.active_profile_path)}


def _json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _parse_int(value, default):
    if value in (None, ""):
        return int(default)
    return int(value)


def _parse_bool_flag(value, default=False):
    if value in (None, ""):
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _stdlib_parse_multipart(handler):
    import cgi  # deprecated but still available on the target Python used here

    environ = {
        "REQUEST_METHOD": handler.command,
        "CONTENT_TYPE": handler.headers.get("Content-Type", ""),
        "CONTENT_LENGTH": handler.headers.get("Content-Length", "0"),
    }
    return cgi.FieldStorage(
        fp=handler.rfile,
        headers=handler.headers,
        environ=environ,
        keep_blank_values=True,
    )


def _save_uploaded_file(file_item, destination):
    with destination.open("wb") as f:
        while True:
            chunk = file_item.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return destination


def run_with_flask(service, host, port):
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.get("/health")
    def health():
        payload = service.health_payload()
        payload["service_framework"] = "flask"
        return jsonify(payload)

    @app.post("/embed-audio")
    def embed_audio():
        if "audio" not in request.files:
            return jsonify({"error": "Missing multipart file field: audio"}), 400
        num_eval = _parse_int(request.form.get("num_eval"), 5)
        preprocess = _bool_from_disable_flag(
            _parse_bool_flag(request.form.get("disable_inference_preprocess"), False)
        )
        with service.request_dir("http_embed") as request_dir:
            audio_path = request_dir / "audio.wav"
            request.files["audio"].save(str(audio_path))
            return jsonify(
                service.embed_audio_file(
                    audio_path,
                    num_eval=num_eval,
                    preprocess_for_inference=preprocess,
                )
            )

    @app.post("/verify-audio")
    def verify_audio():
        if "audio" not in request.files:
            return jsonify({"error": "Missing multipart file field: audio"}), 400
        threshold = request.form.get("threshold")
        if threshold in (None, ""):
            return jsonify({"error": "Missing threshold"}), 400
        num_eval = _parse_int(request.form.get("num_eval"), 5)
        preprocess = _bool_from_disable_flag(
            _parse_bool_flag(request.form.get("disable_inference_preprocess"), False)
        )
        profile_path = request.form.get("profile_path") or None
        with service.request_dir("http_verify") as request_dir:
            audio_path = request_dir / "audio.wav"
            request.files["audio"].save(str(audio_path))
            return jsonify(
                service.verify_audio_file(
                    audio_path,
                    threshold=float(threshold),
                    profile_path=profile_path,
                    num_eval=num_eval,
                    preprocess_for_inference=preprocess,
                )
            )

    @app.post("/profile/push")
    def profile_push():
        if "profile" not in request.files:
            return jsonify({"error": "Missing multipart file field: profile"}), 400
        with service.request_dir("http_profile") as request_dir:
            profile_path = request_dir / "profile.npy"
            request.files["profile"].save(str(profile_path))
            return jsonify(service.publish_profile_file(profile_path))

    @app.post("/profile/clear")
    def profile_clear():
        return jsonify(service.clear_profile())

    @app.errorhandler(Exception)
    def handle_error(exc):  # pragma: no cover - runtime path
        return jsonify({"error": str(exc)}), 500

    app.run(host=host, port=int(port), debug=False, threaded=False)


def run_with_stdlib(service, host, port):
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        server_version = "RemoteRknnHTTP/1.0"

        def _write_json(self, payload, status=200):
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            try:
                if self.path == "/health":
                    payload = service.health_payload()
                    payload["service_framework"] = "stdlib"
                    self._write_json(payload, status=200)
                    return
                self._write_json({"error": f"Unsupported path: {self.path}"}, status=404)
            except Exception as exc:  # pragma: no cover - runtime path
                self._write_json({"error": str(exc)}, status=500)

        def do_POST(self):  # noqa: N802
            try:
                if self.path == "/profile/clear":
                    self._write_json(service.clear_profile(), status=200)
                    return

                form = _stdlib_parse_multipart(self)
                if self.path == "/embed-audio":
                    if "audio" not in form:
                        self._write_json({"error": "Missing multipart file field: audio"}, status=400)
                        return
                    num_eval = _parse_int(form.getfirst("num_eval"), 5)
                    preprocess = _bool_from_disable_flag(
                        _parse_bool_flag(form.getfirst("disable_inference_preprocess"), False)
                    )
                    with service.request_dir("http_embed") as request_dir:
                        audio_path = request_dir / "audio.wav"
                        _save_uploaded_file(form["audio"], audio_path)
                        self._write_json(
                            service.embed_audio_file(
                                audio_path,
                                num_eval=num_eval,
                                preprocess_for_inference=preprocess,
                            ),
                            status=200,
                        )
                        return

                if self.path == "/verify-audio":
                    if "audio" not in form:
                        self._write_json({"error": "Missing multipart file field: audio"}, status=400)
                        return
                    threshold = form.getfirst("threshold")
                    if threshold in (None, ""):
                        self._write_json({"error": "Missing threshold"}, status=400)
                        return
                    num_eval = _parse_int(form.getfirst("num_eval"), 5)
                    preprocess = _bool_from_disable_flag(
                        _parse_bool_flag(form.getfirst("disable_inference_preprocess"), False)
                    )
                    profile_path = form.getfirst("profile_path") or None
                    with service.request_dir("http_verify") as request_dir:
                        audio_path = request_dir / "audio.wav"
                        _save_uploaded_file(form["audio"], audio_path)
                        self._write_json(
                            service.verify_audio_file(
                                audio_path,
                                threshold=float(threshold),
                                profile_path=profile_path,
                                num_eval=num_eval,
                                preprocess_for_inference=preprocess,
                            ),
                            status=200,
                        )
                        return

                if self.path == "/profile/push":
                    if "profile" not in form:
                        self._write_json({"error": "Missing multipart file field: profile"}, status=400)
                        return
                    with service.request_dir("http_profile") as request_dir:
                        profile_path = request_dir / "profile.npy"
                        _save_uploaded_file(form["profile"], profile_path)
                        self._write_json(service.publish_profile_file(profile_path), status=200)
                        return

                self._write_json({"error": f"Unsupported path: {self.path}"}, status=404)
            except Exception as exc:  # pragma: no cover - runtime path
                self._write_json({"error": str(exc)}, status=500)

        def log_message(self, format, *args):  # noqa: A003
            return

    server = HTTPServer((host, int(port)), Handler)
    server.serve_forever()


def main():
    args = parse_args()
    service = RemoteRknnHttpService(
        model_path=args.model,
        workdir=args.workdir,
        target=args.target,
    )
    try:
        if args.force_stdlib:
            run_with_stdlib(service, args.host, args.port)
            return
        try:
            import flask  # noqa: F401
        except ImportError:
            run_with_stdlib(service, args.host, args.port)
        else:
            run_with_flask(service, args.host, args.port)
    finally:
        service.close()


if __name__ == "__main__":
    main()
