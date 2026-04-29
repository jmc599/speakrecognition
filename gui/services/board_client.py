import json
import shlex
from pathlib import Path
from uuid import uuid4

import numpy as np
import paramiko

from gui.services.paths import local_tmp_dir


HTTP_CONNECT_TIMEOUT = 3
HTTP_HEALTH_TIMEOUT = (HTTP_CONNECT_TIMEOUT, 3)
HTTP_EMBED_TIMEOUT = (HTTP_CONNECT_TIMEOUT, 20)
HTTP_VERIFY_TIMEOUT = (HTTP_CONNECT_TIMEOUT, 20)
HTTP_PROFILE_TIMEOUT = (HTTP_CONNECT_TIMEOUT, 10)
DEFAULT_REMOTE_HTTP_PORT = 8765


class BoardClient:
    def __init__(self, config):
        self.config = config
        self._client = None
        self._http_session = None
        self._selected_transport = None
        self._transport_info = None

    def close(self):
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
        if self._http_session is not None:
            try:
                self._http_session.close()
            finally:
                self._http_session = None
        self._selected_transport = None
        self._transport_info = None

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def _connect(self):
        if self._client is not None:
            transport = self._client.get_transport()
            if transport is not None and transport.is_active():
                return self._client
            self.close()

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = {
            "hostname": self.config["board_host"],
            "port": int(self.config["board_port"]),
            "username": self.config["board_username"],
            "timeout": 10,
            "look_for_keys": True,
            "allow_agent": True,
        }
        password = self.config.get("board_password", "")
        if password:
            kwargs["password"] = password
        client.connect(**kwargs)
        self._client = client
        return self._client

    @staticmethod
    def _requests_module():
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "HTTP transport requires the 'requests' package. "
                "Install requirements_gui.txt to enable remote HTTP mode."
            ) from exc
        return requests

    def _http_session_client(self):
        requests = self._requests_module()
        if self._http_session is None:
            self._http_session = requests.Session()
        return self._http_session

    def _transport_preference(self):
        return str(self.config.get("remote_transport_preference", "http_first") or "http_first")

    def _resolved_http_base_url(self):
        explicit = str(self.config.get("remote_http_base_url", "") or "").strip()
        if explicit:
            return explicit.rstrip("/")
        board_host = str(self.config.get("board_host", "") or "").strip()
        return f"http://{board_host}:{DEFAULT_REMOTE_HTTP_PORT}".rstrip("/")

    def _run_command(self, client, command):
        stdin, stdout, stderr = client.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode("utf-8", errors="replace")
        stderr_text = stderr.read().decode("utf-8", errors="replace")
        return exit_status, stdout_text, stderr_text

    def _run_command_checked(self, client, command, label):
        exit_status, stdout_text, stderr_text = self._run_command(client, command)
        if exit_status != 0:
            raise RuntimeError(
                f"{label} failed.\n"
                f"stdout:\n{stdout_text}\n"
                f"stderr:\n{stderr_text}"
            )
        return stdout_text, stderr_text

    def _remote_runner_script(self):
        return (
            self.config.get("remote_runner_script")
            or self.config.get("remote_embed_script")
            or ""
        )

    def _remote_active_profile_path(self):
        return f"{self.config['remote_workdir'].rstrip('/')}/active_profile.npy"

    def _remote_tmp_profile_path(self):
        return f"{self.config['remote_workdir'].rstrip('/')}/active_profile.tmp.npy"

    def _remote_job_dir(self, prefix):
        return f"{self.config['remote_workdir'].rstrip('/')}/{prefix}_{uuid4().hex}"

    def ensure_remote_dir(self, remote_dir):
        client = self._connect()
        self._run_command_checked(
            client,
            f"mkdir -p {shlex.quote(remote_dir)}",
            "Remote mkdir",
        )
        return {"remote_dir": remote_dir}

    def upload_file(self, local_path, remote_path):
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        client = self._connect()
        parent = str(Path(remote_path).parent).replace("\\", "/")
        self.ensure_remote_dir(parent)
        sftp = client.open_sftp()
        try:
            sftp.put(str(local_path), remote_path)
        finally:
            sftp.close()
        return {"remote_path": remote_path}

    def remove_remote_path(self, remote_path, recursive=False):
        client = self._connect()
        command = (
            f"rm -rf {shlex.quote(remote_path)}"
            if recursive
            else f"rm -f {shlex.quote(remote_path)}"
        )
        self._run_command_checked(client, command, "Remote remove")
        return {"remote_path": remote_path}

    def _download_numpy(self, sftp, remote_path, local_path):
        sftp.get(remote_path, str(local_path))
        return np.load(local_path).astype(np.float32)

    @staticmethod
    def _parse_stdout_json(stdout_text):
        lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
        if not lines:
            return None
        candidate = lines[-1]
        if not (candidate.startswith("{") and candidate.endswith("}")):
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def _http_json(self, method, path, *, files=None, data=None, timeout=HTTP_VERIFY_TIMEOUT):
        session = self._http_session_client()
        requests = self._requests_module()
        url = f"{self._resolved_http_base_url()}{path}"
        try:
            response = session.request(
                method=method,
                url=url,
                files=files,
                data=data,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"HTTP {method} {path} request failed: {exc}") from exc
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.status_code >= 400:
            if isinstance(payload, dict):
                message = payload.get("error") or payload.get("message") or json.dumps(
                    payload, ensure_ascii=False
                )
            else:
                message = response.text
            raise RuntimeError(
                f"HTTP {method} {path} failed with status {response.status_code}: {message}"
            )
        if not isinstance(payload, dict):
            raise RuntimeError(f"HTTP {method} {path} returned invalid JSON.")
        return payload

    def _ssh_test_connection(self):
        client = self._connect()
        runner_script = self._remote_runner_script()
        runner_dir = str(Path(runner_script).parent).replace("\\", "/")
        import_check = (
            "import numpy; "
            "from rknnlite.api import RKNNLite; "
            "import utils.rknn_audio; "
            "print('remote_rknn_ready')"
        )
        version_check = "import sys; print(sys.version.split()[0])"
        mkdir_cmd = (
            f"mkdir -p {shlex.quote(self.config['remote_workdir'])} && "
            f"test -f {shlex.quote(self.config['remote_model_path'])} && "
            f"test -f {shlex.quote(runner_script)} && "
            f"cd {shlex.quote(runner_dir)} && "
            f"{shlex.quote(self.config['remote_python'])} -c {shlex.quote(import_check)} && "
            f"{shlex.quote(self.config['remote_python'])} -c {shlex.quote(version_check)}"
        )
        stdout_text, stderr_text = self._run_command_checked(
            client,
            mkdir_cmd,
            "Remote validation",
        )
        return {
            "mode": "remote_rknn",
            "transport": "ssh",
            "board_host": self.config["board_host"],
            "python_version": stdout_text.strip().splitlines()[-1]
            if stdout_text.strip()
            else stderr_text.strip(),
            "model_path": self.config["remote_model_path"],
            "runner_script": runner_script,
        }

    def _http_test_connection(self):
        payload = self._http_json(
            "GET",
            "/health",
            timeout=HTTP_HEALTH_TIMEOUT,
        )
        return {
            "mode": "remote_rknn",
            "transport": "http",
            "board_host": self.config["board_host"],
            "http_base_url": self._resolved_http_base_url(),
            "python_version": str(payload.get("python_version", "")),
            "model_path": str(payload.get("model_path", self.config["remote_model_path"])),
            "runner_script": str(payload.get("runner_script", self._remote_runner_script())),
            "model_fingerprint": str(payload.get("model_fingerprint", "")),
            "runtime_ready": bool(payload.get("runtime_ready", False)),
            "service_framework": str(payload.get("service_framework", "")),
        }

    def _ensure_transport_selected(self):
        if self._transport_info is not None:
            return self._transport_info

        preference = self._transport_preference()
        http_error = None

        if preference != "ssh_only":
            try:
                result = self._http_test_connection()
                self._selected_transport = "http"
                self._transport_info = result
                return result
            except Exception as exc:
                http_error = str(exc)

        try:
            result = self._ssh_test_connection()
            self._selected_transport = "ssh"
            if http_error:
                result["transport_fallback_reason"] = http_error
            self._transport_info = result
            return result
        except Exception as ssh_exc:
            if http_error:
                raise RuntimeError(
                    "HTTP and SSH transports both failed.\n"
                    f"HTTP error: {http_error}\n"
                    f"SSH error: {ssh_exc}"
                ) from ssh_exc
            raise

    def test_connection(self):
        return self._ensure_transport_selected()

    def _http_embed_audio(self, audio_path):
        local_output = Path(local_tmp_dir()) / f"embed_{uuid4().hex}_embedding.npy"
        with audio_path.open("rb") as f:
            payload = self._http_json(
                "POST",
                "/embed-audio",
                files={"audio": (audio_path.name, f, "audio/wav")},
                timeout=HTTP_EMBED_TIMEOUT,
            )
        embedding = np.asarray(payload["embedding"], dtype=np.float32).reshape(1, -1)
        np.save(local_output, embedding.astype(np.float32))
        return {
            "embedding_path": str(local_output),
            "embedding": embedding,
            "stdout": "",
            "stderr": "",
            "transport": "http",
        }

    def _ssh_embed_audio(self, audio_path, prefix="job", remote_dir=None):
        client = self._connect()
        remote_dir = remote_dir or self._remote_job_dir(prefix)
        remote_audio = f"{remote_dir}/audio.wav"
        remote_output = f"{remote_dir}/embedding.npy"
        local_output = Path(local_tmp_dir()) / f"{prefix}_{uuid4().hex}_embedding.npy"

        try:
            self._run_command_checked(client, f"mkdir -p {shlex.quote(remote_dir)}", "Remote mkdir")
            sftp = client.open_sftp()
            try:
                sftp.put(str(audio_path), remote_audio)
            finally:
                sftp.close()

            command = (
                f"{shlex.quote(self.config['remote_python'])} "
                f"{shlex.quote(self._remote_runner_script())} "
                f"embed-audio "
                f"--audio {shlex.quote(remote_audio)} "
                f"--model {shlex.quote(self.config['remote_model_path'])} "
                f"--output-embedding {shlex.quote(remote_output)}"
            )
            stdout_text, stderr_text = self._run_command_checked(
                client,
                command,
                "Remote audio embedding",
            )

            sftp = client.open_sftp()
            try:
                embedding = self._download_numpy(sftp, remote_output, local_output)
            finally:
                sftp.close()

            return {
                "embedding_path": str(local_output),
                "embedding": embedding,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "transport": "ssh",
            }
        finally:
            self._run_command(client, f"rm -rf {shlex.quote(remote_dir)}")

    def embed_audio(self, audio_path, prefix="job", remote_dir=None):
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        result = self._ensure_transport_selected()
        if result["transport"] == "http":
            return self._http_embed_audio(audio_path)
        return self._ssh_embed_audio(audio_path, prefix=prefix, remote_dir=remote_dir)

    def _http_verify_audio(self, audio_path, threshold, profile_path=None):
        local_output = Path(local_tmp_dir()) / f"verify_{uuid4().hex}_result.json"
        data = {"threshold": f"{float(threshold):.6f}"}
        if profile_path:
            data["profile_path"] = str(profile_path)
        with audio_path.open("rb") as f:
            payload = self._http_json(
                "POST",
                "/verify-audio",
                files={"audio": (audio_path.name, f, "audio/wav")},
                data=data,
                timeout=HTTP_VERIFY_TIMEOUT,
            )
        local_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "result_path": str(local_output),
            "score": float(payload["score"]),
            "threshold": float(payload["threshold"]),
            "passed": bool(payload["passed"]),
            "decision": str(payload["decision"]),
            "preprocess_for_inference": bool(payload.get("preprocess_for_inference", True)),
            "num_eval": int(payload.get("num_eval", 5)),
            "stdout": "",
            "stderr": "",
            "transport": "http",
        }

    def _ssh_verify_audio(self, audio_path, threshold, prefix="job", profile_path=None, remote_dir=None):
        client = self._connect()
        remote_dir = remote_dir or self._remote_job_dir(prefix)
        remote_audio = f"{remote_dir}/audio.wav"
        remote_output = f"{remote_dir}/result.json"
        local_output = Path(local_tmp_dir()) / f"{prefix}_{uuid4().hex}_result.json"
        profile_path = profile_path or self._remote_active_profile_path()

        try:
            self._run_command_checked(client, f"mkdir -p {shlex.quote(remote_dir)}", "Remote mkdir")
            sftp = client.open_sftp()
            try:
                sftp.put(str(audio_path), remote_audio)
            finally:
                sftp.close()

            command = (
                f"{shlex.quote(self.config['remote_python'])} "
                f"{shlex.quote(self._remote_runner_script())} "
                f"verify-audio "
                f"--audio {shlex.quote(remote_audio)} "
                f"--model {shlex.quote(self.config['remote_model_path'])} "
                f"--profile {shlex.quote(profile_path)} "
                f"--threshold {float(threshold):.6f} "
                f"--output-json {shlex.quote(remote_output)}"
            )
            stdout_text, stderr_text = self._run_command_checked(
                client,
                command,
                "Remote audio verification",
            )

            payload = self._parse_stdout_json(stdout_text)
            if payload is None:
                sftp = client.open_sftp()
                try:
                    sftp.get(remote_output, str(local_output))
                finally:
                    sftp.close()
                with local_output.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                local_output.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            return {
                "result_path": str(local_output),
                "score": float(payload["score"]),
                "threshold": float(payload["threshold"]),
                "passed": bool(payload["passed"]),
                "decision": str(payload["decision"]),
                "preprocess_for_inference": bool(payload.get("preprocess_for_inference", True)),
                "num_eval": int(payload.get("num_eval", 5)),
                "stdout": stdout_text,
                "stderr": stderr_text,
                "transport": "ssh",
            }
        finally:
            self._run_command(client, f"rm -rf {shlex.quote(remote_dir)}")

    def verify_audio(
        self,
        audio_path,
        threshold,
        prefix="job",
        profile_path=None,
        remote_dir=None,
    ):
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        result = self._ensure_transport_selected()
        if result["transport"] == "http":
            return self._http_verify_audio(audio_path, threshold=threshold, profile_path=profile_path)
        return self._ssh_verify_audio(
            audio_path,
            threshold=threshold,
            prefix=prefix,
            profile_path=profile_path,
            remote_dir=remote_dir,
        )

    def _http_push_active_profile(self, embedding_path):
        with embedding_path.open("rb") as f:
            payload = self._http_json(
                "POST",
                "/profile/push",
                files={"profile": (embedding_path.name, f, "application/octet-stream")},
                timeout=HTTP_PROFILE_TIMEOUT,
            )
        return {
            "active_profile_path": str(payload.get("active_profile_path", self._remote_active_profile_path())),
            "transport": "http",
        }

    def _ssh_push_active_profile(self, embedding_path):
        client = self._connect()
        remote_workdir = self.config["remote_workdir"]
        remote_tmp = self._remote_tmp_profile_path()
        remote_active = self._remote_active_profile_path()
        try:
            self._run_command_checked(client, f"mkdir -p {shlex.quote(remote_workdir)}", "Remote mkdir")
            sftp = client.open_sftp()
            try:
                sftp.put(str(embedding_path), remote_tmp)
            finally:
                sftp.close()
            self._run_command_checked(
                client,
                f"mv {shlex.quote(remote_tmp)} {shlex.quote(remote_active)}",
                "Remote profile publish",
            )
            return {"active_profile_path": remote_active, "transport": "ssh"}
        finally:
            self._run_command(client, f"rm -f {shlex.quote(remote_tmp)}")

    def push_active_profile(self, embedding_path):
        embedding_path = Path(embedding_path)
        if not embedding_path.is_file():
            raise FileNotFoundError(f"Embedding not found: {embedding_path}")

        result = self._ensure_transport_selected()
        if result["transport"] == "http":
            return self._http_push_active_profile(embedding_path)
        return self._ssh_push_active_profile(embedding_path)

    def _http_clear_active_profile(self):
        payload = self._http_json(
            "POST",
            "/profile/clear",
            timeout=HTTP_PROFILE_TIMEOUT,
        )
        return {
            "active_profile_path": str(payload.get("active_profile_path", self._remote_active_profile_path())),
            "transport": "http",
        }

    def _ssh_clear_active_profile(self):
        client = self._connect()
        self._run_command_checked(
            client,
            (
                f"rm -f {shlex.quote(self._remote_active_profile_path())} "
                f"{shlex.quote(self._remote_tmp_profile_path())}"
            ),
            "Remote profile clear",
        )
        return {"active_profile_path": self._remote_active_profile_path(), "transport": "ssh"}

    def clear_active_profile(self):
        result = self._ensure_transport_selected()
        if result["transport"] == "http":
            return self._http_clear_active_profile()
        return self._ssh_clear_active_profile()
