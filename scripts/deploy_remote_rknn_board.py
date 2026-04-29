import argparse
import json
import posixpath
from pathlib import Path
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import paramiko

from gui.services.config_store import ConfigStore


DEFAULT_UPLOADS = {
    'rknn_remote_runner.py': 'rknn_remote_runner.py',
    'rknn_remote_http_server.py': 'rknn_remote_http_server.py',
    'utils/rknn_audio.py': 'utils/rknn_audio.py',
    'utils/inference_audio.py': 'utils/inference_audio.py',
    'utils/legacy_feature.py': 'utils/legacy_feature.py',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Deploy Remote RKNN runner files to the board and optionally pin a static eth1 IP.',
    )
    parser.add_argument('--board-host', type=str, default='')
    parser.add_argument('--board-port', type=int, default=0)
    parser.add_argument('--board-username', type=str, default='')
    parser.add_argument('--board-password', type=str, default='')
    parser.add_argument('--remote-runner-script', type=str, default='')
    parser.add_argument('--remote-workdir', type=str, default='')
    parser.add_argument('--remote-python', type=str, default='')
    parser.add_argument('--remote-model-path', type=str, default='')
    parser.add_argument('--remote-http-base-url', type=str, default='')
    parser.add_argument('--http-bind', type=str, default='0.0.0.0')
    parser.add_argument('--http-port', type=int, default=8765)
    parser.add_argument('--start-http-service', action='store_true')
    parser.add_argument('--local-root', type=str, default='.')
    parser.add_argument('--configure-static-ip', action='store_true')
    parser.add_argument('--apply-network-now', action='store_true')
    parser.add_argument('--disable-connman', action='store_true')
    parser.add_argument('--eth-iface', type=str, default='eth1')
    parser.add_argument('--board-ip', type=str, default='192.168.50.2')
    parser.add_argument('--prefix-len', type=int, default=24)
    return parser.parse_args()


def merged_config(args):
    cfg = ConfigStore().load()
    if args.board_host:
        cfg['board_host'] = args.board_host
    if args.board_port:
        cfg['board_port'] = args.board_port
    if args.board_username:
        cfg['board_username'] = args.board_username
    if args.board_password:
        cfg['board_password'] = args.board_password
    if args.remote_runner_script:
        cfg['remote_runner_script'] = args.remote_runner_script
        cfg['remote_embed_script'] = args.remote_runner_script
    if args.remote_workdir:
        cfg['remote_workdir'] = args.remote_workdir
    if args.remote_python:
        cfg['remote_python'] = args.remote_python
    if args.remote_model_path:
        cfg['remote_model_path'] = args.remote_model_path
    if args.remote_http_base_url:
        cfg['remote_http_base_url'] = args.remote_http_base_url
    return cfg


def connect(cfg):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs = {
        'hostname': cfg['board_host'],
        'port': int(cfg['board_port']),
        'username': cfg['board_username'],
        'look_for_keys': True,
        'allow_agent': True,
        'timeout': 10,
    }
    if cfg.get('board_password'):
        kwargs['password'] = cfg['board_password']
    client.connect(**kwargs)
    return client


def run_checked(client, command, label):
    stdin, stdout, stderr = client.exec_command(command)
    status = stdout.channel.recv_exit_status()
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if status != 0:
        raise RuntimeError(f'{label} failed.\nstdout:\n{out}\nstderr:\n{err}')
    return out, err


def remote_utils_dir(remote_runner_script):
    return posixpath.join(posixpath.dirname(remote_runner_script), 'utils')


def upload_files(client, cfg, local_root):
    sftp = client.open_sftp()
    try:
        runner_dir = posixpath.dirname(cfg['remote_runner_script'])
        utils_dir = remote_utils_dir(cfg['remote_runner_script'])
        run_checked(client, f"mkdir -p {runner_dir} {utils_dir} {cfg['remote_workdir']}", 'mkdir')

        local_root = Path(local_root)
        uploaded = []
        for rel_local, rel_remote in DEFAULT_UPLOADS.items():
            local_path = local_root / rel_local
            if not local_path.is_file():
                raise FileNotFoundError(f'Local file not found: {local_path}')
            if rel_remote.startswith('utils/'):
                remote_path = posixpath.join(utils_dir, Path(rel_remote).name)
            else:
                remote_path = posixpath.join(runner_dir, Path(rel_remote).name)
            sftp.put(str(local_path), remote_path)
            uploaded.append({'local': str(local_path), 'remote': remote_path})

        manifest = {
            'remote_runner_script': cfg['remote_runner_script'],
            'remote_workdir': cfg['remote_workdir'],
            'remote_model_path': cfg['remote_model_path'],
            'uploads': uploaded,
        }
        manifest_path = posixpath.join(cfg['remote_workdir'], 'remote_rknn_deploy_manifest.json')
        with sftp.open(manifest_path, 'w') as f:
            f.write(json.dumps(manifest, ensure_ascii=False, indent=2))
        return uploaded, manifest_path
    finally:
        sftp.close()


def configure_static_ip(client, iface, board_ip, prefix_len):
    interfaces_text = (
        '# interface file auto-generated by buildroot\n\n'
        'auto lo\n'
        'iface lo inet loopback\n\n'
        f'auto {iface}\n'
        f'iface {iface} inet static\n'
        f'    address {board_ip}\n'
        '    netmask 255.255.255.0\n'
    )
    late_script = (
        '#!/bin/sh\n'
        'case "$1" in\n'
        '  start)\n'
        f'    ip link set {iface} up\n'
        f'    ip addr flush dev {iface}\n'
        f'    ip addr add {board_ip}/{prefix_len} dev {iface}\n'
        '    ;;\n'
        '  stop)\n'
        '    ;;\n'
        '  restart|reload)\n'
        '    "$0" stop\n'
        '    "$0" start\n'
        '    ;;\n'
        '  *)\n'
        '    echo "Usage: $0 {start|stop|restart|reload}"\n'
        '    exit 1\n'
        '    ;;\n'
        'esac\n\n'
        'exit 0\n'
    )

    backup_cmd = (
        'test -f /etc/network/interfaces.codex.bak || '
        'cp /etc/network/interfaces /etc/network/interfaces.codex.bak 2>/dev/null || true'
    )
    run_checked(client, backup_cmd, 'backup interfaces')

    sftp = client.open_sftp()
    try:
        with sftp.open('/etc/network/interfaces', 'w') as f:
            f.write(interfaces_text)
        with sftp.open('/etc/init.d/S41codex_eth_static', 'w') as f:
            f.write(late_script)
    finally:
        sftp.close()

    run_checked(client, 'chmod +x /etc/init.d/S41codex_eth_static', 'chmod static ip script')


def resolved_http_base_url(cfg, http_port):
    explicit = str(cfg.get('remote_http_base_url', '') or '').strip()
    if explicit:
        return explicit.rstrip('/')
    return f"http://{cfg['board_host']}:{int(http_port)}"


def configure_http_service(client, cfg, http_bind, http_port):
    runner_dir = posixpath.dirname(cfg['remote_runner_script'])
    server_script = posixpath.join(runner_dir, 'rknn_remote_http_server.py')
    pid_file = '/var/run/rknn_remote_http.pid'
    log_file = posixpath.join(cfg['remote_workdir'], 'rknn_remote_http.log')
    start_script = posixpath.join(runner_dir, 'start_rknn_remote_http_server.sh')
    stop_script = posixpath.join(runner_dir, 'stop_rknn_remote_http_server.sh')
    init_script_path = '/etc/init.d/S52rknn_remote_http'

    start_script_text = f"""#!/bin/sh
PID_FILE="{pid_file}"
LOG_FILE="{log_file}"
mkdir -p "{cfg['remote_workdir']}" /var/run
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  exit 0
fi
nohup {cfg['remote_python']} "{server_script}" --model "{cfg['remote_model_path']}" --workdir "{cfg['remote_workdir']}" --host "{http_bind}" --port "{int(http_port)}" >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
"""

    stop_script_text = f"""#!/bin/sh
PID_FILE="{pid_file}"
if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE")"
  kill "$PID" 2>/dev/null || true
  rm -f "$PID_FILE"
fi
"""

    init_script_text = f"""#!/bin/sh
case "$1" in
  start)
    "{start_script}"
    ;;
  stop)
    "{stop_script}"
    ;;
  restart|reload)
    "{stop_script}"
    sleep 1
    "{start_script}"
    ;;
  *)
    echo "Usage: $0 {{start|stop|restart|reload}}"
    exit 1
    ;;
esac

exit 0
"""

    sftp = client.open_sftp()
    try:
        with sftp.open(start_script, 'w') as f:
            f.write(start_script_text)
        with sftp.open(stop_script, 'w') as f:
            f.write(stop_script_text)
        with sftp.open(init_script_path, 'w') as f:
            f.write(init_script_text)
    finally:
        sftp.close()

    run_checked(client, f'chmod +x "{start_script}" "{stop_script}" "{init_script_path}"', 'chmod http service scripts')
    return {
        'server_script': server_script,
        'start_script': start_script,
        'stop_script': stop_script,
        'init_script': init_script_path,
        'http_base_url': resolved_http_base_url(cfg, http_port),
    }


def disable_connman(client):
    disable_cmd = (
        'if [ -f /etc/init.d/S45connman ]; then '
        'mv /etc/init.d/S45connman /etc/init.d/connman.disabled; '
        'fi; '
        'pkill connmand 2>/dev/null || true; '
        'pkill connmanctl 2>/dev/null || true'
    )
    run_checked(client, disable_cmd, 'disable connman')


def validate_remote(client, cfg):
    runner_dir = posixpath.dirname(cfg['remote_runner_script'])
    import_check = (
        'import numpy; '
        'from rknnlite.api import RKNNLite; '
        'import utils.rknn_audio; '
        'print("remote_rknn_ready")'
    )
    version_check = 'import sys; print(sys.version.split()[0])'
    cmd = (
        f"test -f {cfg['remote_model_path']} && "
        f"test -f {cfg['remote_runner_script']} && "
        f"cd {runner_dir} && "
        f"{cfg['remote_python']} -c {json.dumps(import_check)} && "
        f"{cfg['remote_python']} -c {json.dumps(version_check)}"
    )
    out, _ = run_checked(client, cmd, 'remote validation')
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return lines[-1] if lines else ''


def validate_http_health(base_url, timeout=5, retries=5):
    url = f"{base_url.rstrip('/')}/health"
    last_error = None
    for _ in range(retries):
        try:
            with urlopen(url, timeout=timeout) as response:
                body = response.read().decode('utf-8', errors='replace')
            payload = json.loads(body)
            if payload.get('status') != 'ok':
                raise RuntimeError(f'HTTP health returned unexpected payload: {payload}')
            return payload
        except (URLError, TimeoutError, ValueError, RuntimeError) as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f'HTTP health validation failed for {url}: {last_error}')


def main():
    args = parse_args()
    cfg = merged_config(args)
    client = connect(cfg)
    try:
        uploads, manifest_path = upload_files(client, cfg, args.local_root)
        print(json.dumps({'uploads': uploads, 'manifest_path': manifest_path}, ensure_ascii=False, indent=2))

        http_service_info = configure_http_service(client, cfg, args.http_bind, args.http_port)
        print(json.dumps(http_service_info, ensure_ascii=False, indent=2))

        if args.configure_static_ip:
            configure_static_ip(client, args.eth_iface, args.board_ip, args.prefix_len)
            print(f'static_ip_configured={args.eth_iface}:{args.board_ip}/{args.prefix_len}')
        if args.disable_connman:
            disable_connman(client)
            print('connman_disabled=true')
            if args.apply_network_now:
                run_checked(client, '/etc/init.d/S40network restart || true', 'restart network')
                run_checked(client, '/etc/init.d/S41codex_eth_static start', 'apply static ip')
                out, _ = run_checked(client, f'ip addr show {args.eth_iface}', 'show iface')
                print(out)

        if args.start_http_service:
            run_checked(client, '/etc/init.d/S52rknn_remote_http restart', 'restart http service')
            health_payload = validate_http_health(http_service_info['http_base_url'])
            print(json.dumps({'http_health': health_payload}, ensure_ascii=False, indent=2))

        python_version = validate_remote(client, cfg)
        print(json.dumps({
            'board_host': cfg['board_host'],
            'remote_runner_script': cfg['remote_runner_script'],
            'remote_http_base_url': http_service_info['http_base_url'],
            'remote_model_path': cfg['remote_model_path'],
            'python_version': python_version,
            'status': 'ok',
        }, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    main()
