"""Loopback dashboard + separate SSH polling process posting to /api/update."""
import argparse
from collections import deque
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent


def worker(port, interval):
    token = os.environ['DOWNLOAD_MONITOR_TOKEN']
    source = (HERE / 'probe.py').read_text(encoding='utf-8')
    while True:
        try:
            result = subprocess.run(['ssh', '-o', 'ClearAllForwardings=yes', '-o', 'BatchMode=yes',
                '-o', 'ConnectTimeout=10', '5090-machine', 'python3 -'], input=source,
                capture_output=True, text=True, encoding='utf-8', timeout=45,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if result.returncode:
                raise RuntimeError(result.stderr.strip()[-1500:] or f'SSH exit {result.returncode}')
            data = json.loads(result.stdout)
            data['checked_at'] = time.time()
        except Exception as exc:
            data = {'status': 'connection_error', 'checked_at': time.time(), 'error': str(exc)}
        try:
            request = Request(f'http://127.0.0.1:{port}/api/update', data=json.dumps(data).encode(),
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token}, method='POST')
            with urlopen(request, timeout=5) as response:
                response.read()
        except Exception:
            return  # Parent is gone; do not leave an orphan polling process.
        if data['status'] == 'complete':
            return
        time.sleep(interval)


def serve(port, interval):
    state = {'status': 'starting', 'checked_at': None, 'interval': interval}
    lock = threading.Lock()
    history = deque(maxlen=30)
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, content_type):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                self.send(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/api/status':
                with lock:
                    data = dict(state)
                self.send(200, json.dumps(data).encode(), 'application/json')
            else:
                self.send(404, b'Not found', 'text/plain')

        def do_POST(self):
            if self.path != '/api/update' or not secrets.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+token):
                self.send(403, b'Forbidden', 'text/plain')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length < 262144:
                    raise ValueError('Invalid body size')
                data = json.loads(self.rfile.read(length))
                with lock:
                    if 'downloaded_bytes' in data:
                        history.append((data['checked_at'], data['downloaded_bytes']))
                        if len(history) >= 2:
                            first = history[0]
                            elapsed = data['checked_at'] - first[0]
                            speed = max(0, (data['downloaded_bytes'] - first[1]) / max(elapsed, 1))
                            data['bytes_per_second'] = speed
                            data['eta_seconds'] = ((data['total_bytes'] - data['downloaded_bytes']) / speed
                                                   if speed > 0 and elapsed >= interval else None)
                        data['last_success_at'] = data['checked_at']
                        state.pop('error', None)
                    else:
                        data['bytes_per_second'] = None
                        data['eta_seconds'] = None
                    state.update(data)
                self.send(200, b'{}', 'application/json')
            except (ValueError, KeyError, TypeError):
                self.send(400, b'Invalid update', 'text/plain')
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    env = {**os.environ, 'DOWNLOAD_MONITOR_TOKEN': token}
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', '--port', str(port),
                              '--interval', str(interval)], env=env,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    print(f'Download monitor: http://localhost:{port} (server PID {os.getpid()}, worker PID {child.pid})', flush=True)
    try:
        server.serve_forever()
    finally:
        child.terminate()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if not 5 <= args.interval <= 3600:
        parser.error('interval must be between 5 and 3600 seconds')
    (worker if args.worker else serve)(args.port, args.interval)
