"""Read-only remote probe; sent through SSH stdin, never installed remotely."""
import json
from pathlib import Path
import time

root = Path('/data/huggingface/openpi/pi05_base')
source = json.loads((root / 'official-source.json').read_text())
rows = []
for obj in source['objects']:
    name = obj['name'].removeprefix('checkpoints/pi05_base/')
    path = root / name
    size = int(obj['size'])
    final = path.is_file() and path.stat().st_size == size
    partial = path.with_name(path.name + '.partial')
    parts = path.with_name(path.name + '.parts')
    chunks = {}
    if not final and parts.is_dir():
        for chunk in parts.iterdir():
            offset = chunk.name.removesuffix('.tmp')
            if offset.isdigit() and chunk.is_file():
                start = int(offset)
                if start < size:
                    chunks[start] = max(chunks.get(start, 0), min(chunk.stat().st_size, 4*1024*1024, size-start))
    # Whole-file staging and chunk staging overlap. Never add them together.
    staged = max(sum(chunks.values()), partial.stat().st_size if partial.is_file() else 0)
    rows.append({'name': name, 'total': size, 'downloaded': size if final else min(size, staged),
                 'verified': final})
processes = []
for path in Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    try:
        args = (path / 'cmdline').read_bytes().split(b'\0')
        if any(x.endswith(b'/download_pi05_official.py') or x == b'scripts/download_pi05_official.py' for x in args):
            processes.append(int(path.name))
    except (OSError, PermissionError):
        pass
logs = list(Path('/data/inference-baseline/artifacts').glob('pi05-official-download*.log'))
latest = max(logs, key=lambda x: x.stat().st_mtime) if logs else None
tail = latest.read_text(errors='replace').splitlines()[-14:] if latest else []
complete = False
marker = root / 'verified-files.json'
if marker.is_file():
    manifest = json.loads(marker.read_text())
    complete = ({x['path'] for x in manifest} == {r['name'] for r in rows}
                and all(r['verified'] for r in rows))
status = 'complete' if complete else 'downloading' if processes else 'stopped'
if status == 'stopped' and any('Traceback' in x or 'Error:' in x for x in tail):
    status = 'error'
print(json.dumps({'status': status, 'remote_time': time.time(), 'total_bytes': sum(r['total'] for r in rows),
    'downloaded_bytes': sum(r['downloaded'] for r in rows),
    'verified_bytes': sum(r['total'] for r in rows if r['verified']),
    'verified_files': sum(r['verified'] for r in rows), 'total_files': len(rows),
    'processes': processes, 'log': latest.name if latest else None, 'log_tail': tail, 'files': rows}))
