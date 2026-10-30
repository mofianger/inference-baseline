"""Run on the Linux collector to canonicalize CUDA symbols with GNU c++filt."""
import json
import sqlite3
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
db = sqlite3.connect(root / 'reports/frontend-hw.sqlite')
symbols = db.execute('SELECT DISTINCT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.mangledName').fetchall()
names = [s[0] for s in symbols]
decoded = subprocess.run(['c++filt'], input='\n'.join(names)+'\n', text=True, capture_output=True, check=True).stdout.splitlines()
assert len(names) == len(decoded)
(root/'reports/symbols.json').write_text(json.dumps(dict(zip(names, decoded)), indent=2))
