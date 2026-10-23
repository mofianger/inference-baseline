"""Read-only probe of pinned official GCS range endpoints."""
import json
from pathlib import Path
import time
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

manifest = json.loads(Path("/data/huggingface/openpi/pi05_base/official-source.json").read_text())
item = max(manifest["objects"], key=lambda x: int(x["size"]))
query = urlencode({"generation": item["generation"]})
urls = ["https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/" + quote(item["name"], safe="") + "?alt=media&" + query,
        "https://storage.googleapis.com/openpi-assets/" + item["name"] + "?" + query,
        "https://openpi-assets.storage.googleapis.com/" + item["name"] + "?" + query]
def probe(url):
    start = time.monotonic()
    try:
        with urlopen(Request(url, headers={"Range": "bytes=0-1048575"}), timeout=30) as response:
            data = response.read(1048576)
            print(json.dumps({"url": url, "status": response.status, "range": response.headers.get("Content-Range"),
                              "bytes": len(data), "seconds": time.monotonic() - start}), flush=True)
    except Exception as exc:
        print(json.dumps({"url": url, "error": str(exc)}), flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(probe, urls))
