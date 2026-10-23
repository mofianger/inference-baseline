"""Download the official OpenPI base checkpoint with pinned GCS generations."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
from pathlib import Path
import time
import google_crc32c
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_base/"


def digests(path):
    md5, sha, crc = hashlib.md5(), hashlib.sha256(), google_crc32c.Checksum()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
            crc.update(block)
    return base64.b64encode(md5.digest()).decode(), sha.hexdigest(), base64.b64encode(crc.digest()).decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--chunk-workers", type=int, default=8)
    args = p.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "official-source.json"
    if source.exists():
        manifest = json.loads(source.read_text())
    else:
        items, token = [], None
        while True:
            query = {"prefix": PREFIX, "maxResults": 1000}
            if token:
                query["pageToken"] = token
            with urlopen(f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urlencode(query), timeout=60) as r:
                page = json.load(r)
            items.extend(x for x in page.get("items", []) if not x["name"].endswith("/"))
            token = page.get("nextPageToken")
            if not token:
                break
        if not items:
            raise RuntimeError("Official checkpoint listing is empty")
        manifest = {"source": f"gs://{BUCKET}/{PREFIX}", "objects": [
            {k: x.get(k) for k in ("name", "generation", "size", "md5Hash", "crc32c", "updated")} for x in items]}
        source.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"objects": len(manifest["objects"]), "bytes": sum(int(x["size"]) for x in manifest["objects"])}), flush=True)

    def fetch(item):
        relative = item["name"].removeprefix(PREFIX)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("Invalid object path")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            md5, sha, crc = digests(path)
            if path.stat().st_size != int(item["size"]) or crc != item["crc32c"] or (item["md5Hash"] and md5 != item["md5Hash"]):
                raise ValueError(f"Existing file has unexpected bytes: {relative}")
            return {"path": relative, "sha256": sha, "bytes": path.stat().st_size}
        partial = path.with_name(path.name + ".partial")
        # Orbax includes empty commit markers. No HTTP body is needed, but
        # the file must exist before the common checksum/rename path runs.
        if int(item["size"]) == 0:
            partial.touch(exist_ok=True)
        url = f"https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o/" + quote(item["name"], safe="")
        url += "?" + urlencode({"alt": "media", "generation": item["generation"]})
        if int(item["size"]) > 16 * 1024 * 1024:
            parts = path.with_name(path.name + ".parts")
            parts.mkdir(exist_ok=True)
            chunk_size = 4 * 1024 * 1024
            def fetch_chunk(start):
                end = min(start + chunk_size, int(item["size"])) - 1
                part = parts / str(start)
                if part.exists() and part.stat().st_size == end - start + 1:
                    return part
                for attempt in range(8):
                    try:
                        request = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
                        with urlopen(request, timeout=90) as response:
                            expected = f"bytes {start}-{end}/{item['size']}"
                            if response.status != 206 or response.headers.get("Content-Range") != expected:
                                raise RuntimeError("Server returned an unexpected byte range")
                            temporary = part.with_suffix(".tmp")
                            with temporary.open("wb") as out:
                                shutil.copyfileobj(response, out, 1024 * 1024)
                        if temporary.stat().st_size != end - start + 1:
                            raise RuntimeError("Truncated range")
                        temporary.replace(part)
                        return part
                    except Exception:
                        if attempt == 7:
                            raise
                        time.sleep(min(2 ** attempt, 16))
            with ThreadPoolExecutor(max_workers=args.chunk_workers) as pool:
                chunks = list(pool.map(fetch_chunk, range(0, int(item["size"]), chunk_size)))
            with partial.open("wb") as out:
                for chunk in chunks:
                    with chunk.open("rb") as inp:
                        shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
        for attempt in range(5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset < int(item["size"]):
                    request = Request(url, headers={"Range": f"bytes={offset}-", "Accept-Encoding": "identity"})
                    with urlopen(request, timeout=90) as response:
                        if offset and response.status != 206:
                            raise RuntimeError("Server refused resumable range")
                        with partial.open("ab" if offset else "wb") as out:
                            while block := response.read(4 * 1024 * 1024):
                                out.write(block)
                md5, sha, crc = digests(partial)
                if partial.stat().st_size != int(item["size"]) or crc != item["crc32c"] or (item["md5Hash"] and md5 != item["md5Hash"]):
                    raise ValueError(f"Downloaded checksum mismatch: {relative}")
                partial.rename(path)
                print(f"verified {relative} ({item['size']} bytes)", flush=True)
                return {"path": relative, "sha256": sha, "bytes": path.stat().st_size}
            except ValueError:
                raise
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 16))
        raise RuntimeError("Download did not complete")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(fetch, manifest["objects"]))
    (root / "verified-files.json").write_text(json.dumps(files, indent=2) + "\n")
    print(f"Verified official checkpoint: {root}", flush=True)


if __name__ == "__main__":
    main()
