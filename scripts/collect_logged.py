"""Run the course collector unchanged, recording download provenance."""
import csv
import hashlib
import sys
import threading
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "src"))
import collect

queries = [
    "LE SSERAFIM Sakura",
    "LE SSERAFIM Kim Chaewon",
    "LE SSERAFIM Huh Yunjin",
    "LE SSERAFIM Kazuha",
    "LE SSERAFIM Hong Eunchae",
]
for query in queries:
    folder = root / "data/raw" / collect.safe_dirname(query)
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f"Refusing to overwrite an existing batch: {folder}")

results = root / "results"
results.mkdir(exist_ok=True)
local = threading.local()
lock = threading.Lock()
original_get = collect.requests.get
original_save = collect.Image.Image.save

with (results / "collection_sources.csv").open("x", newline="") as manifest:
    writer = csv.writer(manifest)
    writer.writerow(["file", "query", "image_url", "resolved_url", "source_sha256"])

    def logged_get(url, *args, **kwargs):
        response = original_get(url, *args, **kwargs)
        local.source = (url, response.url, hashlib.sha256(response.content).hexdigest())
        return response

    def logged_save(image, fp, *args, **kwargs):
        original_save(image, fp, *args, **kwargs)
        with lock:
            writer.writerow([str(Path(fp).relative_to(root)), local.query, *local.source])
            manifest.flush()

    collect.requests.get = logged_get
    collect.Image.Image.save = logged_save
    original_fetch_get = logged_get
    current_query = [None]

    def query_get(url, *args, **kwargs):
        local.query = current_query[0]
        return original_fetch_get(url, *args, **kwargs)

    collect.requests.get = query_get
    counts = {}
    for query in queries:
        current_query[0] = query
        counts[query] = collect.collect_one_class(query, 150, root / "data/raw")

    print("\nSummary", flush=True)
    for query, count in counts.items():
        print(f"  {query:30s} {count}", flush=True)
    if min(counts.values()) < 50:
        print("Some categories need a supplemental batch with another search phrase.", flush=True)
