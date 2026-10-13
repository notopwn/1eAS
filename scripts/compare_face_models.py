import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'src')
from check import compare_on_new_images

print('Same 38 image IDs; original framing:')
compare_on_new_images(['clean','face'],Path('data/new_images').resolve())
print('\nSame 38 image IDs; face-focused copies (fallbacks retain original framing):')
compare_on_new_images(['clean','face'],Path('data/face_new_images').resolve())
manifest=json.loads(Path('results/face_crop_manifest.json').read_text())
for r in manifest:assert hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()==r['source_sha256']
for path,digest in json.loads(Path('results/face_protected_hashes.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
print('Existing clean checkpoint, exported demo, and source images unchanged: True')
