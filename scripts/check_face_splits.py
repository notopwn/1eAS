import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'src')
from data import load_folder,split_train_test

manifest=json.loads(Path('results/face_crop_manifest.json').read_text())
mapping={str(Path(r['source']).resolve()):str(Path(r['target']).resolve()) for r in manifest}
baseline=json.loads(Path('results/clean_settings.json').read_text())
X,y,names,paths=load_folder('data/face_clean')
_,_,ptr,_,_,pte=split_train_test(X,y,paths,seed=0)
assert names==baseline['class_names'] and len(paths)==558
assert [str(p.resolve()) for p in pte]==[mapping[p] for p in baseline['test_paths']]
assert not set(ptr)&set(pte)
print('Face-dataset training/test source IDs match the clean split: True')
print('Face-dataset training images:',len(ptr))
print('Face-dataset test images:',len(pte))
for record in manifest:assert hashlib.sha256(Path(record['source']).read_bytes()).hexdigest()==record['source_sha256']
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('results/clean_model.pt'),Path('docs/model.onnx')]}
Path('results/face_protected_hashes.json').write_text(json.dumps(protected,indent=2)+'\n')
print('All source-image hashes unchanged: True')
