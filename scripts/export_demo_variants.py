import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'src')
from export_web import export
from train import load
protected={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['docs/model.onnx','docs/models/face/model.onnx']}
for tag in ['run','scratch','frozen','low_lr','partial_block','regularized']:
    settings=json.loads(Path(f'results/{tag}_settings.json').read_text())
    samples=[]
    for name in settings['test_paths']:
        p=Path(name)
        if not p.exists():p=Path(str(p).replace('/data/clean/','/data/raw/'))
        if p.exists():samples.append(p)
        if len(samples)==3:break
    print('\nExporting',tag,flush=True)
    export(load(Path(f'results/{tag}_model.pt')),settings['class_names'],samples,out_dir=f'docs/models/{tag}')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
print('Original clean and face exports unchanged: True')
