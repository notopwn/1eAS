import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw

root=Path.cwd()
assert not Path('data/face_clean').exists() and not Path('data/face_new_images').exists()
detections=json.loads(Path('results/face_detection_results.json').read_text())
assert len(detections)==596
errors=[d for d in detections if d.get('error')]
assert not errors, f'Face-detector errors need review: {errors[:2]}'
records=[]
for detection in detections:
    source=Path(detection['path'])
    relative=source.relative_to(root)
    kind=relative.parts[1]
    out_root='face_clean' if kind=='clean' else 'face_new_images'
    target=Path('data')/out_root/source.parent.name/(source.name+'.png')
    target.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as im:
        im.load();image=im.convert('RGB');W,H=image.size
        faces=sorted([f for f in detection['faces'] if f['confidence']>=0.5 and min(f['width']*W,f['height']*H)>=24],key=lambda f:-f['width']*f['height'])
        status='no reliable face; original framing retained'
        box=None
        if faces:
            best=faces[0]
            dominant=len(faces)==1 or best['width']*best['height']>=2*faces[1]['width']*faces[1]['height']
            if dominant:
                x=best['x']*W;y=(1-best['y']-best['height'])*H
                w=best['width']*W;h=best['height']*H
                size=max(1,math.ceil(2.0*max(w,h)))
                # PIL pads beyond the boundary without stretching or synthesizing face pixels.
                left=round(x+w/2-size/2);top=round(y+h/2-size/2)
                box=[left,top,left+size,top+size]
                cropped=image.crop(box)
                status='dominant face crop'
            else:
                status='multiple comparable faces; original framing retained'
        if box is None: cropped=image
        cropped.save(target,'PNG')
    records.append({'source':str(relative),'target':str(target),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'status':status,'detected_faces':len(faces),'crop_box':box,'output_size':list(cropped.size)})
counts=Counter((r['source'].split('/')[1],r['status']) for r in records)
for (kind,status),count in sorted(counts.items()):print(kind,status,count)
print('Copied training images:',len(list(Path('data/face_clean').glob('*/*.png'))))
print('Copied independent images:',len(list(Path('data/face_new_images').glob('*/*.png'))))
assert len(list(Path('data/face_clean').glob('*/*.png')))==558
assert len(list(Path('data/face_new_images').glob('*/*.png')))==38
Path('results/face_crop_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
review=[r for r in records if r['status']!='dominant face crop']
Path('results/face_crop_review.json').write_text(json.dumps(review,indent=2)+'\n')
chosen=[]
for folder in sorted(Path('data/face_clean').iterdir()):
    candidate=next(r for r in records if Path(r['target']).parent==folder and r['crop_box'])
    chosen.append(candidate)
for folder in sorted(Path('data/face_new_images').iterdir()):
    candidate=next((r for r in records if Path(r['target']).parent==folder and r['crop_box']),None)
    if candidate: chosen.append(candidate)
canvas=Image.new('RGB',(850,280*len(chosen)),'white');draw=ImageDraw.Draw(canvas)
for i,r in enumerate(chosen):
    y=i*280
    draw.text((10,y+5),r['source'],fill='black')
    for col,name in enumerate([r['source'],r['target']]):
        with Image.open(name) as im:
            im=im.convert('RGB');im.thumbnail((380,225));canvas.paste(im,(10+col*430,y+30))
    draw.text((10,y+260),'Original',fill='black');draw.text((440,y+260),'Face-focused copy',fill='black')
canvas.save('results/face_crop_preview.png')
print('Flagged fallback images:',len(review))
print('Source images unchanged:',all(hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()==r['source_sha256'] for r in records))
