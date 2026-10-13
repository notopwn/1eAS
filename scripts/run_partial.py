"""Compare partial ResNet18 fine-tuning on the unchanged cleaned dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from data import load_folder,split_train_test
from train import build_model,count_trainable,train,save
from evaluate import accuracy,confusion_matrix,worst_examples,plot_history,plot_confusion,plot_worst
from run import Tee

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scope',choices=['block','stage'],required=True)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    tag='partial_'+args.scope
    results=ROOT/'results'
    assert not (results/f'{tag}_model.pt').exists(), 'Do not overwrite a previous experiment'
    assert not (results/f'{tag}_output.txt').exists(), 'Do not overwrite previous output'
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [results/'clean_model.pt',ROOT/'docs/model.onnx']}
    sys.stdout=Tee(sys.stdout,results/f'{tag}_output.txt')
    torch.manual_seed(args.seed)
    X,y,names,paths=load_folder(ROOT/'data/clean')
    Xtr,ytr,ptr,Xte,yte,pte=split_train_test(X,y,paths,seed=args.seed)
    original=json.loads((results/'clean_settings.json').read_text())
    assert [str(p.resolve()) for p in pte]==original['test_paths']
    print(f'{len(X)} images; training on {len(Xtr)}, testing on {len(Xte)}')
    print('Test paths identical to clean run: True')
    model=build_model(len(names),pretrained=True,freeze=True)
    block=model.layer4[-1] if args.scope=='block' else model.layer4
    for p in block.parameters():p.requires_grad=True
    prefix='layer4.1.' if args.scope=='block' else 'layer4.'
    trainable=[name for name,p in model.named_parameters() if p.requires_grad]
    assert all(name.startswith(prefix) or name.startswith('fc.') for name in trainable)
    frozen={name:value.clone() for name,value in model.state_dict().items() if not (name.startswith(prefix) or name.startswith('fc.'))}
    trained=f'last residual {args.scope} + fc'
    n_trainable=count_trainable(model)
    n_parameters=sum(p.numel() for p in model.parameters())
    print('Start: ImageNet; trained:',trained)
    print(f'Trainable parameters: {n_trainable:,} / {n_parameters:,}')
    print(f'Epochs: {args.epochs}; learning rate: {args.lr:g}; batch size: {args.batch_size}; seed: {args.seed}')
    started=time.time()
    history=train(model,Xtr,ytr,Xte,yte,epochs=args.epochs,lr=args.lr,batch_size=args.batch_size)
    minutes=(time.time()-started)/60
    assert all(torch.equal(value,model.state_dict()[name].cpu()) for name,value in frozen.items())
    print('Frozen parameters and running statistics unchanged: True')
    train_accuracy=accuracy(model,Xtr,ytr)
    test_accuracy=accuracy(model,Xte,yte)
    print(f'trained in {minutes:.1f} minutes')
    print(f'train accuracy {train_accuracy:.1%}')
    print(f'test accuracy {test_accuracy:.1%}')
    short={'le_sserafim_hong_eunchae':'Eunchae','le_sserafim_huh_yunjin':'Yunjin','le_sserafim_kazuha':'Kazuha','le_sserafim_kim_chaewon':'Chaewon','le_sserafim_sakura':'Sakura'}
    display=[short[name] for name in names]
    plot_history(history,results/f'{tag}_curves.png')
    plot_confusion(confusion_matrix(model,Xte,yte,len(names)),display,results/f'{tag}_confusion.png')
    mistakes=worst_examples(model,Xte,yte,pte)
    plot_worst(mistakes,display,results/f'{tag}_worst.png')
    save(model,results/f'{tag}_model.pt')
    settings={'tag':tag,'start':'ImageNet','trained':trained,'scope':args.scope,'epochs':args.epochs,'lr':args.lr,'batch_size':args.batch_size,'seed':args.seed,'class_names':names,'n_images':len(X),'n_parameters':n_parameters,'n_trainable':n_trainable,'train_accuracy':round(train_accuracy,4),'test_accuracy':round(test_accuracy,4),'minutes':round(minutes,2),'test_paths':[str(p.resolve()) for p in pte]}
    (results/f'{tag}_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    (results/f'{tag}_history.json').write_text(json.dumps(history,indent=2)+'\n')
    row=f'| {tag} | ImageNet | {trained} | {args.epochs} | {args.lr:g} | {n_trainable:,} | {train_accuracy:.1%} | {test_accuracy:.1%} |'
    print('One line for the experiment table:')
    print(row)
    with (results/'experiment_table.md').open('a') as f:f.write(row+'\n')
    for name,digest in protected.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    print('Existing clean checkpoint and exported demo unchanged: True')

if __name__=='__main__':main()
