"""Augmentation + AdamW + validation-selected early stopping."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torchvision import transforms as T
from PIL import Image
from data import MEAN,STD,load_folder,split_train_test
from train import build_model,count_trainable,save
from evaluate import accuracy,confusion_matrix,worst_examples,plot_confusion,plot_worst
from run import Tee
import matplotlib.pyplot as plt

class AugmentedImages(Dataset):
    def __init__(self,X,y):
        self.X,self.y=X,y
        self.augment=T.Compose([T.RandomHorizontalFlip(),T.ColorJitter(brightness=0.15,contrast=0.15,saturation=0.1,hue=0.02),T.ToTensor(),T.Normalize(MEAN,STD)])
    def __len__(self):return len(self.y)
    def __getitem__(self,i):
        # Recover the center-cropped RGB pixels; augment copies, never files.
        rgb=self.X[i].transpose(1,2,0)*np.asarray(STD,dtype=np.float32)+np.asarray(MEAN,dtype=np.float32)
        picture=Image.fromarray(np.clip(np.rint(rgb*255),0,255).astype(np.uint8))
        return self.augment(picture),int(self.y[i])

def main():
    tag='regularized'
    results=ROOT/'results'
    assert not (results/f'{tag}_model.pt').exists()
    assert not (results/f'{tag}_output.txt').exists()
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [results/'clean_model.pt',ROOT/'docs/model.onnx']}
    sys.stdout=Tee(sys.stdout,results/f'{tag}_output.txt')
    torch.manual_seed(0)
    X,y,names,paths=load_folder(ROOT/'data/clean')
    Xpool,ypool,ppool,Xtest,ytest,ptest=split_train_test(X,y,paths,seed=0)
    baseline=json.loads((results/'clean_settings.json').read_text())
    assert [str(p.resolve()) for p in ptest]==baseline['test_paths']
    Xfit,yfit,pfit,Xval,yval,pval=split_train_test(Xpool,ypool,ppool,test_ratio=0.2,seed=1)
    sets=[set(pfit),set(pval),set(ptest)]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert sets[0]|sets[1]|sets[2]==set(paths)
    del X,Xpool
    print(f'Fit images: {len(pfit)}; validation images: {len(pval)}; test images: {len(ptest)}')
    print('Training/validation/test paths disjoint: True')
    print('Downloaded test paths identical to clean run: True')
    print('Training-only augmentation: horizontal flip; mild brightness/contrast/saturation/hue jitter')
    print('AdamW: lr=0.0001, weight_decay=0.01; batch_size=32; seed=0')
    print('Checkpoint selection: lowest validation loss; maximum 15 epochs; patience 4')
    print('Downloaded test and independent photos are not used for checkpoint selection.')
    model=build_model(len(names))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimiser=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.01)
    loss_fn=nn.CrossEntropyLoss()
    fitting=DataLoader(AugmentedImages(Xfit,yfit),batch_size=32,shuffle=True)
    validating=DataLoader(TensorDataset(torch.from_numpy(Xval),torch.from_numpy(yval)),batch_size=32)
    history={'train_loss':[],'train_acc':[],'validation_loss':[],'validation_acc':[]}
    best_loss=float('inf');best_state=None;best_epoch=0;stale=0
    started=time.time()
    for epoch in range(1,16):
        model.train();total_loss=0.;right=0
        for rows,labels in fitting:
            rows,labels=rows.to(device),labels.to(device)
            optimiser.zero_grad();logits=model(rows);loss=loss_fn(logits,labels)
            loss.backward();optimiser.step()
            total_loss+=loss.item()*len(labels)
            right+=(logits.argmax(1)==labels).sum().item()
        model.eval();val_loss=0.;val_right=0
        with torch.no_grad():
            for rows,labels in validating:
                rows,labels=rows.to(device),labels.to(device)
                logits=model(rows)
                val_loss+=loss_fn(logits,labels).item()*len(labels)
                val_right+=(logits.argmax(1)==labels).sum().item()
        values=[total_loss/len(pfit),right/len(pfit),val_loss/len(pval),val_right/len(pval)]
        for key,value in zip(history,values):history[key].append(value)
        improved=values[2]<best_loss-1e-4
        if improved:
            best_loss=values[2];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        print(f'epoch {epoch}/15: train loss {values[0]:.4f}, train accuracy {values[1]:.4f}, validation loss {values[2]:.4f}, validation accuracy {values[3]:.4f}, best epoch {best_epoch}',flush=True)
        if stale>=4:
            print('Early stopping: validation loss has not improved for four epochs.')
            break
    assert best_state is not None
    model.load_state_dict(best_state);model.eval()
    minutes=(time.time()-started)/60
    train_accuracy=accuracy(model,Xfit,yfit)
    validation_accuracy=accuracy(model,Xval,yval)
    test_accuracy=accuracy(model,Xtest,ytest)
    print(f'Selected epoch: {best_epoch}; epochs actually run: {epoch}')
    print(f'trained in {minutes:.1f} minutes')
    print(f'train accuracy on unaugmented fit images {train_accuracy:.1%}')
    print(f'validation accuracy {validation_accuracy:.1%}')
    print(f'test accuracy {test_accuracy:.1%}')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,keys,title in zip(axes,[['train_loss','validation_loss'],['train_acc','validation_acc']],['Loss','Accuracy']):
        for key in keys:ax.plot(range(1,epoch+1),history[key],label=key)
        ax.axvline(best_epoch,color='gray',linestyle='--',label='selected epoch')
        ax.set_xlabel('epoch');ax.set_title(title);ax.legend()
    axes[1].set_ylim(0,1);fig.tight_layout();fig.savefig(results/f'{tag}_curves.png',dpi=120);plt.close(fig)
    display=[name.removeprefix('le_sserafim_').replace('hong_','').replace('huh_','').replace('kim_','').title() for name in names]
    plot_confusion(confusion_matrix(model,Xtest,ytest,len(names)),display,results/f'{tag}_confusion.png')
    plot_worst(worst_examples(model,Xtest,ytest,ptest),display,results/f'{tag}_worst.png')
    save(model,results/f'{tag}_model.pt')
    settings={'tag':tag,'start':'ImageNet','trained':'all layers, augmentation + AdamW + early stopping','epochs':epoch,'selected_epoch':best_epoch,'max_epochs':15,'patience':4,'lr':1e-4,'weight_decay':0.01,'batch_size':32,'seed':0,'validation_split_seed':1,'class_names':names,'n_images':len(paths),'n_fit':len(pfit),'n_validation':len(pval),'n_test':len(ptest),'n_parameters':sum(p.numel() for p in model.parameters()),'n_trainable':count_trainable(model),'train_accuracy':round(train_accuracy,4),'validation_accuracy':round(validation_accuracy,4),'test_accuracy':round(test_accuracy,4),'minutes':round(minutes,2),'fit_paths':[str(p.resolve()) for p in pfit],'validation_paths':[str(p.resolve()) for p in pval],'test_paths':[str(p.resolve()) for p in ptest],'augmentation':{'horizontal_flip_probability':0.5,'brightness':0.15,'contrast':0.15,'saturation':0.1,'hue':0.02}}
    (results/f'{tag}_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    (results/f'{tag}_history.json').write_text(json.dumps(history,indent=2)+'\n')
    row=f'| {tag} | ImageNet | all layers, augmentation + AdamW + early stopping | {epoch} (selected {best_epoch}) | 0.0001 | {count_trainable(model):,} | {train_accuracy:.1%} | {test_accuracy:.1%} |'
    print(row)
    with (results/'experiment_table.md').open('a') as f:f.write(row+'\n')
    for path,digest in protected.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    print('Existing clean checkpoint and exported demo unchanged: True')

if __name__=='__main__':main()
