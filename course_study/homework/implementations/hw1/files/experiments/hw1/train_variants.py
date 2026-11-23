"""Compare the notebook's three TinyViT variants on identical MNIST examples and seeds."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST

root=Path(os.environ["R"])
sys.path.insert(0,str(root/"tests"))
from test_hw1 import notebook_namespace

def accuracy(model, images, labels, ix):
    model.eval()
    correct=0
    with torch.inference_mode():
        for batch in ix.split(256):
            correct+=int((model(images[batch,None].float()/255).argmax(1)==labels[batch]).sum())
    return correct

def main():
    p=argparse.ArgumentParser()
    p.add_argument("kind",choices=["ffn","geglu","swiglu"])
    p.add_argument("seed",type=int)
    a=p.parse_args()
    out=Path(f"{a.kind}-{a.seed}")
    out.mkdir()
    vit=notebook_namespace("ex4",stop_before=9)["TinyViT"]
    train=MNIST(root/"hw1_pytorch_tutorial/data",train=True,download=False)
    test=MNIST(root/"hw1_pytorch_tutorial/data",train=False,download=False)
    split=torch.randperm(len(train),generator=torch.Generator().manual_seed(103021))
    train_ix,val_ix=split[:48000],split[48000:53000]
    torch.manual_seed(a.seed)
    model=vit(7,64,4,2,256,.1,a.kind)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    gen=torch.Generator().manual_seed(a.seed)
    curve=[]
    for epoch in range(12):
        model.train()
        loss_sum=0.
        for ix in train_ix[torch.randperm(len(train_ix),generator=gen)].split(128):
            y=model(train.data[ix,None].float()/255)
            loss=F.cross_entropy(y,train.targets[ix])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            loss_sum+=loss.item()*len(ix)
        val=accuracy(model,train.data,train.targets,val_ix)
        curve.append(dict(epoch=epoch+1,loss=loss_sum/len(train_ix),val_correct=val))
        print(a.kind,a.seed,epoch+1,val,flush=True)
        if val==max(row["val_correct"] for row in curve):
            torch.save(model.state_dict(),out/"model.pt")
    model.load_state_dict(torch.load(out/"model.pt",weights_only=True))
    correct=accuracy(model,test.data,test.targets,torch.arange(len(test)))
    (out/"result.json").write_text(json.dumps(dict(variant=a.kind,seed=a.seed,curve=curve,train_n=len(train_ix),val_n=len(val_ix),test_n=len(test),correct=correct,parameters=sum(p.numel() for p in model.parameters())),indent=2))
if __name__=="__main__":
    main()
