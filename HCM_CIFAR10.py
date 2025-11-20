import argparse, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import DTD, ImageFolder
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# -------- Optional thop for FLOPs/MACs --------
_HAS_THOP = True
try:
    from thop import profile, clever_format
except Exception:
    _HAS_THOP = False

# ------------------------
# Reproducibility
# ------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------
# Backbone (ResNet18 + HCM heads)
# ------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        return F.relu(out)

class ResNet18_32x32_HCM(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10):
        super().__init__()
        self.in_planes=64
        self.conv1=nn.Conv2d(3,64,3,1,1,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.layer1=self._make_layer(block,64,num_blocks[0],stride=1)
        self.layer2=self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer3=self._make_layer(block,256,num_blocks[2],stride=2)
        self.layer4=self._make_layer(block,512,num_blocks[3],stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.feature_size=512
        self.direction_head=nn.Linear(self.feature_size,num_classes)
        self.magnitude_head=nn.Linear(self.feature_size,1)
    def _make_layer(self,block,planes,n,stride):
        strides=[stride]+[1]*(n-1); layers=[]
        for s in strides:
            layers.append(block(self.in_planes,planes,s))
            self.in_planes=planes
        return nn.Sequential(*layers)
    def forward(self,x,return_feature=False):
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
        feat=self.avgpool(x).flatten(1)
        d=self.direction_head(feat)
        R=self.magnitude_head(feat).squeeze(-1)
        if return_feature: return d,R,feat
        return d,R


# ------------------------
# Confidence utilities
# ------------------------
def compute_final_confidence(direction,R):
    n = torch.norm(direction, dim=1).clamp(min=1e-12)
    conf = torch.where(n >= 1.0, torch.exp(-(n-1.0).abs()), torch.exp(-(1.0/n - 1.0).abs()))
    return conf



# ------------------------
# Data
# ------------------------
CIFAR10_MEAN,CIFAR10_STD=(0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)
def tf_train(): return transforms.Compose([
    transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),transforms.Normalize(CIFAR10_MEAN,CIFAR10_STD)
])
def tf_test(): return transforms.Compose([
    transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize(CIFAR10_MEAN,CIFAR10_STD)
])
def tf_test_mnist(): return transforms.Compose([
    transforms.Resize((32,32)),transforms.Grayscale(3),
    transforms.ToTensor(),transforms.Normalize(CIFAR10_MEAN,CIFAR10_STD)
])

class TinyImageNetTestDataset(Dataset):
    def __init__(self,root,transform=None):
        self.files=[os.path.join(root,f) for f in os.listdir(root) if f.lower().endswith(('jpeg','jpg','png'))]
        self.transform=transform
    def __len__(self): return len(self.files)
    def __getitem__(self,idx):
        img=Image.open(self.files[idx]).convert("RGB")
        return (self.transform(img) if self.transform else img), -1

def make_cifar10_loaders(batch,workers=2):
    tr=datasets.CIFAR10("./data",train=True,transform=tf_train(),download=True)
    te=datasets.CIFAR10("./data",train=False,transform=tf_test(),download=True)
    return DataLoader(tr,batch,True,num_workers=workers,pin_memory=True), \
           DataLoader(te,batch,False,num_workers=workers,pin_memory=True)

def make_ood_loader(name,batch,workers=2,
                    tiny_path="./data/tiny-imagenet-200/test/images",
                    imagenet_val="./data/val"):
    name=name.upper(); tf32=tf_test()
    if   name=="CIFAR100": ds=datasets.CIFAR100("./data",train=False,transform=tf32,download=True)
    elif name=="SVHN":     ds=datasets.SVHN("./data",split="test",transform=tf32,download=True)
    elif name=="DTD":      ds=DTD("./data",split="test",download=True,transform=tf32)
    elif name=="TINYIMAGENET": ds=TinyImageNetTestDataset(tiny_path,transform=tf32)
    elif name=="MNIST":    ds=datasets.MNIST("./data",train=False,transform=tf_test_mnist(),download=True)
    elif name=="PLACE":    ds=ImageFolder(root=imagenet_val,transform=tf32)
    else: raise ValueError(f"Unknown OOD dataset: {name}")
    return DataLoader(ds,batch,False,num_workers=workers,pin_memory=True)


# ------------------------
# Training / Eval
# ------------------------
def train_one_epoch(model,loader,device,opt,criterion,epoch):
    model.train()
    for i,(x,y) in enumerate(loader):
        x,y=x.to(device), y.to(device)
        y_onehot=F.one_hot(y,num_classes=10).float()
        d,R=model(x); R=R.view(-1,1)
        loss_d=criterion(d,y_onehot)
        loss_r=criterion(R*y_onehot,y_onehot)
        loss=loss_d**(3/4)+loss_r**(3/4)
        if epoch<3: loss=0.1*loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if i%100==0: print(f"Epoch {epoch} | {i}/{len(loader)} | Loss {loss.item():.4f}")

@torch.no_grad()
def test_accuracy(model,loader,device):
    model.eval(); tot=correct=0
    for x,y in loader:
        x,y=x.to(device), y.to(device)
        d,R=model(x); logits=d*R.unsqueeze(-1)
        pred=logits.argmax(1)
        correct+=(pred==y).sum().item(); tot+=x.size(0)
    acc=100*correct/tot
    print(f"[Cls] Acc: {acc:.2f}%"); return acc


# ------------------------
# OOD metrics
# ------------------------
@torch.no_grad()
def hcm_confidences(model,loader,device):
    model.eval(); confs=[]
    for x,_ in loader:
        x=x.to(device); d,R=model(x)
        dir_conf=compute_final_confidence(d,R)
        confs.extend(dir_conf.cpu().numpy().tolist())
    return confs

def ood_metrics(id_conf,ood_conf,name):
    y=np.concatenate([np.zeros(len(id_conf)),np.ones(len(ood_conf))])
    scores=np.concatenate([1-np.array(id_conf),1-np.array(ood_conf)])
    auroc=roc_auc_score(y,scores); aupr=average_precision_score(y,scores)
    fpr,tpr,_=roc_curve(y,scores); idx=np.where(tpr>=0.95)[0]
    fpr95=float(fpr[idx[0]]) if len(idx) else 1.0
    print(f"[HCM | OOD={name}] AUROC: {auroc:.4f} | AUPR: {aupr:.4f} | FPR@95TPR: {fpr95:.4f}")
    return auroc,aupr,fpr95


# ------------------------
# Cost instrumentation
# ------------------------
def _unwrap(model):  # DataParallel-safe
    return model.module if isinstance(model, nn.DataParallel) else model

def _primary_device_of(model):
    return next(model.parameters()).device

def count_params(model):
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def get_flops_macs(model, img_shape=(3,32,32), batch_size=1):
    if not _HAS_THOP:
        print("[FLOPs] thop not found. `pip install thop` to enable FLOPs/MACs.")
        return None
    module = _unwrap(model)
    module.eval()
    dev = _primary_device_of(module)
    dummy = torch.randn(batch_size, *img_shape, device=dev)
    macs, params = profile(module, inputs=(dummy,), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.3f")
    print(f"[FLOPs/MACs] MACs={macs_str} | Params(thop)={params_str}")
    return macs, params, (macs_str, params_str)

@torch.no_grad()
def measure_pure_inference_cost(model, img_shape=(3,32,32), batch_sizes=(1,), warmup=50, iters=200):
    results = {}
    module = _unwrap(model); module.eval()
    dev = _primary_device_of(module)
    on_cuda = (dev.type == "cuda")
    for B in batch_sizes:
        x = torch.randn(B, *img_shape, device=dev)
        if on_cuda: torch.cuda.reset_peak_memory_stats(dev)
        # warmup
        for _ in range(warmup): _ = model(x)
        # timed
        if on_cuda:
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(dev); s.record()
            for _ in range(iters): _ = model(x)
            e.record(); torch.cuda.synchronize(dev)
            total_ms = s.elapsed_time(e); peak_mem_mb = torch.cuda.max_memory_allocated(dev)/(1024**2)
        else:
            t0 = time.perf_counter()
            for _ in range(iters): _ = model(x)
            total_ms = (time.perf_counter()-t0)*1000.0; peak_mem_mb = None
        avg_ms = total_ms/iters; thr = B*1000.0/avg_ms
        results[B] = {"avg_ms":avg_ms,"throughput":thr,"peak_mem_mb":peak_mem_mb}
        print(f"[Pure Inference] batch={B} | {avg_ms:.3f} ms/iter | {thr:.1f} img/s"
              + (f" | peak_mem={peak_mem_mb:.1f} MB" if peak_mem_mb is not None else ""))
    return results

@torch.no_grad()
def measure_hcm_scoring_cost(model, loader, num_batches=20):
    module = _unwrap(model); module.eval()
    dev = _primary_device_of(module)
    on_cuda = (dev.type == "cuda")
    processed=0
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(dev)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(dev); s.record()
    else:
        t0 = time.perf_counter()
    for i,(x,_) in enumerate(loader):
        if i>=num_batches: break
        x = x.to(dev, non_blocking=True)
        d,R = model(x)
        _ = compute_final_confidence(d,R)  # drop the result; just measure
        processed += x.size(0)
    if on_cuda:
        e.record(); torch.cuda.synchronize(dev)
        total_ms = s.elapsed_time(e); peak_mem_mb = torch.cuda.max_memory_allocated(dev)/(1024**2)
    else:
        total_ms = (time.perf_counter()-t0)*1000.0; peak_mem_mb=None
    ms_per_img = total_ms/processed; thr = 1000.0/ms_per_img
    print(f"[HCM Score] over {processed} imgs | {ms_per_img:.3f} ms/img | {thr:.1f} img/s"
          + (f" | peak_mem={peak_mem_mb:.1f} MB" if peak_mem_mb is not None else ""))
    return {"avg_ms_per_img":ms_per_img,"throughput":thr,"peak_mem_mb":peak_mem_mb}

def save_and_size(model, path):
    torch.save(model.state_dict(), path)
    try:
        size_mb = os.path.getsize(path)/(1024**2)
        print(f"[Checkpoint] saved to {path} ({size_mb:.2f} MB)")
    except Exception:
        print(f"[Checkpoint] saved to {path}")


# ------------------------
# Main
# ------------------------
def parse_args():
    p=argparse.ArgumentParser(description="HCM multi-OOD (CIFAR-10).")
    p.add_argument("--seed",type=int,default=3)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--lr",type=float,default=0.1)
    p.add_argument("--momentum",type=float,default=0.9)
    p.add_argument("--wd",type=float,default=1e-4)
    # p.add_argument("--milestones",type=int,nargs="*",default=[60,120,180])
    p.add_argument("--milestones",type=int,nargs="*",default=[30,60,90])
    p.add_argument("--gamma",type=float,default=0.1)
    p.add_argument("--oods",type=str,nargs="+",
                   default=["CIFAR100","SVHN","DTD","TINYIMAGENET","MNIST","PLACE"])
    p.add_argument("--tiny_path",type=str,default="./data/tiny-imagenet-200/test/images")
    p.add_argument("--imagenet_val",type=str,default="./data/val")
    # instrumentation
    p.add_argument("--warmup",type=int,default=50)
    p.add_argument("--iters",type=int,default=200)
    p.add_argument("--score_bench_batches",type=int,default=20)
    return p.parse_args()

def main():
    args=parse_args(); set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader,test_loader=make_cifar10_loaders(args.batch, args.workers)

    # ---- Model & DP wrapping (safe order) ----
    model = ResNet18_32x32_HCM()
    use_dp = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_dp:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion=nn.MSELoss()
    opt=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.wd)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=args.milestones,gamma=args.gamma)

    # ---- Train ----
    for e in range(1,args.epochs+1):
        train_one_epoch(model,train_loader,device,opt,criterion,e)
        test_accuracy(model,test_loader,device)
        sched.step()
    test_accuracy(model,test_loader,device)

    # ---- Static cost ----
    total_params = count_params(model)
    print(f"[Params] total: {total_params/1e6:.3f} M")
    _ = get_flops_macs(model, img_shape=(3,32,32), batch_size=1)

    # ---- Pure inference (eval, dummy input) ----
    print("\n== Pure inference (eval, dummy input) ==")
    _ = measure_pure_inference_cost(
        model, img_shape=(3,32,32),
        batch_sizes=(1, args.batch), warmup=args.warmup, iters=args.iters
    )

    # ---- HCM scoring cost (subset) ----
    print("\n== HCM scoring cost on ID subset ==")
    _ = measure_hcm_scoring_cost(model, test_loader, num_batches=args.score_bench_batches)

    # ---- Multi-OOD ----
    id_conf=hcm_confidences(model,test_loader,device)
    for name in args.oods:
        ood_loader=make_ood_loader(name,args.batch,args.workers,tiny_path=args.tiny_path,imagenet_val=args.imagenet_val)
        print(f"\n==> OOD: {name}")
        _ = measure_hcm_scoring_cost(model, ood_loader, num_batches=min(args.score_bench_batches, 20))
        ood_conf=hcm_confidences(model,ood_loader,device)
        ood_metrics(id_conf,ood_conf,name)

    save_and_size(model,f"CIFAR10_resnet18_hcm_seed{args.seed}.pth")
    print("Saved checkpoint.")

if __name__=="__main__":
    main()
