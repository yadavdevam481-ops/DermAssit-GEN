from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import lpips

def image_files(root):
    return sorted(list(Path(root).glob('*.png')) + list(Path(root).glob('*.jpg')) + list(Path(root).glob('*.jpeg')))

def load_tensor(path, size=224):
    t=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor()])
    return t(Image.open(path).convert('RGB'))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--real-dir',required=True); p.add_argument('--synthetic-dir',required=True); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--max-images',type=int,default=1000); p.add_argument('--pairs',type=int,default=200); p.add_argument('--output',default='outputs/generation_metrics.json'); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    random.seed(a.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    real=image_files(a.real_dir)[:a.max_images]; fake=image_files(a.synthetic_dir)[:a.max_images]
    if len(real)<2 or len(fake)<2: raise ValueError('Each directory needs at least two images.')
    fid=FrechetInceptionDistance(normalize=False).to(device)
    for paths,flag in [(real,True),(fake,False)]:
        for i in range(0,len(paths),a.batch_size):
            b=torch.stack([load_tensor(x) for x in paths[i:i+a.batch_size]])
            fid.update((b*255).to(torch.uint8).to(device), real=flag)
    fid_value=float(fid.compute().cpu())
    ssim_metric=StructuralSimilarityIndexMeasure(data_range=1.0).to(device); lp=lpips.LPIPS(net='alex').to(device).eval()
    n=min(a.pairs,len(real),len(fake)); rp=random.sample(real,n); fp=random.sample(fake,n); ssim=[]; lpv=[]
    with torch.no_grad():
        for r,f in zip(rp,fp):
            x=load_tensor(r).unsqueeze(0).to(device); y=load_tensor(f).unsqueeze(0).to(device)
            ssim.append(float(ssim_metric(x,y).cpu())); lpv.append(float(lp(x*2-1,y*2-1).mean().cpu()))
    pool=fake[:min(len(fake),200)]; random.shuffle(pool); div=[]
    with torch.no_grad():
        for i in range(min(a.pairs,len(pool)//2)):
            x=load_tensor(pool[2*i]).unsqueeze(0).to(device); y=load_tensor(pool[2*i+1]).unsqueeze(0).to(device)
            div.append(float(lp(x*2-1,y*2-1).mean().cpu()))
    out={'fid':fid_value,'paired_ssim_mean':float(np.mean(ssim)),'paired_lpips_mean':float(np.mean(lpv)),'synthetic_pairwise_lpips_diversity':float(np.mean(div)),'num_real':len(real),'num_synthetic':len(fake)}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
