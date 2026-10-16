"""Auditable experiments using upstream BBDM models and bridge equations.

Cache the frozen VQ encodings, keep test data unopened during training, select
EMA weights by fixed validation generation metrics, and retain resumable state.
This is our experiment runner, not code published by GDS2SEM's authors.
"""
import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent / 'BBDM'


def dump(path, obj):
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def setup(gpu, seed):
    sys.path.insert(0, str(REPO))
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_device(gpu)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch, torch.device(f'cuda:{gpu}')


def build(cfg, device, initialize=False):
    from utils import dict2namespace
    from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
    from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
    from runners.utils import weights_init
    cls = LatentBrownianBridgeModel if cfg['model']['model_type']=='LBBDM' else BrownianBridgeModel
    model = cls(dict2namespace(cfg).model).to(device)
    if initialize:
        model.apply(weights_init)
    return model


def rgb(t):
    import torch
    return ((t.detach().cpu().float().clamp(-1,1).permute(0,2,3,1)+1)*127.5+.5).clamp(0,255).to(torch.uint8).numpy()


def read_pairs(root, split, names=None):
    import torch
    names = sorted(p.name for p in (root/split/'A').glob('*.png')) if names is None else names
    assert names
    values=[]
    for side in ('A','B'):
        imgs=[]
        for name in names:
            with Image.open(root/split/side/name) as im:
                assert im.size == (64,64) and im.mode == 'RGB'
                imgs.append(np.array(im))
        values.append(torch.from_numpy(np.stack(imgs)).permute(0,3,1,2).float()/127.5-1)
    return names, *values


def choose_val(root, count):
    names=sorted(p.name for p in (root/'val/A').glob('*.png'))
    return sorted(names, key=lambda n: hashlib.sha256(('validation:'+n).encode()).hexdigest())[:count]


def read_style(root, split, names):
    """Read optional user-supplied acquisition settings from a local CSV."""
    import torch
    records = {}
    for row in csv.DictReader((root/'style.csv').open()):
        if row['split'] != split:
            continue
        foreground, background = float(row['foreground']), float(row['background'])
        if not 0 <= background < foreground <= 255:
            raise ValueError('Expected 0 <= background < foreground <= 255')
        records[row['file']] = [foreground/255, background/255]
    return torch.tensor([records[name] for name in names], dtype=torch.float32)


def normalize_appearance(images, style, inverse=False):
    fg,bg=style[:,0,None,None,None],style[:,1,None,None,None]
    unit=(images+1)/2
    return (bg+(fg-bg)*unit)*2-1 if inverse else ((unit-bg)/(fg-bg))*2-1


def config(args):
    import yaml
    cfg=yaml.safe_load(args.config.read_text())
    cfg['model']['BB']['params'].update(objective='grad',loss_type='l2',sample_step=args.sample_steps)
    cfg['model']['BB']['optimizer']['lr']=args.lr
    cfg['model']['BB']['lr_scheduler']['patience']=3000
    cfg['data']['dataset_config']['dataset_path']=str(args.data)
    cfg['data']['dataset_name']=args.output.name
    cfg['model']['model_name']=args.arch
    cfg['model']['appearance_normalization']=args.normalize_style
    cfg['training'].update(n_epochs=args.epochs,n_steps=args.epochs*(args.train_count or 10000)//args.batch)
    cfg['data']['train']['batch_size']=args.batch
    if args.arch=='pixel':
        cfg['model']['model_type']='BBDM'
        cfg['model'].pop('VQGAN',None)
        unet=cfg['model']['BB']['params']['UNetParams']
        unet.update(image_size=64,model_channels=64,channel_mult=[1,2,4],attention_resolutions=[4],num_head_channels=32)
    return cfg


def cache_encoded(net, a, b, device):
    import torch
    if net.model_config.model_type != 'LBBDM':
        return a,b
    aa,bb=[],[]
    with torch.no_grad():
        for start in range(0,len(a),64):
            aa.append(net.encode(a[start:start+64].to(device),cond=True).cpu())
            bb.append(net.encode(b[start:start+64].to(device),cond=False).cpu())
    return torch.cat(aa),torch.cat(bb)


@contextlib.contextmanager
def ema_weights(net, ema):
    import torch
    raw={n:p.detach().clone() for n,p in net.denoise_fn.named_parameters()}
    with torch.no_grad():
        for n,p in net.denoise_fn.named_parameters():
            p.copy_(ema[n])
    try:
        yield
    finally:
        with torch.no_grad():
            for n,p in net.denoise_fn.named_parameters():
                p.copy_(raw[n])


def evaluate(net, a, b, names, out, device, seed, batch=8, threshold=130, style=None):
    import torch
    from image_metrics import metric, means
    out.mkdir(parents=True,exist_ok=False)
    net.eval()
    preds=[]
    # Evaluation consumes its own seeded random stream, not the training stream.
    with torch.random.fork_rng(devices=[device.index]), torch.no_grad():
        for start in range(0,len(a),batch):
            torch.manual_seed(seed+start)
            y=a[start:start+batch].to(device)
            if net.model_config.model_type=='LBBDM':
                y=net.encode(y,cond=True)
            x=y
            for i in range(len(net.steps)):
                x,_=net.p_sample(x,y,None,i,clip_denoised=False)
            if net.model_config.model_type=='LBBDM':
                x=net.decode(x,cond=False)
            if getattr(net.model_config,'appearance_normalization',False):
                assert style is not None,'Acquisition parameters are required for this enhanced model'
                x=normalize_appearance(x,style[start:start+batch].to(device),inverse=True)
            if not torch.isfinite(x).all():
                raise FloatingPointError('Nonfinite generation')
            preds.extend(rgb(x))
    targets,layouts=rgb(b),rgb(a)
    records=[dict(name=n,**metric(t,p,threshold)) for n,t,p in zip(names,targets,preds)]
    avg=means([{k:v for k,v in r.items() if k!='name'} for r in records])
    report={'count':len(names),'mean':avg,'threshold':threshold,'seed':seed,'sample_steps':len(net.steps),
            'convention':f'PIL L, SSIM 7x7 uniform/sample covariance, bright foreground >{threshold}, per-image average',
            'per_image':records}
    dump(out/'metrics.json',report)
    np.savez_compressed(out/'images.npz',names=names,layout=layouts,target=targets,prediction=np.stack(preds))
    # All fixed examples are saved, with the first 16 in the overview.
    canvas=Image.new('RGB',(576,min(16,len(names))*206+24),'white')
    d=ImageDraw.Draw(canvas)
    for c,label in enumerate(['Layout','Target SEM','Generated SEM']):
        d.text((c*192+4,4),label,fill='black')
    for row,(name,x,t,p) in enumerate(zip(names,layouts,targets,preds)):
        if row>=16:
            break
        for col,img in enumerate([x,t,p]):
            canvas.paste(Image.fromarray(img).resize((192,192),Image.Resampling.NEAREST),(col*192,24+row*206))
        d.text((4,216+row*206),name,fill='black')
    canvas.save(out/'overview.png')
    return report


def atomic_torch(path, value):
    import torch
    temp=path.with_name(path.name+'.tmp')
    torch.save(value,temp)
    os.replace(temp,path)


def train(args):
    import yaml
    torch,device=setup(args.gpu,args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=config(args)
    cfg['evaluation']={'threshold':args.threshold}
    (args.output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    spec=vars(args).copy()
    spec={k:str(v) if isinstance(v,Path) else v for k,v in spec.items()}
    spec.update(status='PREPARING',source_config_sha256=sha(args.config),script_sha256=sha(Path(__file__)),
                upstream_model_sha256=sha(REPO/'model/BrownianBridge/BrownianBridgeModel.py'),
                split_policy='train for optimization, validation for selection; no test reads',
                differences_from_old_run=['noise objective changed to original BBDM grad objective',
                    'frozen encodings cached without augmentation','selection on fixed validation IoU, then PSNR',
                    'explicit metric reports, compact checkpoint storage'])
    if args.arch=='pixel':
        spec['differences_from_old_run']+=['engineering comparison: pixel space, smaller UNet, no VQ']
    dump(args.output/'run.json',spec)
    net=build(cfg,device,initialize=True)
    if args.initialize_from:
        saved=torch.load(args.initialize_from,map_location='cpu',weights_only=True)
        assert saved['config']['model']['BB']['params']['UNetParams']==cfg['model']['BB']['params']['UNetParams']
        assert saved['config']['model']['BB']['params']['objective']==cfg['model']['BB']['params']['objective']
        net.load_state_dict(saved['model'],strict=True)
        spec['initialization']={'checkpoint':str(args.initialize_from),'sha256':sha(args.initialize_from),
            'source_epoch':saved['epoch'],'new_optimizer':True,'new_scheduler':True,'counters_reset':True}
        del saved
    names,pa,pb=read_pairs(args.data,'train')
    if args.train_count:
        keep=sorted(range(len(names)),key=lambda i:hashlib.sha256(('fit:'+names[i]).encode()).hexdigest())[:args.train_count]
        names=[names[i] for i in keep]; pa,pb=pa[keep],pb[keep]
    # The fit experiment intentionally evaluates the same training examples.
    if args.train_count:
        vn,va,vb=names,pa,pb
    else:
        vn,va,vb=read_pairs(args.data,'val',choose_val(args.data,args.val_count))
    val_style=None
    if args.normalize_style:
        if args.arch!='pixel':
            raise ValueError('Appearance-normalized experiment is implemented in pixel space only')
        train_style=read_style(args.data,'train',names)
        val_style=read_style(args.data,'train' if args.train_count else 'val',vn)
        pb=normalize_appearance(pb,train_style)
        spec['additional_input']='User-supplied foreground/background acquisition settings; not inferred from validation pixels'
    a,b=cache_encoded(net,pa,pb,device)
    if args.arch=='latent':
        # Verify caching preserves the exact frozen encoder path.
        with torch.no_grad():
            check=net.encode(pa[:8].to(device),cond=True)
            error=(a[:8].to(device)-check).abs()
            dump(args.output/'cache_check.json',{'max_absolute_error':float(error.max()),'mean_absolute_error':float(error.mean())})
            assert float(error.max())<1e-4, 'Frozen encoder cache differs beyond float32 tolerance'
            assert not any(p.requires_grad for p in net.vqgan.parameters())
    params=list(net.denoise_fn.parameters())
    optimizer=torch.optim.Adam(params,lr=args.lr,betas=(.9,.999))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',threshold_mode='rel',**cfg['model']['BB']['lr_scheduler'])
    ema={n:p.detach().clone() for n,p in net.denoise_fn.named_parameters()}
    batch=args.batch
    started=time.time(); step=0; best=(-1.,-1.); running=[]; history=[]
    spec.update(status='TRAINING',train_count=len(a),validation_count=len(va),
                validation_scope='seen training only; not held-out' if args.train_count else 'held-out validation',
                trainable_parameters=sum(p.numel() for p in params),train_names=names,validation_names=vn)
    dump(args.output/'run.json',spec)
    for epoch in range(1,args.epochs+1):
        order=torch.randperm(len(a))
        for start in range(0,len(a)-batch+1,batch):
            ids=order[start:start+batch]
            x,y=b[ids].to(device),a[ids].to(device)
            net.train()
            optimizer.zero_grad(set_to_none=True)
            t=torch.randint(0,net.num_timesteps,(len(ids),),device=device)
            loss,_=net.p_losses(x,y,None,t)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if step==0:
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            optimizer.step()
            scheduler.step(loss.detach())
            step+=1
            with torch.no_grad():
                for n,p in net.denoise_fn.named_parameters():
                    if step<100:
                        ema[n].copy_(p)
                    else:
                        ema[n].lerp_(p,.005)
            running.append(float(loss.detach()))
            if step%100==0:
                state={'status':'TRAINING','epoch':epoch,'step':step,'loss_mean_100':float(np.mean(running[-100:])),
                       'lr':optimizer.param_groups[0]['lr'],'elapsed_s':time.time()-started,
                       'gpu_peak_GiB':torch.cuda.max_memory_allocated(device)/2**30}
                print(json.dumps(state),flush=True)
                dump(args.output/'status.json',state)
        if epoch%args.eval_every==0 or epoch==args.epochs:
            with ema_weights(net,ema):
                ev=evaluate(net,va,vb,vn,args.output/f'val_epoch_{epoch:03d}',device,args.seed,threshold=args.threshold,style=val_style)
                iou,psnr=ev['mean'][f'IoU_{args.threshold}'],ev['mean']['PSNR_gray']
                score=(psnr,iou) if args.selection=='psnr' else (iou,psnr)
                if score>best:
                    best=score
                    atomic_torch(args.output/'best_ema.pt',{'model':net.state_dict(),'config':cfg,'epoch':epoch,'step':step,'validation':ev['mean']})
                    dump(args.output/'best.json',{'epoch':epoch,'step':step,'validation':ev['mean'],'selection':f'fixed validation {args.selection} first'})
            history.append({'epoch':epoch,'step':step,'train_loss_mean':float(np.mean(running)),**ev['mean']})
            dump(args.output/'history.json',history)
            print(json.dumps(history[-1]),flush=True)
            atomic_torch(args.output/'latest_resume.pt',{'model':net.state_dict(),'ema':ema,'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'epoch':epoch,'step':step,'config':cfg,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device),'best_score':best})
            running=[]
    spec.update(status='TRAINING_COMPLETE',epochs_completed=args.epochs,steps_completed=step,elapsed_s=time.time()-started)
    dump(args.output/'run.json',spec)
    dump(args.output/'status.json',{k:v for k,v in spec.items() if k not in ('train_names','validation_names')})
    print('TRAINING_COMPLETE',flush=True)


def eval_only(args):
    torch,device=setup(args.gpu,args.seed)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    threshold=saved['config'].get('evaluation',{}).get('threshold',args.threshold)
    net=build(saved['config'],device)
    net.load_state_dict(saved['model'],strict=True)
    if args.sample_steps:
        net.sample_step=args.sample_steps; net.register_schedule(); net.to(device)
    del saved
    names,a,b=read_pairs(args.data,args.split)
    style=read_style(args.data,args.split,names) if getattr(net.model_config,'appearance_normalization',False) else None
    report=evaluate(net,a,b,names,args.output,device,args.seed,threshold=threshold,style=style)
    dump(args.output/'provenance.json',{'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha(args.checkpoint),
            'split':args.split,'data':str(args.data),'scope':'our independent split and explicit metrics, not author split'})
    print(json.dumps(report['mean']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','evaluate'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--config',type=Path,help='User-provided model YAML; required for training')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--arch',choices=['latent','pixel'],default='latent')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--train-count',type=int,default=0)
    p.add_argument('--val-count',type=int,default=64)
    p.add_argument('--eval-every',type=int,default=1)
    p.add_argument('--sample-steps',type=int,default=200)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--initialize-from',type=Path,help='EMA checkpoint for fine-tuning; resets optimizer and counters')
    p.add_argument('--threshold',type=int,default=130)
    p.add_argument('--normalize-style',action='store_true',help='Enhanced experiment with known imaging gray levels as extra input')
    p.add_argument('--selection',choices=['iou','psnr'],default='iou')
    p.add_argument('--split',choices=['val','test'],default='val')
    args=p.parse_args()
    if args.action == 'train' and args.config is None:
        p.error('--config is required for training')
    if args.action == 'evaluate' and args.checkpoint is None:
        p.error('--checkpoint is required for evaluation')
    train(args) if args.action=='train' else eval_only(args)


if __name__=='__main__':
    main()
