"""Small controlled CycleGAN / CycleGAN+HV pilot using pinned author networks.

Custom training adapter. NOT official SPPE-GAN. Training consumes train.json only.
"""
import argparse,hashlib,importlib.util,itertools,json,math,os,random,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def js(path,x):Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def read(path):return json.loads(Path(path).read_text())
def state_hash(nets):
    h=hashlib.sha256()
    for name,net in nets.items():
        for k,t in net.state_dict().items():h.update((name+k).encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def networks(root,training=True):
    mod=load_module('author_networks',Path(__file__).resolve().parent/'third_party/cyclegan/networks.py');norm=mod.get_norm_layer('instance')
    nets={'G_SL':mod.ResnetGenerator(1,1,32,norm_layer=norm,n_blocks=6,use_dropout=False)}
    if training:
        nets.update(G_LS=mod.ResnetGenerator(1,1,32,norm_layer=norm,n_blocks=6,use_dropout=False),
                    D_L=mod.NLayerDiscriminator(1,32,3,norm_layer=norm),D_S=mod.NLayerDiscriminator(1,32,3,norm_layer=norm))
    return mod,nets
def gray(path,layout=False):
    a=np.asarray(Image.open(path).convert('RGB'))
    return np.max(255-a,axis=2).astype(np.uint8) if layout else np.asarray(Image.fromarray(a).convert('L'))
def cache_data(root):
    rows=read(root/'benchmark/train.json');out={'S':[],'L':[]}
    for r in rows:
        assert r['split']=='train'
        for domain,side in [('S','B'),('L','A')]:
            a=gray(r[side+'_source'],domain=='L');h,w=a.shape
            im=Image.fromarray(a).resize((max(2,round(w*128/h)),128),Image.Resampling.BICUBIC)
            a=np.asarray(im)
            if a.shape[1]<128:
                d=128-a.shape[1];a=np.pad(a,((0,0),(d//2,d-d//2)),mode='reflect')
            out[domain].append(a.copy())
    return rows,out
def sample(cache,domain,rng,batch,trace,seen):
    parts=[]
    for _ in range(batch):
        ix=int(rng.randint(len(cache[domain])));seen.add(ix);a=cache[domain][ix]
        x=int(rng.randint(a.shape[1]-128+1));h=int(rng.randint(2));v=int(rng.randint(2))
        a=a[:,x:x+128]
        if h:a=np.fliplr(a)
        if v:a=np.flipud(a)
        parts.append(a.copy());trace.update(f'{domain},{ix},{x},{h},{v};'.encode())
    return torch.from_numpy(np.stack(parts))[:,None].to('cuda',dtype=torch.float32).div_(127.5).sub_(1)
def train(root,group):
    proto=read(root/'benchmark/protocol.json');cfg=proto['training'];seed=proto['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    out=root/'models'/group;out.mkdir(parents=True,exist_ok=False)
    js(out/'status.json',{'status':'initializing','pid':os.getpid(),'group':group})
    mod,nets=networks(root);init=root/'benchmark/initial_networks.pt'
    if group=='B':
        assert not init.exists()
        for net in nets.values():mod.init_weights(net,'normal',0.02)
        torch.save({k:n.state_dict() for k,n in nets.items()},init)
    else:
        state=torch.load(init,map_location='cpu',weights_only=True)
        for k,n in nets.items():n.load_state_dict(state[k])
    initial_hash=state_hash(nets)
    for n in nets.values():n.to('cuda').train()
    G,F,DL,DS=[nets[k] for k in ['G_SL','G_LS','D_L','D_S']]
    optG=torch.optim.Adam(itertools.chain(G.parameters(),F.parameters()),lr=cfg['lr'],betas=tuple(cfg['betas']))
    optD=torch.optim.Adam(itertools.chain(DL.parameters(),DS.parameters()),lr=cfg['lr'],betas=tuple(cfg['betas']))
    lossGAN=mod.GANLoss('lsgan').cuda();L1=nn.L1Loss()
    poolmod=load_module('author_image_pool',Path(__file__).resolve().parent/'third_party/cyclegan/image_pool.py')
    poolL,poolS=poolmod.ImagePool(cfg['image_pool']),poolmod.ImagePool(cfg['image_pool'])
    rows,cache=cache_data(root);rng=np.random.RandomState(seed+100);trace=hashlib.sha256();seenS=set();seenL=set()
    hv_weight=4.0 if group=='C' else 0.0
    run={'group':group,'definition':proto['groups'][group],'initial_state_sha256':initial_hash,
         'training_manifest_sha256':hashlib.sha256((root/'benchmark/train.json').read_bytes()).hexdigest(),
         'config':cfg,'lambda_hv':hv_weight,'torch':str(torch.__version__),'cuda':torch.version.cuda,
         'GPU':torch.cuda.get_device_name(),'pid':os.getpid(),'seed':seed,
         'randomness':'same initialization and independent explicit NumPy data RNG; cuDNN benchmark means bitwise training reproducibility is not promised'}
    js(out/'run_config.json',run);started=time.perf_counter();steps=cfg['steps']
    logs=open(out/'train.jsonl','w',buffering=1)
    print(json.dumps({'event':'training_started','group':group,'steps':steps,'train_samples':len(rows),'initial_hash':initial_hash}),flush=True)
    for step in range(1,steps+1):
        t=time.perf_counter();S=sample(cache,'S',rng,cfg['batch_size'],trace,seenS);L=sample(cache,'L',rng,cfg['batch_size'],trace,seenL)
        lr=cfg['lr']*(1-max(0,step-cfg['lr_decay_start'])/(steps-cfg['lr_decay_start']+1))
        for opt in [optG,optD]:
            for pg in opt.param_groups:pg['lr']=lr
        for net in [DL,DS]:net.requires_grad_(False)
        optG.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            fakeL=G(S);fakeS=F(L);recS=F(fakeL);recL=G(fakeS)
            gan=lossGAN(DL(fakeL),True)+lossGAN(DS(fakeS),True)
            cycle=L1(recS,S)+L1(recL,L)
            identity=L1(G(L),L)+L1(F(S),S)
            hv=torch.zeros((),device='cuda')
            if hv_weight:
                dims=[3] if step%2 else [2]
                flipL=G(torch.flip(S,dims));flipS=F(torch.flip(L,dims))
                hv=(L1(fakeL,torch.flip(flipL,dims))+L1(flipL,torch.flip(fakeL,dims))+
                    L1(fakeS,torch.flip(flipS,dims))+L1(flipS,torch.flip(fakeS,dims)))
            lossG=gan+cfg['lambda_cycle']*cycle+cfg['lambda_identity_effective']*identity+hv_weight*hv
        if not torch.isfinite(lossG):raise RuntimeError(f'Nonfinite generator loss at {step}')
        lossG.backward();optG.step()
        for net in [DL,DS]:net.requires_grad_(True)
        optD.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            lossD=(lossGAN(DL(L),True)+lossGAN(DL(poolL.query(fakeL.detach())),False)+
                   lossGAN(DS(S),True)+lossGAN(DS(poolS.query(fakeS.detach())),False))*.5
        if not torch.isfinite(lossD):raise RuntimeError(f'Nonfinite discriminator loss at {step}')
        lossD.backward();optD.step()
        if step==1 or step%50==0 or step==steps:
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            row={'step':step,'group':group,'G':float(lossG.detach()),'D':float(lossD.detach()),
                 'GAN':float(gan.detach()),'cycle_L1':float(cycle.detach()),'identity_L1':float(identity.detach()),
                 'HV_raw':float(hv.detach()),'lr':lr,'elapsed_seconds':elapsed,'estimated_remaining_seconds':elapsed/step*(steps-step),
                 'peak_GPU_GB':torch.cuda.max_memory_allocated()/1e9}
            logs.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            js(out/'status.json',dict(status='running',pid=os.getpid(),**row))
    logs.close();torch.cuda.synchronize();elapsed=time.perf_counter()-started
    checkpoint={'networks':{k:n.state_dict() for k,n in nets.items()},'optimizer_G':optG.state_dict(),
                'optimizer_D':optD.state_dict(),'step':steps,'run':run}
    torch.save(checkpoint,out/'final.pt')
    final=dict(run,status='completed',steps_completed=steps,training_seconds=elapsed,
               batch_sequence_sha256=trace.hexdigest(),unique_sem_train_tiles_seen=len(seenS),unique_layout_train_tiles_seen=len(seenL),
               final_state_sha256=state_hash(nets),checkpoint_sha256=hashlib.sha256((out/'final.pt').read_bytes()).hexdigest())
    if group=='C':
        b=read(root/'models/B/training_summary.json')
        assert final['initial_state_sha256']==b['initial_state_sha256']
        assert final['batch_sequence_sha256']==b['batch_sequence_sha256']
        final['B_C_initialization_and_batch_sequence_identical']=True
    js(out/'training_summary.json',final);js(out/'status.json',{'status':'completed','steps':steps,'seconds':elapsed})
    print(json.dumps({'event':'training_completed','group':group,'seconds':elapsed,'batch_sequence_sha256':trace.hexdigest()}),flush=True)

def infer(root,group):
    import cv2
    torch.set_num_threads(4);cv2.setNumThreads(4)
    _,nets=networks(root,False);G=nets['G_SL'].cuda().eval()
    checkpoint=root/'models'/group/'final.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    G.load_state_dict(state['networks']['G_SL']);del state
    original=gray(root/'inputs/B_coordinate_mosaic.png');H,W=original.shape
    factor=float(read(root/'benchmark/protocol.json').get('inference_scale',1.0));target=(round(W*factor),round(H*factor))
    a=cv2.resize(original,target,interpolation=cv2.INTER_CUBIC);h,w=a.shape
    ys=list(range(0,max(1,h-128+1),96));xs=list(range(0,max(1,w-128+1),96))
    if ys[-1]!=h-128:ys.append(h-128)
    if xs[-1]!=w-128:xs.append(w-128)
    positions=[(y,x) for y in ys for x in xs];acc=np.zeros((h,w),np.float32);weight=np.zeros_like(acc)
    window=np.maximum(np.outer(np.hanning(128),np.hanning(128)),.001).astype(np.float32)
    out=root/'generated'/group;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(positions),32):
            pp=positions[start:start+32];batch=torch.from_numpy(np.stack([a[y:y+128,x:x+128] for y,x in pp]))[:,None].cuda().float()/127.5-1
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=G(batch)
            pred=((pred[:,0].float().clamp(-1,1)+1)*127.5).cpu().numpy()
            for (y,x),p in zip(pp,pred):acc[y:y+128,x:x+128]+=p*window;weight[y:y+128,x:x+128]+=window
    assert weight.min()>0
    generated=cv2.resize(acc/weight,(W,H),interpolation=cv2.INTER_AREA).clip(0,255).astype(np.uint8)
    Image.fromarray(generated).save(out/'SEM_to_layout.png');torch.cuda.synchronize()
    report={'group':group,'seconds':time.perf_counter()-started,'tiles':len(positions),'output_size':[W,H],
            'input_sha256':hashlib.sha256((root/'inputs/B_coordinate_mosaic.png').read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256((out/'SEM_to_layout.png').read_bytes()).hexdigest(),
            'coordinates_used':False,'input_scale':factor,'stride':96,'tile_size':128}
    js(out/'inference.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','infer']);p.add_argument('--root',required=True);p.add_argument('--group',choices=['B','C'],required=True);a=p.parse_args()
    if a.command=='train':train(Path(a.root),a.group)
    else:infer(Path(a.root),a.group)
