"""Audit and freeze the A/B/C benchmark; no modification of original data."""
import argparse,ast,csv,hashlib,json,re,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def box(r,side):
    m=re.search(r'_\((\d+),(\d+),(\d+),(\d+)\)\.',Path(r[side+'_source']).name)
    x,y,_,_=map(int,m.groups());w,h=ast.literal_eval(r[side+'_original_size'])
    return [x,y,x+w,y+h]
def overlap(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--mosaic-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--inference-scale',type=float,default=1.0)
    p.add_argument('--scales',type=lambda s:[float(x) for x in s.split(',')],default=[0.5,0.75,1.0,1.25,1.5])
    a=p.parse_args()
    root=Path(a.root);data=root/'benchmark';data.mkdir(exist_ok=False)
    source=a.pairs
    rows=list(csv.DictReader(source.open()));splits={s:[r for r in rows if r['split']==s] for s in ['train','val','test']}
    active=sum(splits.values(),[]);raw_checks=[]
    for r in active:
        r['layout_bbox']=box(r,'A');r['sem_bbox']=box(r,'B')
        for side in ['A','B']:
            actual=digest(r[side+'_source']);assert actual==r[side+'_source_sha256'],r['id']
        r['layout_sha256']=r['A_source_sha256'];r['sem_sha256']=r['B_source_sha256']
    checks=[]
    for s,t in [('train','val'),('train','test'),('val','test')]:
        for key in ['A_pixels64_sha256','B_pixels64_sha256']:
            n=len({r[key] for r in splits[s]} & {r[key] for r in splits[t]})
            checks.append({'splits':[s,t],'kind':key,'overlap_count':n});assert n==0,checks[-1]
        for key in ['layout_bbox','sem_bbox']:
            n=sum(overlap(x[key],y[key])>0 for x in splits[s] for y in splits[t])
            checks.append({'splits':[s,t],'kind':'spatial_'+key,'overlap_count':n});assert n==0,checks[-1]
    for s,rr in splits.items():
        rr.sort(key=lambda r:r['id']);write(data/(s+'.json'),rr)
    shutil.copy2(source,data/'pairs_original.csv')
    mosaic=a.mosaic_dir
    (root/'inputs').mkdir(exist_ok=True)
    for name in ['A_coordinate_mosaic.png','B_coordinate_mosaic.png','mosaic_manifest.json']:
        dest=root/'inputs'/name
        if not dest.exists():dest.symlink_to(mosaic/name)
    proto={'benchmark_id':'user_supplied_spatial_benchmark','seed':a.seed,
      'train_count':len(splits['train']),'validation_count':len(splits['val']),'test_count':len(splits['test']),
      'sample_units':'User-supplied raster layout / SEM tiles; the user must document independent sample units.',
      'all_split_counts':dict(Counter(r['split'] for r in rows)),
      'training_source':'Only train.json A_source and B_source; original TIFF, no 64x64 anisotropic images.',
      'train_task':'Unpaired SEM->layout and layout->SEM; pairing coordinates are not supplied to networks.',
      'training_preprocess':'Layout visible ink=max(255-RGB); grayscale SEM; each tile uniformly resized to height 128, padded by reflection when narrower than 128, random 128x128 crop; independent H/V augmentation in both domains.',
      'network':'Official CycleGAN ResnetGenerator 6 blocks, ngf=32, 1 input/output channel; PatchGAN ndf=32, 3 layers.',
      'training':{'steps':1400,'batch_size':8,'lr':0.0002,'lr_decay_start':700,'betas':[0.5,0.999],'lambda_cycle':10,'lambda_identity_effective':5,'image_pool':50,'precision':'bfloat16 autocast; FP32 weights','checkpoint_rule':'fixed last step, no test-based selection'},
      'groups':{'A':'Measured SEM directly matched','B':'CycleGAN SEM->layout, lambda_HV=0','C':'Same initialization, batches and steps as B; add SPPE-GAN equations 8-10 HV consistency, lambda_HV=4; not full SPPE-GAN'},
      'hv_implementation':'Alternate horizontal/vertical flip by step; both symmetric L1 terms in each of two generator directions; no detach.',
      'inference_scale':a.inference_scale,
      'inference':'Use the user-specified scale, fixed overlapping 128x128 tiles, stride 96, Hann blending, resize generated full image back to exact native SEM dimensions.',
      'matching':{'primary':'edge correlation','secondary':'SIFT/FLANN/RANSAC','scales':a.scales,'topk':5,'sem_preprocess':'Identical edge_image(domain=sem) for A/B/C search canvases','query_preprocess':'Identical original A_source through edge_image(domain=layout)','search_region':'entire supplied SEM, no GT ROI or row restriction'},
      'metrics':['Top1/Top3/Top5 metadata-reference bbox IoU>=0.5','median/P90 center distance in native SEM pixels','search and conversion runtime','within-test identical-layout pattern groups'],
      'metric_limits':'Metadata-reference localization proxy only; not pixel-registered ground truth, nm precision, independent-wafer generalization, or paper reproduction. Repeated layouts may make source location ambiguous.',
      'integrity_checks':checks,'pairs_sha256':digest(source),
      'input_hashes':{n:digest(root/'inputs'/n) for n in ['A_coordinate_mosaic.png','B_coordinate_mosaic.png']},
      'split_manifest_hashes':{s:digest(data/(s+'.json')) for s in splits},
      'within_split_layout_unique':{s:len({r['A_pixels64_sha256'] for r in rr}) for s,rr in splits.items()}}
    write(data/'protocol.json',proto)
    sem=Image.open(root/'inputs/B_coordinate_mosaic.png').convert('RGB');d=ImageDraw.Draw(sem,'RGBA')
    colors={'train':(60,150,255,65),'val':(255,190,20,110),'test':(255,40,80,110)}
    for s,rr in splits.items():
        for r in rr:d.rectangle(r['sem_bbox'],fill=colors[s])
    sem.thumbnail((1000,1000));sem.save(data/'split_map.png')
    print(json.dumps({k:proto[k] for k in ['benchmark_id','train_count','validation_count','test_count','all_split_counts','within_split_layout_unique','integrity_checks']},indent=2),flush=True)

if __name__=='__main__':main()
