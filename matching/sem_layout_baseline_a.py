#!/usr/bin/env python3
"""CPU baseline A: small raster layout -> full measured SEM localization.

No model training, GPU, network access, or writes to input data. Metadata
coordinates enter evaluation ONLY, never the matching function. All positions
and distances in outputs use original SEM pixels, not nm. Version 1.0.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB')).copy()


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def edge_image(rgb, domain):
    if domain=='layout':
        # All visible colors contribute. This does NOT infer a physical layer.
        gray=np.max(255-rgb,axis=2).astype(np.uint8)
        gray=cv2.GaussianBlur(gray,(3,3),0.6)
        edges=cv2.Canny(gray,20,60)
    else:
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        gray=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
        gray=cv2.GaussianBlur(gray,(3,3),0.8)
        edges=cv2.Canny(gray,50,120)
    soft=cv2.GaussianBlur(edges.astype(np.float32)/255.0,(5,5),0.9)
    return edges, soft


def bbox_iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return float(inter/union) if union>0 else 0.0


def rectangle_polygon(x,y,w,h):
    return [[float(x),float(y)],[float(x+w),float(y)],[float(x+w),float(y+h)],[float(x),float(y+h)]]


def edge_search(sem_soft, layout, scales, topk):
    """Exhaustive translation search. Deliberately no metadata/GT arguments."""
    H,W=sem_soft.shape;qh,qw=layout.shape[:2];candidates=[];diagnostics=[]
    for scale in scales:
        w=max(1,round(qw*scale));h=max(1,round(qh*scale))
        if min(w,h)<12 or w>W or h>H:
            diagnostics.append({'scale':scale,'status':'size_out_of_range','size':[w,h]});continue
        resized=cv2.resize(layout,(w,h),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
        hard,soft=edge_image(resized,'layout')
        count=int(np.count_nonzero(hard))
        if count<12 or float(np.std(soft))<1e-5:
            diagnostics.append({'scale':scale,'status':'insufficient_edges','edge_pixels':count});continue
        scores=cv2.matchTemplate(sem_soft,soft,cv2.TM_CCOEFF_NORMED)
        scores[~np.isfinite(scores)]=-2
        for _ in range(topk*4):
            _,value,_,loc=cv2.minMaxLoc(scores)
            if value<=-1:break
            x,y=loc
            candidates.append({'score':float(value),'bbox_xyxy':[x,y,x+w,y+h],
                               'polygon_xy':rectangle_polygon(x,y,w,h),'scale_requested':float(scale),
                               'scale_xy_actual':[w/qw,h/qh],
                               'query_to_sem_affine':[[w/qw,0,float(x)],[0,h/qh,float(y)]]})
            # Remove neighboring offsets for this scale before spatial NMS.
            rx=max(6,round(w*.45));ry=max(6,round(h*.45))
            scores[max(0,y-ry):min(scores.shape[0],y+ry+1),max(0,x-rx):min(scores.shape[1],x+rx+1)]=-2
        diagnostics.append({'scale':scale,'status':'searched','size':[w,h],'edge_pixels':count})
    chosen=[]
    for c in sorted(candidates,key=lambda c:c['score'],reverse=True):
        if all(bbox_iou(c['bbox_xyxy'],p['bbox_xyxy'])<0.3 for p in chosen):chosen.append(c)
        if len(chosen)>=topk:break
    for i,c in enumerate(chosen,1):c['rank']=i
    return {'status':'candidates_found' if chosen else 'no_valid_template',
            'candidates':chosen,'diagnostics':diagnostics,
            'score_meaning':'Normalized edge correlation, NOT probability or calibrated confidence',
            'scope':'full SEM, specified uniform scales, same orientation, translation only'}


def sift_features(soft):
    if not hasattr(cv2,'SIFT_create'):return None,[] ,None
    detector=cv2.SIFT_create(nfeatures=12000,contrastThreshold=0.015,edgeThreshold=10)
    k,d=detector.detectAndCompute(np.uint8(np.clip(soft*255,0,255)),None)
    return detector,k,d


def sift_search(layout, sem_shape, sem_kp, sem_desc):
    _,qsoft=edge_image(layout,'layout');detector,qkp,qdesc=sift_features(qsoft)
    base={'query_keypoints':len(qkp),'sem_keypoints':len(sem_kp),'candidates':[],
          'scope':'SIFT on edge images + FLANN ratio 0.75 + RANSAC similarity transform',
          'score_meaning':'RANSAC inlier ratio, NOT probability'}
    if detector is None: return dict(base,status='SIFT_unavailable')
    if qdesc is None or sem_desc is None or len(qkp)<6 or len(sem_kp)<2:
        return dict(base,status='insufficient_keypoints')
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=64))
    pairs=matcher.knnMatch(qdesc,sem_desc,k=2)
    good=[p[0] for p in pairs if len(p)==2 and p[0].distance<0.75*p[1].distance]
    # Keep at most one correspondence to each SEM keypoint.
    unique={}
    for m in sorted(good,key=lambda m:m.distance):unique.setdefault(m.trainIdx,m)
    good=list(unique.values());base['ratio_matches']=len(good)
    if len(good)<6:return dict(base,status='insufficient_ratio_matches')
    src=np.float32([qkp[m.queryIdx].pt for m in good]);dst=np.float32([sem_kp[m.trainIdx].pt for m in good])
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3.0,maxIters=5000,confidence=.995)
    if M is None or mask is None:return dict(base,status='geometry_failed')
    keep=mask.ravel().astype(bool);n=int(keep.sum());base['inliers']=n
    h,w=layout.shape[:2];H,W=sem_shape
    quad=np.float32([[0,0],[w,0],[w,h],[0,h]])
    poly=cv2.transform(quad[None],M)[0]
    scale=float(math.hypot(M[0,0],M[1,0]))
    coverage=float(np.ptp(src[keep,0])/w*np.ptp(src[keep,1])/h) if n else 0
    base.update(estimated_scale=scale,inlier_spatial_coverage=coverage)
    if n<4 or coverage<.04 or not .1<=scale<=4 or not np.isfinite(poly).all():
        return dict(base,status='weak_or_implausible_geometry')
    if np.any(poly[:,0]<0) or np.any(poly[:,0]>W) or np.any(poly[:,1]<0) or np.any(poly[:,1]>H):
        return dict(base,status='polygon_outside_SEM')
    residual=np.linalg.norm(cv2.transform(src[None],M)[0][keep]-dst[keep],axis=1)
    c={'rank':1,'score':n/len(good),'bbox_xyxy':[float(poly[:,0].min()),float(poly[:,1].min()),float(poly[:,0].max()),float(poly[:,1].max())],
       'polygon_xy':poly.astype(float).tolist(),'query_to_sem_affine':M.tolist(),
       'ransac_inliers':n,'ransac_inlier_median_residual_px':float(np.median(residual))}
    base['candidates']=[c]
    return dict(base,status='candidate_found')


def reference_from_row(row):
    m=re.search(r'_\((\d+),(\d+),(\d+),(\d+)\)\.',Path(row['B_source']).name)
    if not m:raise ValueError('Cannot parse B_source coordinates')
    x,y,_,_=map(int,m.groups());w,h=ast.literal_eval(row['B_original_size'])
    return [x,y,x+w,y+h]


def evaluate(candidates, reference):
    """Metadata reference is used after matching; it is not pixel-perfect GT."""
    cx=(reference[0]+reference[2])/2;cy=(reference[1]+reference[3])/2
    results=[]
    for c in candidates:
        b=c['bbox_xyxy'];results.append({'rank':c['rank'],'reference_bbox_iou':bbox_iou(b,reference),
          'reference_center_distance_px':math.hypot((b[0]+b[2])/2-cx,(b[1]+b[3])/2-cy)})
    return {'reference_type':'filename origin + actual tile dimensions; NOT pixel-perfect registration GT',
            'reference_bbox_xyxy':reference,'per_candidate':results,
            'top1_reference_hit_iou50':bool(results and results[0]['reference_bbox_iou']>=.5),
            'top3_reference_hit_iou50':any(v['reference_bbox_iou']>=.5 for v in results[:3]),
            'top5_reference_hit_iou50':any(v['reference_bbox_iou']>=.5 for v in results[:5])}


COLORS=[(255,70,50),(255,180,0),(0,210,255),(210,80,255),(255,100,180)]

def render_result(out, layout, sem, result, reference):
    Image.fromarray(layout).save(out/'query_layout.png')
    qhard,_=edge_image(layout,'layout');Image.fromarray(qhard).save(out/'query_edges.png')
    canvas=Image.fromarray(sem.copy());draw=ImageDraw.Draw(canvas)
    draw.rectangle((0,0,canvas.width,42),fill='black')
    draw.text((8,8),'Predictions: ranked colors | Metadata reference: GREEN (evaluation only)',fill='white')
    if reference:
        draw.rectangle(reference,outline=(0,255,80),width=4)
    for i,c in reversed(list(enumerate(result['candidates']))):
        p=[tuple(x) for x in c['polygon_xy']];color=COLORS[i%len(COLORS)]
        draw.line(p+[p[0]],fill=color,width=3)
        x,y=c['bbox_xyxy'][:2];draw.text((x,max(43,y-15)),f"#{i+1} {c['score']:.3f}",fill=color)
    canvas.save(out/'overview.png')
    if result['candidates']:
        c=result['candidates'][0];x0,y0,x1,y1=c['bbox_xyxy'];H,W=sem.shape[:2]
        box=[max(0,int(math.floor(x0))-8),max(0,int(math.floor(y0))-8),min(W,int(math.ceil(x1))+8),min(H,int(math.ceil(y1))+8)]
        crop=sem[box[1]:box[3],box[0]:box[2]].copy();Image.fromarray(crop).save(out/'top1_sem_crop.png')
        M=np.array(c['query_to_sem_affine'],np.float32);M[:,2]-=[box[0],box[1]]
        edges,_=edge_image(layout,'layout')
        warped=cv2.warpAffine(edges,M,(crop.shape[1],crop.shape[0]))>80
        overlay=crop.copy();overlay[warped]=(255,50,50)
        im=Image.fromarray(overlay);im.resize((im.width*3,im.height*3)).save(out/'top1_overlay_3x.png')
        if reference:
            rx0,ry0,rx1,ry1=map(int,reference)
            Image.fromarray(sem[ry0:ry1,rx0:rx1]).save(out/'reference_sem_crop_for_review_only.png')


def make_contact(out, cases):
    thumb_w,thumb_h=360,230
    sheet=Image.new('RGB',(thumb_w*3,thumb_h*len(cases)),(235,235,235));d=ImageDraw.Draw(sheet)
    for i,(name,folder,label) in enumerate(cases):
        for j,(fn,title) in enumerate([('query_layout.png','LAYOUT'),('top1_sem_crop.png','TOP1 SEM'),('top1_overlay_3x.png','OVERLAY')]):
            f=folder/fn;x=j*thumb_w;y=i*thumb_h
            d.text((x+5,y+5),name+' '+title,fill='black');d.text((x+5,y+20),label,fill='black')
            if f.exists():
                im=Image.open(f).convert('RGB');im.thumbnail((thumb_w-10,thumb_h-50));sheet.paste(im,(x+5,y+42))
    sheet.save(out/'review_contact.png')


def run(args):
    cv2.setNumThreads(args.threads);cv2.setRNGSeed(42)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    sem=load_rgb(args.sem);hard,soft=edge_image(sem,'sem')
    Image.fromarray(hard).save(out/'sem_edges.png')
    methods=['edge','sift'] if args.method=='both' else [args.method]
    skp=[];sdesc=None
    if 'sift' in methods:_,skp,sdesc=sift_features(soft)
    config={'algorithm_version':'1.0','python':sys.version,'platform':platform.platform(),
       'opencv':cv2.__version__,'numpy':np.__version__,'sem_path':str(Path(args.sem).resolve()),
       'sem_sha256':sha256(args.sem),'sem_size_wh':[sem.shape[1],sem.shape[0]],
       'scales':args.scales,'threads':args.threads,'topk':args.topk,'method':args.method,
       'orientation_assumption':'edge method uses the same orientation; no rotation or mirror search',
       'layout_channels':'all visible colored lines collapsed into ink contrast, no physical-layer identification',
       'no_training':True,'coordinate_units':'original SEM pixels','scores_are_probabilities':False}
    if args.command=='batch':
        rows=[r for r in csv.DictReader(open(args.pairs,newline='')) if r['split']==args.split]
        rows=sorted(rows,key=lambda r:r['id'])
        if not rows:raise ValueError('No rows for requested split')
        if args.limit and args.limit<len(rows):rows=[rows[i] for i in np.linspace(0,len(rows)-1,args.limit,dtype=int)]
        queries=[(r['id'],r['A_source'],r) for r in rows]
        config.update(pairs_csv=str(Path(args.pairs).resolve()),pairs_sha256=sha256(args.pairs),
                      split=args.split,selection='deterministic evenly spaced IDs; no outcome-based filtering')
    else:queries=[('query',args.layout,None)]
    save_json(out/'run_config.json',config);records=[];contact=[];started=time.perf_counter()
    for i,(name,path,row) in enumerate(queries,1):
        layout=load_rgb(path)
        for method in methods:
            t=time.perf_counter()
            result=edge_search(soft,layout,args.scales,args.topk) if method=='edge' else sift_search(layout,soft.shape,skp,sdesc)
            elapsed=time.perf_counter()-t
            reference=reference_from_row(row) if row else None
            result.update(query_id=name,query_path=path,query_sha256=sha256(path),
                          query_size_wh=[layout.shape[1],layout.shape[0]],method=method,search_seconds=elapsed)
            if reference:result['evaluation']=evaluate(result['candidates'],reference)
            folder=out/name/method;folder.mkdir(parents=True)
            save_json(folder/'result.json',result);render_result(folder,layout,sem,result,reference)
            ev=result.get('evaluation',{});pc=ev.get('per_candidate',[])
            rec={'query_id':name,'method':method,'status':result['status'],'search_seconds':elapsed,
                 'top1_score':result['candidates'][0]['score'] if result['candidates'] else '',
                 'top1_reference_bbox_iou':pc[0]['reference_bbox_iou'] if pc else '',
                 'top1_reference_center_distance_px':pc[0]['reference_center_distance_px'] if pc else '',
                 'top1_reference_hit_iou50':ev.get('top1_reference_hit_iou50',''),
                 'top3_reference_hit_iou50':ev.get('top3_reference_hit_iou50',''),
                 'top5_reference_hit_iou50':ev.get('top5_reference_hit_iou50','')}
            records.append(rec)
            if method==methods[0]:contact.append((name,folder,f"{method}: IoU={rec['top1_reference_bbox_iou']}"[:45]))
            print(json.dumps({'progress':f'{i}/{len(queries)}',**rec},ensure_ascii=False),flush=True)
        with open(out/'summary.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for offset in range(0,len(contact),10):
        dest=out if len(contact)<=10 else out/f'review_{offset//10+1:02d}'
        dest.mkdir(exist_ok=True);make_contact(dest,contact[offset:offset+10])
    summaries={}
    for method in methods:
        rr=[r for r in records if r['method']==method]
        ss={'queries':len(rr),'with_candidates':sum(r['top1_score']!='' for r in rr),
            'mean_search_seconds':float(np.mean([r['search_seconds'] for r in rr]))}
        if args.command=='batch':
            for k in [1,3,5]:
                if k>1 and method=='sift':continue
                key=f'top{k}_reference_hit_iou50';ss[key+'_count']=sum(bool(r[key]) for r in rr);ss[key+'_rate']=ss[key+'_count']/len(rr)
        summaries[method]=ss
    report={'status':'completed','total_seconds_including_rendering':time.perf_counter()-started,'methods':summaries,
      'interpretation':'A-group baseline only. Metadata bbox overlap is a coarse reference proxy, not validated localization accuracy, nm error, or SPPE-GAN reproduction.',
      'evaluation_leakage_control':'Search functions accept images and public scale grid only. Reference coordinates are used after search. Full SEM is searched.'}
    save_json(out/'summary.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def selftest():
    cv2.setNumThreads(2)
    # Geometry and scoring plumbing, not a semiconductor performance test.
    rng=np.random.default_rng(481);query=np.full((90,135,3),255,np.uint8)
    for _ in range(16):
        x,y=rng.integers([4,4],[108,69]);ww,hh=rng.integers([4,4],[20,18])
        cv2.rectangle(query,(int(x),int(y)),(int(x+ww),int(y+hh)),(0,40,220),1)
    scaled=cv2.resize(query,(108,72),interpolation=cv2.INTER_AREA)
    _,template=edge_image(scaled,'layout');search=rng.uniform(0,.025,(310,480)).astype(np.float32)
    search[151:223,247:355]=template
    found=edge_search(search,query,[.6,.8,1.0],5)
    b=found['candidates'][0]['bbox_xyxy'];assert b==[247,151,355,223],b
    assert evaluate(found['candidates'],b)['top1_reference_hit_iou50']
    blank=edge_search(search,np.full((90,135,3),255,np.uint8),[1.0],5)
    assert blank['status']=='no_valid_template'
    assert bbox_iou([0,0,10,10],[0,0,10,10])==1
    assert bbox_iou([0,0,10,10],[20,20,30,30])==0
    print(json.dumps({'selftest':'passed','known_translation_scale_recovered':b,'blank_query_rejected':True}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('selftest')
    for name in ['batch','match']:
        p=sub.add_parser(name)
        p.add_argument('--sem',required=True,help='Full measured SEM image')
        p.add_argument('--out',required=True,help='NEW output directory; existing directories are rejected')
        p.add_argument('--method',choices=['edge','sift','both'],default='edge')
        p.add_argument('--scales',type=lambda s:[float(v) for v in s.split(',')],default=[.5,.75,1.0,1.25,1.5],help='SEM pixels / original layout pixel, comma-separated')
        p.add_argument('--topk',type=int,default=5);p.add_argument('--threads',type=int,default=2)
        if name=='batch':
            p.add_argument('--pairs',required=True);p.add_argument('--split',choices=['train','val','test'],default='val')
            p.add_argument('--limit',type=int,default=5,help='0 uses all requested split rows')
        else:p.add_argument('--layout',required=True,help='Small ORIGINAL raster layout, not distorted 64x64 resized image')
    args=parser.parse_args()
    if args.command=='selftest':selftest();return
    if not args.scales or any(not math.isfinite(s) or s<=0 for s in args.scales):parser.error('scales must be finite and positive')
    if args.topk<1 or args.threads<1:parser.error('topk and threads must be positive')
    if args.command=='batch' and args.limit<0:parser.error('limit must be >= 0')
    run(args)


if __name__=='__main__':main()
