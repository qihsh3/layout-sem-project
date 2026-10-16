"""One frozen localization benchmark for raw SEM, CycleGAN, and CycleGAN+HV."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from sem_layout_baseline_a import load_rgb,edge_image,edge_search,sift_features,sift_search,evaluate

def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def review(root,sem,query,result,reference):
    root.mkdir(parents=True,exist_ok=True);Image.fromarray(query).save(root/'layout.png')
    overview=Image.fromarray(sem.copy());d=ImageDraw.Draw(overview)
    d.rectangle(reference,outline=(0,255,60),width=3)
    for c in reversed(result['candidates']):d.rectangle(c['bbox_xyxy'],outline=(255,70,30) if c['rank']==1 else (255,210,0),width=2)
    overview.thumbnail((1000,1000));overview.save(root/'overview.png')
    if result['candidates']:
        c=result['candidates'][0];x0,y0,x1,y1=c['bbox_xyxy'];H,W=sem.shape[:2]
        box=[max(0,int(x0)-6),max(0,int(y0)-6),min(W,int(np.ceil(x1))+6),min(H,int(np.ceil(y1))+6)]
        crop=sem[box[1]:box[3],box[0]:box[2]].copy();Image.fromarray(crop).save(root/'top1_SEM.png')
        M=np.asarray(c['query_to_sem_affine'],np.float32);M[:,2]-=[box[0],box[1]]
        edge,_=edge_image(query,'layout');warped=cv2.warpAffine(edge,M,(crop.shape[1],crop.shape[0]))>80
        crop[warped]=(255,50,50);Image.fromarray(crop).save(root/'overlay.png')
    x0,y0,x1,y1=map(int,reference);Image.fromarray(sem[y0:y1,x0:x1]).save(root/'reference_SEM.png')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--group',choices=['A','B','C','oracle'],required=True);p.add_argument('--split',choices=['val','test'],required=True);a=p.parse_args()
    root=Path(a.root);proto=json.loads((root/'benchmark/protocol.json').read_text());rows=json.loads((root/f'benchmark/{a.split}.json').read_text())
    out=root/'results'/a.group/a.split;out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(4);cv2.setRNGSeed(proto['seed']);original=load_rgb(root/'inputs/B_coordinate_mosaic.png')
    if a.group=='A':path=root/'inputs/B_coordinate_mosaic.png';sem=original
    elif a.group=='oracle':
        path=root/'inputs/A_coordinate_mosaic.png';layout=load_rgb(path);sem=cv2.resize(layout,(original.shape[1],original.shape[0]),interpolation=cv2.INTER_AREA)
    else:path=root/f'generated/{a.group}/SEM_to_layout.png';sem=load_rgb(path)
    started=time.perf_counter();hard,soft=edge_image(sem,'sem');Image.fromarray(hard).save(out/'search_edges.png')
    _,kp,desc=sift_features(soft);setup=time.perf_counter()-started;records=[];full=[]
    preview_ids={rows[i]['id'] for i in np.linspace(0,len(rows)-1,min(8,len(rows)),dtype=int)}
    for n,row in enumerate(rows,1):
        q=load_rgb(row['A_source'])
        for method in ['edge','sift']:
            t=time.perf_counter()
            r=edge_search(soft,q,proto['matching']['scales'],5) if method=='edge' else sift_search(q,soft.shape,kp,desc)
            seconds=time.perf_counter()-t
            # The only use of reference coordinates occurs AFTER image search.
            ev=evaluate(r['candidates'],row['sem_bbox']);pc=ev['per_candidate']
            rec={'id':row['id'],'group':a.group,'split':a.split,'method':method,'status':r['status'],'search_seconds':seconds,
                 'top1_score':r['candidates'][0]['score'] if r['candidates'] else None,
                 'top1_iou':pc[0]['reference_bbox_iou'] if pc else None,'top1_center_error_px':pc[0]['reference_center_distance_px'] if pc else None,
                 'top1_hit':ev['top1_reference_hit_iou50'],'top3_hit':ev['top3_reference_hit_iou50'],'top5_hit':ev['top5_reference_hit_iou50']}
            records.append(rec);full.append(dict(rec,query_sha256=hashlib.sha256(Path(row['A_source']).read_bytes()).hexdigest(),result=r,evaluation=ev))
            if method=='edge' and row['id'] in preview_ids:review(out/'review'/row['id'],original,q,r,row['sem_bbox'])
        if n%10==0 or n==len(rows):print(json.dumps({'group':a.group,'split':a.split,'progress':f'{n}/{len(rows)}','elapsed_seconds':time.perf_counter()-started}),flush=True)
    write(out/'per_query.json',full)
    with (out/'per_query.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary={'group':a.group,'split':a.split,'n_queries':len(rows),'benchmark_id':proto['benchmark_id'],
             'manifest_sha256':hashlib.sha256((root/f'benchmark/{a.split}.json').read_bytes()).hexdigest(),
             'search_canvas_sha256':hashlib.sha256(sem.tobytes()).hexdigest(),'matcher_setup_seconds':setup,
             'total_seconds':time.perf_counter()-started,'methods':{},'metric':'metadata source-location proxy; not nanometer accuracy'}
    for method in ['edge','sift']:
        rr=[r for r in records if r['method']==method];err=[r['top1_center_error_px'] for r in rr if r['top1_center_error_px'] is not None]
        stat={'with_candidates':len(err),'median_center_error_px_conditional_on_candidate':float(np.median(err)) if err else None,
              'p90_center_error_px_conditional_on_candidate':float(np.percentile(err,90)) if err else None,'mean_search_seconds':float(np.mean([r['search_seconds'] for r in rr]))}
        for k in ([1,3,5] if method=='edge' else [1]):
            count=sum(r[f'top{k}_hit'] for r in rr);stat[f'top{k}_hits']=count;stat[f'top{k}_rate']=count/len(rr)
        summary['methods'][method]=stat
    write(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
