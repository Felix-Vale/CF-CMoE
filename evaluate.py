import argparse,csv,json
from pathlib import Path
import torch
from models import CFCMoE
from train import load_config,make_dataset,make_loader,validate

def main():
    p=argparse.ArgumentParser();
    p.add_argument('--checkpoint',default=r"F:\CF_CMoE\runs\OCTA-6M\epoch_197.pt");
    p.add_argument('--config');
    p.add_argument('--split',choices=['val','test'],default='test');
    p.add_argument('--csv', default='F:\CF_CMoE\csv\OCTA-6M-test.csv');
    p.add_argument('--output',default=r'F:\CF_CMoE\out\Ours');
    p.add_argument('--device',default='auto');
    p.add_argument('--threshold',type=float);
    p.add_argument('--save-predictions',action='store_true');a=p.parse_args()
    dev=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=load_config(a.config) if a.config else ck['config'];cfg['model']=ck['config']['model']
    if a.csv:cfg['data'][a.split+'_csv']=a.csv
    model=CFCMoE(**cfg['model']).to(dev);model.load_state_dict(ck['model']);loader=make_loader(make_dataset(cfg,a.split),cfg);sc=ck.get('schedule',{'eta':1.,'route_mix':1.});out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    summary,rows=validate(model,loader,dev,cfg['training'].get('amp',True),sc['eta'],sc['route_mix'],a.threshold if a.threshold is not None else cfg['training'].get('threshold',.5),out/'predictions' if a.save_predictions else None)
    with (out/'per_image.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    from train import append_json
    # overwrite summary for repeatable evaluation
    (out/'summary.json').unlink(missing_ok=True);append_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
