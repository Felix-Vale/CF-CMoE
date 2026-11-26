"""Pair files by stem (or remove explicitly specified suffixes); precompute geometry."""
import argparse,csv
from pathlib import Path
from train import load_config,make_dataset

def pair(root,suffix):
    result={}
    for p in sorted(Path(root).rglob('*')):
        if p.suffix.lower() not in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.gif','.ppm'}:continue
        stem=p.stem
        if suffix:
            if not stem.endswith(suffix):continue
            stem=stem[:-len(suffix)]
        if stem in result:raise ValueError(f'duplicate key {stem} in {root}')
        result[stem]=str(p.resolve())
    if not result:raise ValueError(f'no valid files: {root}')
    return result

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True);m=sub.add_parser('manifest');m.add_argument('--images',required=True);m.add_argument('--masks',required=True);m.add_argument('--fov');m.add_argument('--image-suffix',default='');m.add_argument('--mask-suffix',default='');m.add_argument('--fov-suffix',default='');m.add_argument('--output',required=True);g=sub.add_parser('geometry');g.add_argument('--config',default='config.yaml');g.add_argument('--split',choices=['train','val','test'],default='train');a=p.parse_args()
    if a.cmd=='manifest':
        im=pair(a.images,a.image_suffix);ma=pair(a.masks,a.mask_suffix);fv=pair(a.fov,a.fov_suffix) if a.fov else None
        if set(im)!=set(ma) or fv is not None and set(im)!=set(fv):raise ValueError(f'unmatched image/mask/FOV keys. image-only={list(set(im)-set(ma))[:10]}, mask-only={list(set(ma)-set(im))[:10]}')
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=['id','image','mask','fov']);w.writeheader();w.writerows(dict(id=k,image=im[k],mask=ma[k],fov=fv[k] if fv else '') for k in sorted(im))
        print(f'{len(im)} samples -> {out}')
    else:
        ds=make_dataset(load_config(a.config),a.split)
        for i in range(len(ds)):
            _,m,f=ds._load(i);ds._target(m,f)
            if (i+1)%10==0 or i+1==len(ds):print(f'{i+1}/{len(ds)}')
if __name__=='__main__':main()
