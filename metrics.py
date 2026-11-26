import numpy as np
from skimage.morphology import skeletonize
from sklearn.metrics import roc_auc_score


def metrics_one(prob,t,threshold=.5):
    prob=np.asarray(prob).squeeze(); f=np.asarray(t['fov']).squeeze()>0; y=(np.asarray(t['mask']).squeeze()>0)&f; p=(prob>=threshold)&f
    tp=np.sum(p&y); fp=np.sum(p&~y&f); fn=np.sum(~p&y); tn=np.sum(~p&~y&f)
    def div(a,b): return float(a/b) if b else float('nan')
    def dice(a,b):
        den=a.sum()+b.sum(); return float(2*(a&b).sum()/den) if den else 1.
    skp=skeletonize(p); sky=skeletonize(y); prec=div((skp&y).sum(),skp.sum()); rec=div((sky&p).sum(),sky.sum())
    cl=1. if not sky.any() and not skp.any() else 0. if not sky.any() or not skp.any() or prec+rec==0 else 2*prec*rec/(prec+rec)
    thin=(np.asarray(t['thin']).squeeze()>0)&y; j=(np.asarray(t['junction']).squeeze()>0)&f
    auc=float(roc_auc_score(y[f].astype(int),prob[f])) if f.any() and np.unique(y[f]).size==2 else float('nan')
    return dict(dice=dice(p,y),iou=div(tp,tp+fp+fn) if tp+fp+fn else 1.,auc=auc,sensitivity=div(tp,tp+fn),specificity=div(tn,tn+fp),fpr=div(fp,tn+fp),accuracy=div(tp+tn,tp+tn+fp+fn),cldice=float(cl),thin_recall=div((p&thin).sum(),thin.sum()),junction_dice=dice(p&j,y&j) if j.any() else float('nan'))

def aggregate(rows):
    if not rows:return {}
    return {k:float(np.mean(v)) if (v:=[r[k] for r in rows if np.isfinite(r[k])]) else float('nan') for k in rows[0] if k!='id'}
