import warnings; warnings.filterwarnings("ignore")
import numpy as np,json
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score,brier_score_loss
XL=np.load('/tmp/XL.npy');y=np.load('/tmp/yl.npy')
t=pq.read_table('/tmp/ucd/data/levircd_test.parquet',columns=['imageA']).to_pydict()
g=np.array([d["path"].split("_")[1] for d in t["imageA"]])
def mk():return make_pipeline(StandardScaler(),RandomForestClassifier(400,random_state=0,class_weight="balanced",n_jobs=-1))
def ece(p,yy,b=10):
    e=0;bins=np.minimum((p*b).astype(int),b-1)
    for i in range(b):
        m=bins==i
        if m.any():e+=m.mean()*abs(p[m].mean()-yy[m].mean())
    return e
R=[];rel=[]
for s in range(5):
    tr,te=next(GroupShuffleSplit(1,test_size=.3,random_state=s).split(XL,y,g))
    a,c=next(GroupShuffleSplit(1,test_size=.25,random_state=s).split(XL[tr],y[tr],g[tr]));fit,cal=tr[a],tr[c]
    m=mk().fit(XL[fit],y[fit]);pc=m.predict_proba(XL[cal])[:,1];pt=m.predict_proba(XL[te])[:,1]
    iso=IsotonicRegression(out_of_bounds="clip").fit(pc,y[cal]);pi=iso.predict(pt)
    R.append(dict(seed=s,ece_raw=ece(pt,y[te]),ece_iso=ece(pi,y[te]),brier_raw=brier_score_loss(y[te],pt),brier_iso=brier_score_loss(y[te],pi),auc_raw=roc_auc_score(y[te],pt),auc_iso=roc_auc_score(y[te],pi)))
    print(R[-1],flush=True)
    rel.append((pt.tolist(),pi.tolist(),y[te].tolist()))
json.dump(dict(R=R,rel=rel),open('/tmp/calib.json','w'))
