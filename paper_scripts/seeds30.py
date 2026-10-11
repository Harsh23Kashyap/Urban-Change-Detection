import warnings; warnings.filterwarnings("ignore")
import numpy as np,json,io
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from sklearn.metrics import roc_auc_score
XL=np.load('/tmp/XL.npy');yl=np.load('/tmp/yl.npy')
t=pq.read_table('/tmp/ucd/data/levircd_test.parquet',columns=['imageA']).to_pydict()
g=np.array([d["path"].split("_")[1] for d in t["imageA"]]);N=len(yl)
R=[];G=[]
for s in range(30):
    tr,te=train_test_split(np.arange(N),test_size=.3,stratify=yl,random_state=s)
    m=make_pipeline(StandardScaler(),RandomForestClassifier(400,random_state=0,class_weight="balanced",n_jobs=-1)).fit(XL[tr],yl[tr]);R.append(roc_auc_score(yl[te],m.predict_proba(XL[te])[:,1]))
    tr,te=next(GroupShuffleSplit(1,test_size=.3,random_state=s).split(XL,yl,g))
    m=make_pipeline(StandardScaler(),RandomForestClassifier(400,random_state=0,class_weight="balanced",n_jobs=-1)).fit(XL[tr],yl[tr]);G.append(roc_auc_score(yl[te],m.predict_proba(XL[te])[:,1]))
    print(s,R[-1],G[-1],flush=True)
R=np.array(R);G=np.array(G);d=R-G
res=dict(n=N,seeds=30,random=[R.mean(),R.std()],grouped=[G.mean(),G.std()],diff=[d.mean(),d.std(),d.std(ddof=1)/np.sqrt(30)],pos_frac=float((d>0).mean()),R=R.tolist(),G=G.tolist())
json.dump(res,open('/tmp/seeds30.json','w'));print(res)
