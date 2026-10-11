"""Multi-seed re-evaluation of the two feasibility probes (features identical to the originals).
EuroSAT: 5 random stratified 70/30 seeds. LEVIR-CD: 5 seeds of random stratified split AND scene-grouped split
(scene id parsed from filename test_<scene>_<k>). Reports mean +- std. Writes results/multiseed.json"""
import warnings; warnings.filterwarnings("ignore")
import io,json,numpy as np,torch,torch.nn as nn
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq
from torchvision import models,transforms
from torchvision.datasets import EuroSAT
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score
R=Path(__file__).resolve().parent.parent;D=R/"data"
torch.manual_seed(0);np.random.seed(0)
norm=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
net=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1);net.classifier=nn.Identity();net.eval()
def met(y,p):
    q=(p>=.5).astype(int);return dict(acc=accuracy_score(y,q),bacc=balanced_accuracy_score(y,q),f1=f1_score(y,q),auroc=roc_auc_score(y,p))
def agg(L):return {k:[round(float(np.mean([d[k] for d in L])),4),round(float(np.std([d[k] for d in L])),4)] for k in L[0]}
# EuroSAT
tf=transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor(),norm])
ds=EuroSAT(root=str(D),download=True,transform=tf)
lab=np.array([l for _,l in ds.samples]);rng=np.random.RandomState(0);keep=[]
for c in range(10):
    i=np.where(lab==c)[0];rng.shuffle(i);keep+=list(i[:800])
bu={ds.classes.index(c) for c in("Residential","Industrial","Highway")}
y=np.array([int(lab[i] in bu) for i in keep])
dl=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,keep),batch_size=64)
with torch.no_grad():X=np.vstack([net(x).numpy() for x,_ in dl])
out={"eurosat":{}}
for name,mk in[("logreg",lambda:LogisticRegression(max_iter=2000,class_weight="balanced")),("rf",lambda:RandomForestClassifier(400,random_state=0,class_weight="balanced",n_jobs=-1))]:
    L=[]
    for s in range(5):
        a,b,c,d=train_test_split(X,y,test_size=.3,stratify=y,random_state=s)
        m=make_pipeline(StandardScaler(),mk()).fit(a,c);L.append(met(d,m.predict_proba(b)[:,1]))
    out["eurosat"][name]=agg(L);print("eurosat",name,out["eurosat"][name])
# LEVIR
t=pq.read_table(D/"levircd_test.parquet").to_pydict();N=len(t["imageA"])
ld=lambda d:Image.open(io.BytesIO(d["bytes"])).convert("RGB").resize((112,112))
yl=np.array([int((np.asarray(Image.open(io.BytesIO(t["label"][i]["bytes"])).convert("L").resize((112,112)))>127).mean()>.02) for i in range(N)])
def ef(i):
    A,B=ld(t["imageA"][i]),ld(t["imageB"][i])
    with torch.no_grad():ea=net(norm(transforms.functional.to_tensor(A))[None])[0].numpy();eb=net(norm(transforms.functional.to_tensor(B))[None])[0].numpy()
    d=np.abs(np.asarray(A,np.float32)/255-np.asarray(B,np.float32)/255)
    return np.concatenate([np.abs(ea-eb),[d.mean(),d.std(),d.max(),(d>.2).mean()]])
XL=np.array([ef(i) for i in range(N)]);g=np.array([t["imageA"][i]["path"].split("_")[1] for i in range(N)])
out["levir"]={"n":N,"scenes":len(set(g))}
for nm in("random","scene_grouped"):
    L=[]
    for s in range(5):
        if nm=="random":tr,te=train_test_split(np.arange(N),test_size=.3,stratify=yl,random_state=s)
        else:tr,te=next(GroupShuffleSplit(1,test_size=.3,random_state=s).split(XL,yl,g))
        m=make_pipeline(StandardScaler(),RandomForestClassifier(400,random_state=0,class_weight="balanced",n_jobs=-1)).fit(XL[tr],yl[tr]);L.append(met(yl[te],m.predict_proba(XL[te])[:,1]))
    out["levir"][nm]=agg(L);print("levir",nm,out["levir"][nm])
json.dump(out,open(R/"results"/"multiseed.json","w"),indent=1)
