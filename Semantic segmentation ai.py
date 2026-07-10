import sys, os, subprocess, time, datetime, threading, queue

def _install(pkg, import_name=None):
    name = import_name or pkg.replace("-","_").split("[")[0]
    try: __import__(name)
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q",
            "--break-system-packages"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

for _p,_i in [("ultralytics","ultralytics"),("opencv-python","cv2"),
    ("torch","torch"),("torchvision","torchvision"),
    ("numpy","numpy"),("tqdm","tqdm"),("pillow","PIL")]:
    _install(_p,_i)

import cv2, numpy as np
from tqdm import tqdm
import torch
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
)
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except: _YOLO_OK = False

os.makedirs("screenshots", exist_ok=True)

# ── COLORS (BGR) ──────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "sky":(200,160,50),"road":(110,25,5),"sidewalk":(90,40,20),
    "vegetation":(15,120,15),"building":(80,70,60),"ground":(50,50,30),
    "pothole":(0,0,230),           # vivid RED
    "cone":(230,210,30),           # cyan
    "aeroplane":(200,200,50),"bicycle":(100,220,255),"bird":(60,200,200),
    "boat":(200,200,100),"bottle":(180,80,180),"bus":(30,160,220),
    "car":(20,190,255),"cat":(200,120,200),"chair":(160,160,200),
    "cow":(80,200,120),"diningtable":(160,200,160),"dog":(200,140,100),
    "horse":(100,160,200),"motorbike":(50,220,255),"person":(170,50,240),
    "pottedplant":(30,180,60),"sheep":(140,200,140),"sofa":(200,180,120),
    "train":(80,120,220),"tvmonitor":(220,200,80),
    "truck":(10,150,230),"traffic light":(30,30,220),"stop sign":(20,20,200),
    "fire hydrant":(200,100,100),"motorcycle":(50,220,255),
}
VOC_NAMES = ["__background__","aeroplane","bicycle","bird","boat","bottle","bus",
    "car","cat","chair","cow","diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"]
_VOC_LUT   = np.array([CLASS_COLORS.get(n,(130,130,130)) for n in VOC_NAMES],dtype=np.uint8)
SCENE_MAP  = {0:"building",1:"sky",2:"road",3:"sidewalk",4:"vegetation",5:"ground"}
_SCENE_LUT = np.array([CLASS_COLORS.get(SCENE_MAP[i],(130,130,130)) for i in range(6)],dtype=np.uint8)
KEEP_CLS   = {"person","car","truck","bus","motorcycle","bicycle",
              "traffic light","stop sign","fire hydrant","dog","cat"}

def _c(n): return CLASS_COLORS.get(n,(130,130,130))

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda"), torch.cuda.get_device_name(0)
    return torch.device("cpu"), "CPU"

# ── HEURISTIC SCENE ───────────────────────────────────────────────────────────
def heuristic_scene(frame):
    H,W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h=hsv[:,:,0].astype(np.float32); s=hsv[:,:,1].astype(np.float32); v=hsv[:,:,2].astype(np.float32)
    a = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:,:,1].astype(np.float32)
    lbl = np.zeros((H,W), np.uint8)

    veg=(h>=28)&(h<=85)&(s>=28)&(v>=22)&(a<124); lbl[veg]=4
    sz=np.zeros((H,W),bool); sz[:int(H*.52),:]=True
    sky=sz&((h>=86)&(h<=140)&(s>=18)&(v>=85)|(s<40)&(v>145))&~veg; lbl[sky]=1
    rz=np.zeros((H,W),bool); rz[int(H*.48):,:]=True
    road=rz&(s<70)&(v>=28)&(v<=215)&~veg; lbl[road]=2
    side=rz&(s<45)&(v>172)&~veg&~road; lbl[side]=3
    gz=np.zeros((H,W),bool); gz[int(H*.62):,:]=True
    lbl[gz&(h>=8)&(h<=40)&(s>=16)&(s<88)&(v>=26)&(v<158)&~veg&~road]=5
    return lbl

# ── POTHOLE DETECTOR — clean discrete blobs only ──────────────────────────────
#
#  KEY INSIGHT from the reference image:
#   • Potholes are CIRCULAR/OVAL dark depressions
#   • They appear in the MIDDLE band of the road (not edges, not far distance)
#   • They are darker than the surrounding asphalt
#   • We use per-COLUMN local contrast so road perspective doesn't trick us
#   • We reject anything too close to frame edges (road markings, shadows)
#
def detect_potholes(frame, scene_lbl):
    H,W = frame.shape[:2]

    # Only work in the actual road region from DeepLab/heuristic
    road_px = (scene_lbl==2) | (scene_lbl==3)
    # Further restrict to middle vertical band — no edges, no far distance
    band = np.zeros((H,W),bool)
    band[int(H*0.40):int(H*0.88), int(W*0.05):int(W*0.95)] = True
    roi = road_px & band
    if not np.any(roi):
        return np.zeros((H,W),bool)

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # CLAHE on grayscale to normalise warm/cool lighting
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(
            gray.astype(np.uint8)).astype(np.float32)

    # ----- PER-COLUMN local mean (handles road perspective gradient) -----
    # Compute column-wise smooth mean, then subtract
    col_mean = cv2.GaussianBlur(eq, (1, 61), 0)          # tall vertical kernel
    # Also global neighbourhood mean
    nbr_mean = cv2.GaussianBlur(eq, (51, 51), 0)
    # Use the brighter of the two as reference so we catch both dark holes
    # AND bright reflective patches inside holes
    ref = np.maximum(col_mean, nbr_mean)

    # Local texture (std dev) — potholes have rough broken edges
    sq_mean = cv2.GaussianBlur(eq**2, (15,15), 0)
    mean_sq = cv2.GaussianBlur(eq,    (15,15), 0)**2
    lstd    = np.sqrt(np.clip(sq_mean - mean_sq, 0, None))

    # Candidate: dark relative to reference AND has some texture AND inside ROI
    dark_delta = 14       # must be at least this much darker than local ref
    min_std    = 3.5      # must have some texture (not flat shadow)
    cand = roi & ((ref - eq) > dark_delta) & (lstd > min_std)

    # Also catch water/reflective bright patches strictly within dark region
    bright = roi & ((eq - nbr_mean) > 22) & (lstd > 6)
    # Only keep bright if it's surrounded by dark candidates (inside a pothole)
    dark_dilated = cv2.dilate(cand.astype(np.uint8)*255,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
    bright = bright & (dark_dilated > 0)
    cand   = cand | bright

    cand = cand.astype(np.uint8)*255

    # Morphological: close to fill holes inside blobs, open to remove thin lines
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k_close)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,  k_open)

    # --- SHAPE FILTER: keep only roughly circular/oval blobs ---
    n,lim,stats,_ = cv2.connectedComponentsWithStats(cand)
    mask = np.zeros((H,W),bool)
    min_area = 500
    max_area = int(H*W*0.12)   # max 12% — no more full-frame floods
    for lbl in range(1,n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if not (min_area < area < max_area): continue
        bw   = stats[lbl, cv2.CC_STAT_WIDTH]
        bh_  = stats[lbl, cv2.CC_STAT_HEIGHT]
        # Aspect ratio check — potholes are roughly round, not long thin lines
        if bw==0 or bh_==0: continue
        aspect = max(bw,bh_) / min(bw,bh_)
        if aspect > 5.0: continue          # reject road markings (very elongated)
        # Solidity check — potholes are solid blobs, not thin lines
        blob   = (lim==lbl).astype(np.uint8)
        solidity = area / max(bw*bh_, 1)
        if solidity < 0.12: continue       # reject wispy/thin detections
        mask[lim==lbl] = True

    return mask

# ── CONE DETECTOR — orange HSV blobs only in road zone ────────────────────────
def detect_cones(frame, scene_lbl):
    H,W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Orange hue range (traffic cones)
    lo1=np.array([0, 140,140]); hi1=np.array([16,255,255])
    lo2=np.array([168,140,140]); hi2=np.array([180,255,255])
    orange = cv2.inRange(hsv,lo1,hi1) | cv2.inRange(hsv,lo2,hi2)

    # Restrict to road/sidewalk zone, not sky
    valid = np.zeros((H,W),bool)
    valid[int(H*0.25):, :] = True
    valid &= (scene_lbl != 1) & (scene_lbl != 4)  # not sky, not vegetation
    orange[~valid] = 0

    # Morphological cleanup — cones are tall thin objects
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(5,7))
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, k)
    orange = cv2.morphologyEx(orange, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))

    # Shape filter: cones are taller than wide, not too big
    n,lim,stats,_ = cv2.connectedComponentsWithStats(orange)
    mask = np.zeros((H,W),bool)
    for lbl in range(1,n):
        area = stats[lbl,cv2.CC_STAT_AREA]
        bw   = stats[lbl,cv2.CC_STAT_WIDTH]
        bh_  = stats[lbl,cv2.CC_STAT_HEIGHT]
        if not (60 < area < int(H*W*0.025)): continue
        if bw==0: continue
        aspect = bh_/bw   # cones are taller than wide
        if aspect < 0.6: continue   # reject flat horizontal orange (road markings)
        mask[lim==lbl]=True
    return mask

# ── DEEPLAB ───────────────────────────────────────────────────────────────────
class SceneSegmenter:
    SZ=384
    def __init__(self,device):
        self.device=device
        print("  [SEG] Loading LRASPP-MobileNetV3 ...")
        w=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        self.model=lraspp_mobilenet_v3_large(weights=w).eval().to(device)
        if device.type=="cuda": self.model=self.model.half()
        self.tfm=w.transforms(); print("  [SEG] Ready")

    @torch.no_grad()
    def predict(self,rgb,oh,ow):
        from PIL import Image
        pil=Image.fromarray(cv2.resize(rgb,(self.SZ,self.SZ)))
        inp=self.tfm(pil).unsqueeze(0).to(self.device)
        if self.device.type=="cuda": inp=inp.half()
        pred=self.model(inp)["out"].argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        return cv2.resize(pred,(ow,oh),interpolation=cv2.INTER_NEAREST)

# ── YOLO ──────────────────────────────────────────────────────────────────────
class YOLOSegmenter:
    SZ=448
    def __init__(self,device):
        if not _YOLO_OK: self.model=None; return
        print("  [YOLO] Loading yolov8n-seg.pt ...")
        self.model=YOLO("yolov8n-seg.pt")
        if str(device)=="cuda": self.model.to("cuda")
        print("  [YOLO] Ready")
    def predict(self,frame):
        if self.model is None: return None
        kw={"verbose":False,"imgsz":self.SZ,"conf":0.48,"iou":0.45}
        if torch.cuda.is_available(): kw["half"]=True
        return self.model(frame,**kw)[0]

# ── ASYNC ─────────────────────────────────────────────────────────────────────
class AsyncInference:
    def __init__(self,seg,yolo):
        self.seg=seg; self.yolo=yolo
        self._q=queue.Queue(maxsize=1); self._out=None; self._lk=threading.Lock()
        threading.Thread(target=self._worker,daemon=True).start()
    def push(self,f,H,W):
        try: self._q.put_nowait((f.copy(),H,W))
        except queue.Full: pass
    def get(self):
        with self._lk: return self._out
    def _worker(self):
        while True:
            f,H,W=self._q.get(); t0=time.perf_counter()
            dl=self.seg.predict(cv2.cvtColor(f,cv2.COLOR_BGR2RGB),H,W)
            yo=self.yolo.predict(f); ms=(time.perf_counter()-t0)*1000
            with self._lk: self._out=(dl,yo,ms)

# ── COMPOSITE ─────────────────────────────────────────────────────────────────
def build_layers(H,W,scene,dl,yr,ph,cones):
    col=np.zeros((H,W,3),np.float32)
    alp=np.zeros((H,W),np.float32)
    edg=np.zeros((H,W,3),np.float32)
    names=set()

    # 1. Scene tint (subtle)
    col[:]=_SCENE_LUT[scene].astype(np.float32); alp[:]=0.28
    for k,v in SCENE_MAP.items():
        if np.any(scene==k): names.add(v)

    # 2. DeepLab VOC objects
    obj=dl>0
    if np.any(obj):
        dc=_VOC_LUT[dl]; col[obj]=dc[obj].astype(np.float32); alp[obj]=0.55
        for i in range(1,len(VOC_NAMES)):
            if np.any(dl==i): names.add(VOC_NAMES[i])
        m8=obj.astype(np.uint8)*255
        edg[cv2.morphologyEx(m8,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0]=[80,230,255]

    # 3. POTHOLES — vivid red, solid blobs
    if np.any(ph):
        col[ph]=np.array([0,0,225],np.float32); alp[ph]=0.80
        names.add("pothole")
        m8=ph.astype(np.uint8)*255
        edg[cv2.morphologyEx(m8,cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))>0]=[0,110,255]

    # 4. CONES — cyan
    if np.any(cones):
        col[cones]=np.array([230,210,30],np.float32); alp[cones]=0.82
        names.add("cone")
        m8=cones.astype(np.uint8)*255
        edg[cv2.morphologyEx(m8,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0]=[255,220,0]

    # 5. YOLO road objects
    if yr is not None and yr.masks is not None:
        for i in range(len(yr.masks.data)):
            cn=yr.names.get(int(yr.boxes.cls[i]),"object")
            if cn not in KEEP_CLS: continue
            cf=float(yr.boxes.conf[i])
            bm=cv2.resize(yr.masks.data[i].cpu().numpy(),(W,H),
                          interpolation=cv2.INTER_LINEAR)>0.45
            if not np.any(bm): continue
            col[bm]=np.array(_c(cn),np.float32); alp[bm]=min(0.84,0.60+cf*0.20)
            names.add(cn)
            m8=bm.astype(np.uint8)*255
            edg[cv2.morphologyEx(m8,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0]=[50,255,210]

    return col,alp,edg,names

def composite(frame,col,alp,edg):
    out=frame.astype(np.float32); a3=alp[:,:,np.newaxis]
    return np.clip(out*(1-a3)+col*a3+edg*0.65,0,255).astype(np.uint8)

# ── HUD ───────────────────────────────────────────────────────────────────────
FM=cv2.FONT_HERSHEY_DUPLEX
def _t(img,txt,pos,sc,col,th=1,sh=True):
    x,y=pos
    if sh: cv2.putText(img,txt,(x+1,y+1),FM,sc,(0,0,0),th+1,cv2.LINE_AA)
    cv2.putText(img,txt,(x,y),FM,sc,col,th,cv2.LINE_AA)

def hud(frame,fps,fi,tot,ms,dev,names,ts):
    H,W=frame.shape[:2]
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(W,36),(8,8,8),-1)
    cv2.addWeighted(ov,0.72,frame,0.28,0,frame)
    cv2.line(frame,(0,36),(W,36),(0,200,255),1)
    _t(frame,f"FPS {fps:4.1f}",(10,25),0.55,(60,255,60))
    _t(frame,f"INF {ms:.0f}ms",(125,25),0.55,(200,200,200))
    _t(frame,dev,(W-155,25),0.48,(255,190,50))
    y0=H-26; ov2=frame.copy()
    cv2.rectangle(ov2,(0,y0),(W,H),(8,8,8),-1)
    cv2.addWeighted(ov2,0.72,frame,0.28,0,frame)
    cv2.line(frame,(0,y0),(W,y0),(0,200,255),1)
    if tot>0:
        pw=int((fi/max(tot,1))*(W-20))
        cv2.rectangle(frame,(10,y0+4),(10+pw,y0+8),(0,210,255),-1)
    cv2.rectangle(frame,(10,y0+4),(W-10,y0+8),(45,45,45),1)
    _t(frame,f"{fi:05d}/{tot:05d}",(12,y0+20),0.38,(150,150,150))
    _t(frame,ts,(W-138,y0+20),0.38,(120,120,120))
    if names:
        nms=sorted(names)[:16]; lh=19; x0=W-178; yt=44
        ph2=5*2+len(nms)*lh; ov3=frame.copy()
        cv2.rectangle(ov3,(x0-5,yt),(W-5,yt+ph2),(10,10,10),-1)
        cv2.addWeighted(ov3,0.70,frame,0.30,0,frame)
        cv2.rectangle(frame,(x0-5,yt),(W-5,yt+ph2),(0,175,195),1)
        for i,nm in enumerate(nms):
            cy=yt+5+i*lh+12; c=_c(nm)
            cv2.rectangle(frame,(x0,cy-8),(x0+11,cy+2),c,-1)
            cv2.rectangle(frame,(x0,cy-8),(x0+11,cy+2),(210,210,210),1)
            _t(frame,nm[:22],(x0+15,cy),0.34,(205,205,205),sh=False)

# ── PROCESSOR ─────────────────────────────────────────────────────────────────
class Processor:
    def __init__(self,source,interval=4):
        self.source=source; self.interval=interval
        self.device,self.dlabel=get_device()
        print(f"  [DEVICE] {self.dlabel}")
        self.seg=SceneSegmenter(self.device)
        self.yolo=YOLOSegmenter(self.device)
        self.ai=AsyncInference(self.seg,self.yolo)

    def run(self):
        src=self.source; webcam=src in("0","1","2") or src==0
        cap=cv2.VideoCapture(int(src) if webcam else src)
        if not cap.isOpened(): print(f"Cannot open {src}"); sys.exit(1)
        tot=0 if webcam else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in=cap.get(cv2.CAP_PROP_FPS) or 30
        scale=min(1.0,1280/max(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),1))
        W=int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))*scale)
        H=int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))*scale)

        writer=cv2.VideoWriter("output_segmented.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),fps_in,(W,H))
        print(f"  [INFO] {W}x{H} | {fps_in:.0f}fps | {tot} frames")
        print("  Q=quit  P=pause  S=screenshot\n")
        win="Semantic Segmentation AI"
        cv2.namedWindow(win,cv2.WINDOW_NORMAL); cv2.resizeWindow(win,W,H)

        bar=tqdm(total=tot or None,desc="  Processing",unit="frame",dynamic_ncols=True)
        tp=time.time(); fps_s=0.0; fi=0; ms=0.0; paused=False
        dl_lbl=np.zeros((H,W),np.uint8); yr=None

        # Very subtle vignette
        cx,cy=W/2,H/2; Y,X=np.ogrid[:H,:W]
        vig=np.clip(1.0-np.sqrt(((X-cx)/cx)**2+((Y-cy)/cy)**2)*0.12,0.88,1.0).astype(np.float32)

        while True:
            if paused:
                k=cv2.waitKey(30)&0xFF
                if k in(ord("q"),27): break
                if k==ord("p"): paused=False
                continue
            ret,raw=cap.read()
            if not ret: break
            frame=cv2.resize(raw,(W,H)) if scale<1.0 else raw.copy()

            if fi%self.interval==0: self.ai.push(frame,H,W)
            res=self.ai.get()
            if res: dl_lbl,yr,ms=res

            scene=heuristic_scene(frame)
            ph   =detect_potholes(frame, scene)
            cones=detect_cones(frame, scene)

            col,alp,edg,names=build_layers(H,W,scene,dl_lbl,yr,ph,cones)
            out=composite(frame,col,alp,edg)
            out=np.clip(out.astype(np.float32)*vig[:,:,np.newaxis],0,255).astype(np.uint8)

            tn=time.time(); fps_s=0.88*fps_s+0.12/(max(tn-tp,1e-6)); tp=tn
            hud(out,fps_s,fi,tot,ms,self.dlabel,names,
                datetime.datetime.now().strftime("%H:%M:%S"))

            writer.write(out); cv2.imshow(win,out)
            fi+=1; bar.update(1)
            k=cv2.waitKey(1)&0xFF
            if k in(ord("q"),27): break
            elif k==ord("p"): paused=True
            elif k==ord("s"):
                sp=os.path.join("screenshots",
                    f"shot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
                cv2.imwrite(sp,out); print(f"\n  [SHOT] {sp}")

        bar.close(); cap.release(); writer.release(); cv2.destroyAllWindows()
        print(f"\n  [DONE] {fi} frames → output_segmented.mp4\n")

def main():
    if len(sys.argv)<2:
        print("Usage: python semantic_seg_final.py <video.mp4>")
        print("       python semantic_seg_final.py 0   (webcam)")
        sys.exit(0)
    Processor(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 4).run()

if __name__=="__main__": main()