"""Optional MMPose frontend with original area/ROI/temporal-IoU selection."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import JOINTS


def select_box(boxes,previous,width,height):
    widths=boxes[:,2]-boxes[:,0];heights=boxes[:,3]-boxes[:,1];areas=widths*heights
    center=np.column_stack([boxes[:,0]+widths/2,boxes[:,1]+heights/2])
    distance=np.linalg.norm(center-np.array([width/2,height/2]),axis=1)
    scores=areas/areas.max()*60+(1-distance/np.hypot(width/2,height/2))*20
    if previous is not None:
        for j,box in enumerate(boxes):
            overlap=np.maximum(0,np.minimum(box[2:4],previous[2:4])-np.maximum(box[:2],previous[:2])).prod()
            old=(previous[2]-previous[0])*(previous[3]-previous[1])
            if overlap:scores[j]+=20*overlap/(areas[j]+old-overlap)
    return boxes[int(np.argmax(scores)),:4]


def extract_video(video,output,det_config,det_weights,pose_config,pose_weights,device="cuda:0"):
    import cv2
    from mmdet.apis import inference_detector,init_detector
    from mmpose.apis import inference_topdown,init_model
    from mmpose.evaluation.functional import nms
    from mmpose.utils import adapt_mmdet_pipeline
    detector=init_detector(str(det_config),str(det_weights),device=device)
    detector.cfg=adapt_mmdet_pipeline(detector.cfg)
    pose=init_model(str(pose_config),str(pose_weights),device=device)
    capture=cv2.VideoCapture(str(video));fps=capture.get(cv2.CAP_PROP_FPS)
    if abs(fps-20)>1e-5:raise ValueError("Expected 20 fps; resampling requires a separate documented protocol")
    indices=[5,7,9,11,6,8,10,12];previous=None;rows=[];i=0
    while True:
        ok,frame=capture.read()
        if not ok:break
        pred=inference_detector(detector,frame).pred_instances.cpu().numpy()
        keep=(pred.labels==0)&(pred.scores>.3)
        boxes=np.c_[pred.bboxes[keep],pred.scores[keep]]
        if len(boxes)==0:
            raise ValueError(f"No person detected at frame {i}; paper extraction has no full-frame fallback")
        boxes=boxes[nms(boxes,.3)];h,w=frame.shape[:2]
        previous=select_box(boxes,previous,w,h)
        result=inference_topdown(pose,frame,previous[None,:])[0].pred_instances
        points=result.keypoints[0];confidence=result.keypoint_scores[0]
        row={"timestamp_sec":i/fps}
        for name,index in zip(JOINTS,indices):
            row[name+"_x"]=float(points[index,0]);row[name+"_y"]=float(points[index,1])
            row[name+"_score"]=float(confidence[index])
        rows.append(row);i+=1
    capture.release();output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output,index=False,float_format="%.17g")
    return {"frames":i,"fps":fps,"pose_csv":str(output)}


def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    for name in ["det-config","det-weights","pose-config","pose-weights"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",default="cuda:0");args=p.parse_args()
    print(json.dumps(extract_video(args.input,args.output,args.det_config,args.det_weights,args.pose_config,args.pose_weights,args.device)))

if __name__=="__main__":main()
