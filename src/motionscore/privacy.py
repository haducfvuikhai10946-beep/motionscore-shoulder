"""Deterministic opaque head redaction with scene context retained."""
from __future__ import annotations
import numpy as np


def redact_frame(frame,xy,background_boxes=()):
    """xy: eight joints ordered shoulder, elbow, wrist, hip for left then right.

    Additional people are covered by conservative background boxes when provided.
    Review every exported recording for privacy before public distribution.
    """
    import cv2
    xy=np.asarray(xy,float)
    if xy.shape!=(8,2) or not np.isfinite(xy).all():
        return np.full_like(frame,255),{"fallback_full_mask":True}
    h,w=frame.shape[:2];shoulders=xy[[0,4]]
    width=float(np.linalg.norm(shoulders[0]-shoulders[1]))
    if width<10:
        return np.full_like(frame,255),{"fallback_full_mask":True}
    output=frame.copy()
    for box in background_boxes:
        bx1,by1,bx2,by2=np.round(box).astype(int)
        cv2.rectangle(output,(max(0,bx1),max(0,by1)),(min(w-1,bx2),min(h-1,by2)),(255,255,255),-1)
    cx=shoulders[:,0].mean()
    x1=max(0,int(cx-.65*width));x2=min(w,int(cx+.65*width))
    y1=max(0,int(shoulders[:,1].mean()-1.65*width))
    y2=min(h,max(0,int(shoulders[:,1].min()-.06*width)))
    if y2>y1 and x2>x1:output[y1:y2+1,x1:x2+1]=255
    else:output[:]=255
    covered=(xy[:,0]>=x1)&(xy[:,0]<=x2)&(xy[:,1]>=y1)&(xy[:,1]<=y2)
    return output,{"fallback_full_mask":False,"head_box":[x1,y1,x2,y2],
                   "joint_centers_in_head_mask":int(covered.sum())}
