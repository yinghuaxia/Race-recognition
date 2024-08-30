import cv2
import numpy as np
_labels2bgr=None

def InitLabels2bgrDict(labels):
    global _labels2bgr
    if(_labels2bgr!=None):
        return _labels2bgr
    _labels2bgr={}
    for label in labels:
        bgr=np.random.randint(0,255,[3])
        bgr=(bgr[0].item(),bgr[1].item(),bgr[2].item())
        _labels2bgr[label]=bgr
    return _labels2bgr

def DrawMainBox(img,xywh,bgr=(0,255,0),thickness=3):
    x=int(round(xywh[0]))
    y=int(round(xywh[1]))
    w=int(round(xywh[2]))
    h=int(round(xywh[3]))
    x1y1=(x,y)
    x2y2=(x+w,y+h)
    cv2.rectangle(img,x1y1,x2y2,bgr,thickness)
    return img

def DrawLabelBox(img,label,score,xy,bgr=(0,255,0)):
    score=str(score*100)[:2]+"%"
    text=label+": "+score
    x=int(round(xy[0]))
    y=int(round(xy[1]-5))
    ret,baseline=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1.2,2)
    cv2.rectangle(img,(x,y-ret[1]-2),(x+ret[0],y+5),bgr,-1)
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
    return img

def DrawStatus(img,status_str="None",bgr=(0,0,255)):
    cv2.putText(img,"Status: "+status_str,(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,bgr,1,cv2.LINE_AA)
    return img


def Drawing(img,pred_msg,score_thres=0.5,thickness=3):
    global _labels2bgr
    if(np.shape(pred_msg)[0]==0):return img
    scores=pred_msg[...,4].astype("float")
    score_indexs=scores>score_thres
    pred_msg=pred_msg[score_indexs]
    pred_boxes=pred_msg[...,0:4].astype("float")
    scores=pred_msg[...,4].astype("float")
    labels=pred_msg[...,5]

    for i,box in enumerate(pred_boxes):
        # print(f"i: {i}, box: {box}")
        label=labels[i]
        # print(f"label: {label}")
        # bgr=_labels2bgr[label]
        # bgr = label
        score=scores[i]
        img=DrawMainBox(img,box,thickness=thickness)
        img=DrawLabelBox(img,label,score,box[:2])
    return img