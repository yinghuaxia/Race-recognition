import numpy as np
import cv2
import random
from PIL import Image
from opencv_transforms import transforms

def FlipImg(img,bboxes):
    img_w=np.shape(img)[1]
    img=cv2.flip(img,1)
    for i,bbox in enumerate(bboxes):
        bbox[0]=img_w-bbox[0]-bbox[2]
        bboxes[i]=bbox
    return img,bboxes
def CropImg(img,bboxes):
    h,w=np.shape(img)[:2]
    min_x=min(list(map(lambda x:x[0],bboxes)))
    max_x=max(list(map(lambda x:x[0]+x[2]-1,bboxes)))
    min_y=min(list(map(lambda x:x[1],bboxes)))
    max_y=max(list(map(lambda x:x[1]+x[3]-1,bboxes)))
    max_l_trans=min_x
    max_u_trans=min_y
    max_r_trans=w-max_x
    max_d_trans=h-max_y

    crop_xmin=max(0,int(min_x-random.uniform(0,max_l_trans)))
    crop_ymin=max(0,int(min_y-random.uniform(0,max_u_trans)))
    crop_xmax=max(w,int(max_x+random.uniform(0,max_r_trans)))
    crop_ymax=max(h,int(max_y+random.uniform(0,max_d_trans)))

    img=img[crop_ymin:crop_ymax,crop_xmin:crop_xmax]
    bboxes=list(map(lambda x:[float((x[0]-crop_xmin)),
                              float((x[1]-crop_ymin)),
                              float(x[2]),
                              float(x[3]),
                              float(x[4]),
                              x[5]],bboxes))
    return img,bboxes
def AffineImg(img,bboxes):
    h,w=np.shape(img)[:2]
    min_x=min(list(map(lambda x:x[0],bboxes)))
    max_x=max(list(map(lambda x:x[0]+x[2]-1,bboxes)))
    min_y=min(list(map(lambda x:x[1],bboxes)))
    max_y=max(list(map(lambda x:x[1]+x[3]-1,bboxes)))
    max_l_trans=min_x
    max_u_trans=min_y
    max_r_trans=w-max_x
    max_d_trans=h-max_y

    tx=random.uniform(-(max_l_trans-1),(max_r_trans-1))
    ty=random.uniform(-(max_u_trans-1),(max_d_trans-1))

    M=np.array([[1,0,tx],[0,1,ty]])
    img=cv2.warpAffine(img,M,(w,h))
    bboxes=list(map(lambda x:[float(x[0]+tx),float(x[1]+ty),float(x[2]),float(x[3]),float(x[4]),x[5]],bboxes))
    return img,bboxes

def RotateImg(img,bboxes,border_value=(128,128,128)):
    rotate_degree=np.random.uniform(low=-10,high=10)
    h,w=img.shape[:2]
    # Compute the rotation matrix.
    M=cv2.getRotationMatrix2D(center=(w/2,h/2),
                              angle=rotate_degree,
                              scale=1)
    # Get the sine and cosine from the rotation matrix.
    abs_cos_angle=np.abs(M[0, 0])
    abs_sin_angle=np.abs(M[0, 1])
    # Compute the new bounding dimensions of the image.
    new_w=int(h*abs_sin_angle+w*abs_cos_angle)
    new_h=int(h*abs_cos_angle+w*abs_sin_angle)
    # Adjust the rotation matrix to take into account the translation.
    M[0,2]+=new_w//2-w//2
    M[1,2]+=new_h//2-h//2 
    # Rotate the image.
    img=cv2.warpAffine(img,M=M,dsize=(new_w, new_h),flags=cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_CONSTANT,borderValue=border_value)
    new_bboxes=[]
    for bbox in bboxes:
        x,y,w,h,wht,label=bbox
        x1,y1,x2,y2=[x,y,x+w,y+h]
        points=M.dot([[x1,x2,x1,x2],[y1,y2,y2,y1],[1,1,1,1]])
        # Extract the min and max corners again.
        min_xy=np.sort(points,axis=1)[:,:2]
        min_x=np.mean(min_xy[0])
        min_y=np.mean(min_xy[1])
        max_xy=np.sort(points,axis=1)[:,2:]
        max_x=np.mean(max_xy[0])
        max_y=np.mean(max_xy[1])
        bbox=[float(min_x),float(min_y),float(max_x-min_x),float(max_y-min_y),float(wht),label]
        new_bboxes.append(bbox)
    return img,new_bboxes

def BboxShift(img_hw,bbox,shift_range=0.1):
    x,y,w,h,wht,label=bbox
    w_shift_range_b=int(w*shift_range*(-0.5))
    w_shift_range_e=w_shift_range_b*(-1)
    h_shift_range_b=int(h*shift_range*(-0.5))
    h_shift_range_e=h_shift_range_b*(-1)
    x=x+random.randint(w_shift_range_b,w_shift_range_e)
    y=y+random.randint(h_shift_range_b,h_shift_range_e)
    w=w+random.randint(w_shift_range_b,w_shift_range_e)
    h=h+random.randint(h_shift_range_b,h_shift_range_e)
    if(x<0):x=0
    if(y<0):y=0
    if(x+w-1>=img_hw[1]):w=img_hw[1]-x-1
    if(y+h-1>=img_hw[0]):h=img_hw[0]-y-1
    bbox=[float(x),float(y),float(w),float(h),float(wht),label]
    return bbox

def BboxesShift(img,bboxes,shift_range=0.1):
    img_hw=np.shape(img)[:2]
    bboxes=list(map(lambda x:BboxShift(img_hw,x,shift_range),bboxes))
    return img,bboxes

def convert_to_grayscale(img, bboxes):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return img, bboxes

def reverse(img, bboxes):   
    return 255 - img, bboxes

def random_adjust_color_temperature(image, bboxes):
    # 隨機生成 RGB 通道縮放因子
    r_range=(0.8, 1.2)
    g_range=(0.8, 1.2)
    b_range=(0.8, 1.2)
    r_factor = random.uniform(*r_range)
    g_factor = random.uniform(*g_range)
    b_factor = random.uniform(*b_range)
   
    # 調整圖像色溫
    return adjust_color_temperature(image, r_factor, g_factor, b_factor), bboxes
 
def adjust_color_temperature(image, r_factor, g_factor, b_factor):
    # 將圖像轉換為浮點數類型以進行精確計算
    image = image.astype(np.float32)
   
    # 創建調整矩陣
    adjustment_matrix = np.array([r_factor, g_factor, b_factor])
   
    # 應用調整矩陣
    adjusted_image = cv2.multiply(image, adjustment_matrix)
   
    # 確保像素值在合理範圍內
    adjusted_image = np.clip(adjusted_image, 0, 255)
   
    # 轉回8位無符號整數
    return adjusted_image.astype(np.uint8)
def adjust_saturation(image, bboxes):
    # 將圖像從 BGR 轉換為 HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
   
    # 取得 HSV 通道
    h, s, v = cv2.split(hsv)
   
    # 調整飽和度通道，scale 是飽和度調整的比例
    # 例如，scale = 1.5 會將飽和度增加 50%
    scale = 0
    while scale == 0:
        scale = random.randint(500, 1500)
    scale = scale / 1000
    s = np.clip(s * scale, 0, 255).astype(np.uint8)
   
    # 合併調整後的 HSV 通道
    hsv_adjusted = cv2.merge([h, s, v])
   
    # 將圖像從 HSV 轉換回 BGR
    bgr_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
   
    return bgr_adjusted, bboxes

def add_gaussian_noise(image, bboxes):
    mean = 0
    std = 15
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image, bboxes

def gaussian_blur(image, bboxes):
    ksize = random.choice([(3, 3), (5, 5), (7, 7)])
    blurred_img = cv2.GaussianBlur(image, ksize, 0)
    return blurred_img, bboxes

def random_brightness(image, bboxes):
    min_factor=0.3
    max_factor=1.3
    brightness_factor = random.uniform(min_factor, max_factor)
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return adjusted_image, bboxes

def BoxesAugment(img,bboxes):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if(random.random()>0.5):
        img,bboxes=AffineImg(img,bboxes)
    if(random.random()>0.5):
        img,bboxes=FlipImg(img,bboxes)
    if(random.random()>0.5):
        img,bboxes=BboxesShift(img,bboxes)
    if(random.random()>0.5):
        img,bboxes=CropImg(img,bboxes)
    if(random.random()>0.5):
        img,bboxes=RotateImg(img,bboxes)
    # if(random.random()>0.5):
    #     img,bboxes=random_brightness(img,bboxes)
    if(random.random()>0.4):
        img,bboxes=random_adjust_color_temperature(img,bboxes)
    if(random.random()>0.4):
        img,bboxes=adjust_saturation(img,bboxes)
    if(random.random()>0.7):
        img,bboxes=add_gaussian_noise(img,bboxes)
    if(random.random()>0.7):
        img,bboxes=gaussian_blur(img,bboxes)

    return img,bboxes