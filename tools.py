import os
import cv2
import json

dataset_path = r'C:\Users\HW PA\Desktop\20240826\WIN_20240826_11_07_18_Pro'
json_folder = os.path.join(dataset_path, 'json')
# dataset_path = r'C:\Users\HW PA\Desktop\skin_dataset\Mix_20231122_5_Frame'
imgs = os.listdir(dataset_path)

drawing = False
start_point = (0, 0)
end_point = (0, 0)
current_img_idx = 0

def show_bboxes(img, bboxes):
    for box in bboxes:
        bbox_class = box[-1]
        box[0] = round(float(box[0]))
        box[1] = round(float(box[1]))
        box[2] = round(float(box[2]))
        box[3] = round(float(box[3]))
        x, y, w, h = map(int, box[:4])
        if bbox_class == 'asian':
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        elif bbox_class == 'white':
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif bbox_class == 'black':
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        elif bbox_class == 'indian':
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        else:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        img = cv2.putText(img, bbox_class, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return img

def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, img
    # print(f"param: {bboxes}")
    bboxes, json_path, data = param
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            img_copy = img.copy()
            cv2.rectangle(img_copy, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow('Edit Bounding Boxes', img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
        class_name = input("Enter the class for this bounding box: ")
        
        new_bbox = [start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1], 1, class_name]
        bboxes.append(new_bbox)
        data['bboxes'] = bboxes
        print(data)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

def edit_bbox_class(img, bboxes):
    idx = 0
    while True:
        img_copy = img.copy()
        if len(bboxes) == 0:
            break
        x, y, w, h = map(int, bboxes[idx][:4])
        bbox_class = bboxes[idx][-1]
        img_copy = cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
        img_copy = cv2.putText(img_copy, f'Editing: {bbox_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('Edit Bounding Boxes', img_copy)
        
        key = cv2.waitKey(0)
        if key == ord('d'):  # Next bounding box
            idx = (idx + 1) % len(bboxes)
        elif key == ord('a'):  # Previous bounding box
            idx = (idx - 1) % len(bboxes)
        elif key == ord('e'):  # Edit current bounding box class
            new_class = input(f"Enter new class for bounding box {idx + 1} (current: {bbox_class}): ")
            bboxes[idx][-1] = new_class
        elif key == ord('x'):  # Delete current bounding box
            del bboxes[idx]
            if len(bboxes) == 0:
                break
            idx = idx % len(bboxes)
        elif key == ord('q'):  # Quit editing
            break
    return bboxes

i = 0
# print(len(imgs))
while i < len(imgs):
    i += 1
    print(f"i:{i}, img name: {imgs[i]}")
    if not imgs[i].lower().endswith(('jpg', 'jpeg', 'png')):
        continue
    
    json_name = imgs[i].rsplit('.', 1)[0] + '.json'
    img_path = os.path.join(dataset_path, imgs[i])
    json_path = os.path.join(json_folder, json_name)
    if not os.path.isfile(json_path):
        os.remove(img_path)
        continue

    img = cv2.imread(img_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
        bboxes = data.get('bboxes', [])
    # if len(bboxes) > 1:
    img_with_bboxes = show_bboxes(img, bboxes)
    cv2.imshow('Edit Bounding Boxes', img_with_bboxes)
    cv2.setMouseCallback('Edit Bounding Boxes', draw_rectangle, param = (bboxes, json_path, data))
    while True:
        key = cv2.waitKey(0)
        if key == ord('e'):
            bboxes = edit_bbox_class(img, bboxes)
            data['bboxes'] = bboxes
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
            break
        elif key == ord('n'):
            break
        elif key == ord('b'):
            i -= 2
            break
        elif key == ord('d'):
            os.remove(img_path)
            if os.path.isfile(json_path):
                os.remove(json_path)
            break
    cv2.destroyAllWindows()
