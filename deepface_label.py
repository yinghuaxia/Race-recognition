from tqdm import tqdm
from deepface import DeepFace
import json
import cv2
import os
folder_path = r'WIN_20240826_11_07_18_Pro'
dataset_path = folder_path
total_images = sum([len(files) for r, d, files in os.walk(dataset_path)])
with tqdm(total = total_images, desc="Processing Image", unit="image") as pbar:
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        if os.path.isfile(image_path):
            # print(true_label, class_name)
            objs = DeepFace.analyze(
                img_path = image_path,
                enforce_detection=False,
                detector_backend='yolov8',
                actions = ['race'],
            )
            ground_truths = []
            for obj in objs:
                if obj['face_confidence'] == -1:
                    pbar.update(1)
                    continue
                # 獲取 bounding box 相關的座標資訊
                box = obj['region']
                left_top_x = box['x']
                left_top_y = box['y']
                w = box['w']
                h = box['h']
                # confidence = obj['face_confidence']
                confidence = 1
                race = obj['dominant_race']
                if race == 'latino hispanic' or race == 'middle eastern':
                    race = 'indian'
                # 生成 JSON 格式的 ground truth
                ground_truth = [
                    left_top_x, left_top_y, w, h, confidence, race
                ]
                ground_truths.append(ground_truth)
                img = cv2.imread(image_path)
                img = cv2.rectangle(img, (left_top_x, left_top_y), (left_top_x + w, left_top_y + h), (255, 255, 255), 2)
            # 將 ground truth 存入 JSON 檔案
            json_filename = os.path.splitext(image_name)[0] + '.json'
            json_filepath = os.path.join(dataset_path, 'json', json_filename)
            with open(json_filepath, 'w') as json_file:
                new_data = {"bboxes": ground_truths}
                json.dump(new_data, json_file)
            pbar.update(1)
