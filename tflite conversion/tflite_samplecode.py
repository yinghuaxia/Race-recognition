import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2
# from yolov3.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
import os
# from yolov3.configs import *
# from yolov3.utils import read_class_names, image_preprocess
# from yolov3.dataset import Dataset
# from yolov3.yolov4 import Create_Yolo, compute_loss
# from yolov3.utils import load_yolo_weights
# from yolov3.configs import *
from CSLYOLO import CSLYOLO

# flags.DEFINE_string('weights', './checkpoints_176_112/yolov4_custom_Tiny/saved_model', 'path to weights file')
flags.DEFINE_string('weights', r'C:\Users\HW PA\Jenny_temp/weights/saved_model', 'path to weights file')
# flags.DEFINE_string('output', './checkpoints_176_112/yolov4_custom_Tiny.tflite', 'path to output')
flags.DEFINE_string('output', r'C:\Users\HW PA\Jenny_temp/weights/saved_model/v20.tflite', 'path to output')
# flags.DEFINE_integer('input_width', 112, 'path to output')
# flags.DEFINE_integer('input_height', 176, 'path to output')
flags.DEFINE_integer('input_width', 288, 'path to output')
flags.DEFINE_integer('input_height', 208, 'path to output')
flags.DEFINE_string('quantize_mode', 'int8', 'quantize mode (int8, float16, float32)')
flags.DEFINE_string('dataset', "./output.txt", 'path to dataset')

def representative_data_gen():
  fimage = open(FLAGS.dataset).read().split('/n')
  for input_value in range(len(fimage)):

    if os.path.exists(fimage[input_value]):
      original_image=cv2.imread(fimage[input_value])
      original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
      image_data = image_preprocess(np.copy(original_image), [FLAGS.input_width, FLAGS.input_height])
      img_in = image_data[np.newaxis, ...].astype(np.float32)
      print("calibration image {}".format(fimage[input_value]))
      yield [img_in]
    else:
      continue

def save_model_to_savedmodel(model, save_directory):
    # 将模型保存为 SavedModel 格式
    tf.saved_model.save(model, save_directory)
    print(f"模型已成功保存为 SavedModel 格式到: {save_directory}")

def save_tflite():
  TRAIN_CHECKPOINTS_FOLDER = r'C:\Users\HW PA\Jenny_temp'
  TRAIN_CLASSES = 4
  # save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'yolov4_custom_Tiny')
  save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'weights')
  mAP_model = CSLYOLO(input_shape = (FLAGS.input_height, FLAGS.input_width, 3), labels_len = TRAIN_CLASSES) # create second model to measure mAP
  mAP_model.load_weights(os.path.join(save_directory, 'v20.hdf5')) # use keras weights

  # 保存模型为 .pb 文件
  if save_directory:
    saved_model_directory = os.path.join(save_directory, "saved_model")
    save_model_to_savedmodel(mAP_model, saved_model_directory)

  converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.weights)
  converter.target_spec.supported_ops = [
     tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
  ]
  if FLAGS.quantize_mode == 'float16':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    
  elif FLAGS.quantize_mode == 'int8':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

  tflite_model = converter.convert()
  open(FLAGS.output, 'wb').write(tflite_model)

  logging.info("model saved to: {}".format(FLAGS.output))

def demo():
  interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
  interpreter.allocate_tensors()
  logging.info('tflite model loaded')

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  # Get all tensor details
  all_tensor_details = interpreter.get_tensor_details()

  # Iterate through tensor details
  for tensor_detail in all_tensor_details:
      print("Name:", tensor_detail["name"])
      print("Index:", tensor_detail["index"])
      print("Shape:", tensor_detail["shape"])
      print("Data Type:", tensor_detail["dtype"])
      print("Quantization Parameters:", tensor_detail["quantization"])

  input_shape = input_details[0]['shape']

  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

  #print(output_data)

def main(_argv):
  save_tflite()
  demo()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


