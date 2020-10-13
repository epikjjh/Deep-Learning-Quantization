import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import time
import argparse

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def inference(model, input, output):
  interpreter = tflite.Interpreter(
      model_path=model,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  interpreter.allocate_tensors()

  def draw_rect(image, box):
      y_min = int(max(1, (box[0] * height)))
      x_min = int(max(1, (box[1] * width)))
      y_max = int(min(height, (box[2] * height)))
      x_max = int(min(width, (box[3] * width)))
      
      # draw a rectangle on the image
      cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  img = cv2.imread(input)
  new_img = cv2.resize(img, (width, height))

  if floating_model:
    new_img = new_img.astype(np.float32, copy=False)
    new_img = new_img / 255.0
    new_img = (np.float32(new_img) - 127.5) / 127.5

  interpreter.set_tensor(input_details[0]['index'], [new_img])

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  rects = interpreter.get_tensor(output_details[0]['index'])

  scores = interpreter.get_tensor(output_details[2]['index'])
      
  for index, score in enumerate(scores[0]):
    if score > 0.5:
      draw_rect(new_img,rects[0][index])
            
  new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
  cv2.imwrite(output, new_img)
  print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
  
if __name__ == '__main__':
   parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-i', '--input', required=True,
                      help='File path of image to process.')
  parser.add_argument('-o', '--output',
                      help='File path for the result image with annotations')
  args = parser.parse_args()
  inference(args.model, args.input, args.output)
  
