# Deep Learning Quantization
## Board: Google Coral Dev Board  

## TensorFlow Lite delegate
By default, TensorFlow Lite executes each model on the CPU. In order to use TensorFlow Lite on google coral dev board, you should use **TensorFlow Lite delegate**.
<br></br>

## TensorFlow Lite converter
The TensorFlow Lite converter takes a TensorFlow model and generates a TensorFlow Lite model (an optimized FlatBuffer format identified by the .tflite file extension)  
<img src=https://www.tensorflow.org/lite/images/convert/convert.png>
<br></br>

## Load TensorFlow Lite and run an inference
~~~python
interpreter = tflite.Interpreter(model_path,
  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
~~~
The file passed to load_delegate() is the Edge TPU runtime library, and you installed it when you first set up your device. The filename you must use here depends on your host operating system.

## Reference
https://www.tensorflow.org/lite/guide?hl=ko  
https://coral.ai/docs/
