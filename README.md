# Deep Learning Quantization
## Board: Google Coral Dev Board  

## TensorFlow Lite delegate
By default, TensorFlow Lite executes each model on the CPU. In order to use TensorFlow Lite on google coral dev board, you should use **TensorFlow Lite delegate**.

## TensorFlow Lite converter
The TensorFlow Lite converter takes a TensorFlow model and generates a TensorFlow Lite model (an optimized FlatBuffer format identified by the .tflite file extension)
<img src=https://www.tensorflow.org/lite/images/convert/convert.png>

## Reference
https://www.tensorflow.org/lite/guide?hl=ko  
https://coral.ai/docs/
