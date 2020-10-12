#!/bin/bash
echo "ssd mobilenet v1 coco post quantization: compiled for edge TPU"
for ((i=0;i<5;i++))
do
	python3 detect_image.py --model models/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite --labels models/coco_labels.txt --input images/grace_hopper.bmp --output images/grace_hopper_processed_edge_tpu.bmp
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: not compiled for edge TPU"
for ((i=0;i<5;i++))
do
	python3 detect_image.py --model models/ssd_mobilenet_v1_coco_quant_postprocess.tflite --labels models/coco_labels.txt --input images/grace_hopper.bmp --output images/grace_hopper_processed.bmp
done

