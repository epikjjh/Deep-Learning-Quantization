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

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: 32bit float"
for ((i=0;i<5;i++))
do
	python3 detect_image.py --model models/32_float_model.tflite --labels models/coco_labels.txt --input images/grace_hopper_640.bmp --output images/grace_hopper_processed_32_bit.bmp
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: dynamic range (representative dataset: None)"
for ((i=0;i<5;i++))
do
	python3 detect_image.py --model models/dynamic_model.tflite --labels models/coco_labels.txt --input images/grace_hopper_640.bmp --output images/grace_hopper_processed_dynamic.bmp
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: dynamic range (representative dataset: Random)"
for ((i=0;i<5;i++))
do
	python3 detect_image.py --model models/dynamic_model_rep_random.tflite --labels models/coco_labels.txt --input images/grace_hopper_640.bmp --output images/grace_hopper_processed_dynamic_random.bmp
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: dynamic range (representative dataset: COCO)"
for ((i=0;i<5;i++))
do
	python3 detect_image.py --model models/dynamic_model_rep_coco.tflite --labels models/coco_labels.txt --input images/grace_hopper_640.bmp --output images/grace_hopper_processed_dynamic_coco.bmp
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: 16bit float (representative dataset: Random)"
for ((i=0;i<5;i++))
do
	python3 detect_image.py --model models/16_float_model_rep_random.tflite --labels models/coco_labels.txt --input images/grace_hopper_640.bmp --output images/grace_hopper_processed_16_bit_random.bmp
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: 16bit float (representative dataset: COCO)"
for ((i=0;i<5;i++))
do
	python3 detect_image.py --model models/16_float_model_rep_coco.tflite --labels models/coco_labels.txt --input images/grace_hopper_640.bmp --output images/grace_hopper_processed_16_bit_coco.bmp
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: 8bit int (representative dataset: Random)"
for ((i=0;i<5;i++))
do
	python3 detect_image.py --model models/8_int_model_rep_random.tflite --labels models/coco_labels.txt --input images/grace_hopper_640.bmp --output images/grace_hopper_processed_8_bit_random.bmp
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: 8bit int (representative dataset: COCO)"
for ((i=0;i<5;i++))
do
	python3 detect_image.py --model models/8_int_model_rep_coco.tflite --labels models/coco_labels.txt --input images/grace_hopper_640.bmp --output images/grace_hopper_processed_8_bit_coco.bmp
done

