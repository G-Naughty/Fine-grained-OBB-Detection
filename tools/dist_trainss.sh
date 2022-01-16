#!/usr/bin/env bash

#GPU_ID=$1
#
#python tools/train.py --model_name 42_oriented_obb_Trailer --gpu-ids ${GPU_ID}
#python tools/test.py --model_name 42_oriented_obb_Trailer
#python tools/train.py --model_name 41_oriented_obb_C919 --gpu-ids ${GPU_ID}
#python tools/test.py --model_name 41_oriented_obb_C919
#python tools/train.py --model_name 44_oriented_obb_Tugboat --gpu-ids ${GPU_ID}
#python tools/test.py --model_name 44_oriented_obb_Tugboat
#python tools/train.py --model_name 43_oriented_obb_Excavator --gpu-ids ${GPU_ID}
#python tools/test.py --model_name 43_oriented_obb_Excavator
#python tools/train.py --model_name 40_oriented_obb_ARJ21 --gpu-ids ${GPU_ID}
#python tools/test.py --model_name 40_oriented_obb_ARJ21

python result2gaofentype/pkl2txt.py --model_name 42_oriented_obb_Trailer --classes Trailer
python result2gaofentype/txt2xml.py --model_name 42_oriented_obb_Trailer

python result2gaofentype/pkl2txt.py --model_name 43_oriented_obb_Excavator --classes Excavator
python result2gaofentype/txt2xml.py --model_name 43_oriented_obb_Excavator

python result2gaofentype/pkl2txt.py --model_name 44_oriented_obb_Tugboat --classes Tugboat
python result2gaofentype/txt2xml.py --model_name 44_oriented_obb_Tugboat

python result2gaofentype/pkl2txt.py --model_name 40_oriented_obb_ARJ21 --classes ARJ21
python result2gaofentype/txt2xml.py --model_name 40_oriented_obb_ARJ21

python result2gaofentype/pkl2txt.py --model_name 41_oriented_obb_C919 --classes C919
python result2gaofentype/txt2xml.py --model_name 41_oriented_obb_C919

