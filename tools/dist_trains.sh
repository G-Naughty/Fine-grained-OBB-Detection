#!/usr/bin/env bash

GPU_ID=$1

python tools/train.py --model_name 46_oriented_obb_Passenger --gpu-ids ${GPU_ID}
python tools/train.py --model_name 47_oriented_obb_Fishing_Boat --gpu-ids ${GPU_ID}
python tools/train.py --model_name 48_oriented_obb_Dry_Cargo_Ship --gpu-ids ${GPU_ID}
python tools/train.py --model_name 49_oriented_obb_Liquid_Cargo_Ship --gpu-ids ${GPU_ID}

CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/test.py --model_name 46_oriented_obb_Passenger
CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/test.py --model_name 47_oriented_obb_Fishing_Boat
CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/test.py --model_name 48_oriented_obb_Dry_Cargo_Ship
CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/test.py --model_name 49_oriented_obb_Liquid_Cargo_Ship

python tools/test_merge.py --model_name 46_oriented_obb_Passenger
python tools/test_merge.py --model_name 47_oriented_obb_Fishing_Boat
python tools/test_merge.py --model_name 48_oriented_obb_Dry_Cargo_Ship
python tools/test_merge.py --model_name 49_oriented_obb_Liquid_Cargo_Ship

python result2gaofentype/pkl2txt.py --model_name 46_oriented_obb_Passenger --classes Passenger_Ship
python result2gaofentype/txt2xml.py --model_name 46_oriented_obb_Passenger

python result2gaofentype/pkl2txt.py --model_name 47_oriented_obb_Fishing_Boat --classes Fishing_Boat
python result2gaofentype/txt2xml.py --model_name 47_oriented_obb_Fishing_Boat

python result2gaofentype/pkl2txt.py --model_name 48_oriented_obb_Dry_Cargo_Ship --classes Dry_Cargo_Ship
python result2gaofentype/txt2xml.py --model_name 48_oriented_obb_Dry_Cargo_Ship

python result2gaofentype/pkl2txt.py --model_name 49_oriented_obb_Liquid_Cargo_Ship --classes Liquid_Cargo_Ship
python result2gaofentype/txt2xml.py --model_name 49_oriented_obb_Liquid_Cargo_Ship
