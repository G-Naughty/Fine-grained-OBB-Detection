import BboxToolkit as bt
import pickle
import copy
import numpy as np
from tqdm import tqdm
import os
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='To config a model')
    parser.add_argument('--model_name', default='38_oriented_obb_c919')
    args = parser.parse_args()
    return args

def hbb2obb(hbox):
    x1,y1,x2,y2=hbox
    obox= x1,y1,x2,y1,x2,y2,x1,y2
    return obox

args = parse_args()

path1 = "/home/hnu1/OBBDetection/work_dir/GGM/" + args.model_name + "/dets.pkl"
classes=['C919'
    # 'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220',
    # 'A321', 'A330', 'A350', 'ARJ21', 'other-airplane',
    # 'Passenger_Ship', 'Motorboat', 'Fishing_Boat', 'Tugboat', 'Engineering_Ship',
    # 'Liquid_Cargo_Ship', 'Dry_Cargo_Ship', 'Warship', 'other-ship',
    # 'Small_Car', 'Bus', 'Cargo_Truck', 'Dump_Truck', 'Van',
    # 'Trailer', 'Tractor', 'Excavator', 'Truck_Tractor', 'other-vehicle',
    # 'Basketball_Court', 'Tennis_Court', 'Football_Field', 'Baseball_Field',
    # 'Intersection', 'Roundabout', 'Bridge'

]
# path2 = "/home/hnu1/OBBDetection/data/FAIR1M_ss/test/annfiles/ori_annfile.pkl"  #
# with open(path2, 'rb') as f:
#     data2 = pickle.load(f)

with open(path1, 'rb') as f:
    obbdets = pickle.load(f)
    polydets = copy.deepcopy(obbdets)
for i in tqdm(range(len(obbdets))):
    for j in range(len(obbdets[0][1])):
        data = obbdets[i][1][j]
        if data.size != 0:
            polys = []
            for k in range(len(data)):
                poly = hbb2obb(data[k][0:4])
                poly = np.append(poly, data[k][4])
                polys.append(poly)
        else:
            polys = []
        polydets[i][1][j] = polys

# try:
#     shutil.rmtree("/home/hnu1/OBBDetection/work_dir/GGM/" + model_name + "/result")
# except:
#     os.makedirs("/home/hnu1/OBBDetection/work_dir/GGM/" + model_name + "/result/", exist_ok=True)

try:
    shutil.rmtree("/home/hnu1/OBBDetection/work_dir/GGM/" + args.model_name + "/result_txt")
except:
    pass
os.makedirs("/home/hnu1/OBBDetection/work_dir/GGM/" + args.model_name + "/result_txt/", exist_ok=True)
try:
    shutil.rmtree("/home/hnu1/OBBDetection/work_dir/GGM/" + args.model_name + "/test")
except:
    pass
os.makedirs("/home/hnu1/OBBDetection/work_dir/GGM/" + args.model_name + "/test/", exist_ok=True)

savepath = "/home/hnu1/OBBDetection/work_dir/GGM/" + args.model_name + "/result_txt/"
for i in tqdm(range(len(polydets))):
    txtfile = savepath + polydets[i][0] + ".txt"
    f = open(txtfile, "w")
    for j in range(len(polydets[0][1])):
        if polydets[i][1][j] != []:
            for k in range(len(polydets[i][1][j])):
                f.write(str(polydets[i][1][j][k][0]) + " " +
                        str(polydets[i][1][j][k][1]) + " " +
                        str(polydets[i][1][j][k][2]) + " " +
                        str(polydets[i][1][j][k][3]) + " " +
                        str(polydets[i][1][j][k][4]) + " " +
                        str(polydets[i][1][j][k][5]) + " " +
                        str(polydets[i][1][j][k][6]) + " " +
                        str(polydets[i][1][j][k][7]) + " " +
                        str(classes[j]) + " " +
                        # str(data2["cls"][j]) + " " +
                        str(polydets[i][1][j][k][8]) + "\n")
    f.close()
