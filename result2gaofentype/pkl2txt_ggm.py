import BboxToolkit as bt
import pickle
import copy
import numpy as np
path1="/home/hnu1/GGM/OBBDetection/work_dir/oriented_obb_contrast_catbalance/dets.pkl"
path2="/home/hnu1/GGM/OBBDetection/data/FaIR1M/test/annfiles/ori_annfile.pkl"#
with open(path2,'rb') as f:          #/home/disk/FAIR1M_1000_split/val/annfiles/ori_annfile.pkl
    data2 = pickle.load(f)

with open(path1,'rb') as f:
    obbdets = pickle.load(f)
    polydets=copy.deepcopy(obbdets)
for i in range(len(obbdets)):
    for j in range(len(obbdets[0][1])):
        data=obbdets[i][1][j]
        if data.size!= 0:
            polys=[]
            for k in range(len(data)):
                poly = bt.obb2poly(data[k][0:5])
                poly=np.append(poly,data[k][5])
                polys.append(poly)
        else:
            polys=[]
        polydets[i][1][j]=polys

savepath="/home/hnu1/GGM/OBBDetection/work_dir/oriented_obb_contrast_catbalance/result_txt/"
for i in range(len(polydets)):
    txtfile=savepath+polydets[i][0]+".txt"
    f = open(txtfile, "w")
    for j in range(len(polydets[0][1])):
        if polydets[i][1][j]!=[]:
            for k in range(len(polydets[i][1][j])):
                f.write(str(polydets[i][1][j][k][0])+" "+
                        str(polydets[i][1][j][k][1])+" "+
                        str(polydets[i][1][j][k][2])+" "+
                        str(polydets[i][1][j][k][3])+" "+
                        str(polydets[i][1][j][k][4])+" "+
                        str(polydets[i][1][j][k][5])+" "+
                        str(polydets[i][1][j][k][6])+" "+
                        str(polydets[i][1][j][k][7])+" "+
                        str(data2["cls"][j])+" "+
                        str(polydets[i][1][j][k][8])+"\n")
    f.close()