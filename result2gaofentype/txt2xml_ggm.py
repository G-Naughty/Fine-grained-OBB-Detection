'''
@File  :result2gaofen.py
@Author:OYLH
@Date  :2021/8/28上午10:31
'''
import os
import xml.etree.ElementTree as ET
import copy
import json
import numpy as np

from shapely.geometry import Polygon
# result_path = ''
# img_path = ''
# save_path = ''
# imgs = os.listdir(img_path)
# results = os.listdir(result_path)
# for i,img in enumerate(imgs):
#     save = os.path.join(img_path,img).replace('tif','xml')
#
from xml.dom.minidom import Document
import os

def anchor_finetune(object , min_anchor_size=0, img_size=(1024,1024)):
    img_x, img_y = img_size
    img_bbox = np.asarray([0, 0, img_x, 0, img_x, img_y, 0, img_y])
    img_bbox = Polygon(img_bbox[:8].reshape((4, 2)))   #创建图片大小的bbox对象
    object_bbox = np.asarray(object)
    object_bbox = Polygon(object_bbox[:8].reshape((4, 2)))  # 创建图片大小的bbox对象
    if img_bbox.is_valid and img_bbox.is_valid:
        inter = Polygon(img_bbox).intersection(Polygon(object_bbox))  # 求相交四边形
        if inter.area > min_anchor_size:
            object_bbox = inter.convex_hull.exterior.coords  # 得到相交多边形顶点
            bbox_x, bbox_y = object_bbox.xy
            if len(bbox_x) <= 5:  # 4,3个点
                object_bbox = [int(bbox_x[0]), int(bbox_y[0]), int(bbox_x[1]), int(bbox_y[1]), int(bbox_x[2]),
                               int(bbox_y[2]), int(bbox_x[3]), int(bbox_y[3])]
            elif len(bbox_x) == 6:  #5个点
                point = [[bbox_x[i], bbox_y[i]] for i in range(len(bbox_x) -1)]
                area_max = 0
                for i in range(np.shape(point)[0]):
                    bbox = copy.deepcopy(point)
                    bbox.pop(i)
                    area = Polygon(np.asarray(bbox)).area
                    if area > area_max:
                        bbox_now = copy.deepcopy(bbox)
                        area_max = area
                object_bbox = [float(bbox_now[0][0]), float(bbox_now[0][1]), float(bbox_now[1][0]), float(bbox_now[1][1]),
                               float(bbox_now[2][0]), float(bbox_now[2][1]), float(bbox_now[3][0]), float(bbox_now[3][1])]
            elif len(bbox_x) > 6:
                object_bbox = copy.deepcopy(object)
                for i in range(0, len(object_bbox), 2):
                    if object_bbox[i] < 0:
                        object_bbox[i] = 0
                    if object_bbox[i] > img_x:
                        object_bbox[i] = img_x
                    if object_bbox[i+1] < 0:
                        object_bbox[i+1] = 0
                    if object_bbox[i+1] > img_y:
                        object_bbox[i+1] = img_y
                    object_bbox[i] = int(object_bbox[i])
            object = object_bbox
    return object

def makexml(txtPath,xmlPath): #读取txt路径，xml保存路径，数据集图片所在路径
        # with open('/home/disk/FAIR1M_emsemble/imgs_size.json', 'r') as load_f:
        #     img_size = json.load(load_f)  # print('get all img sizes')
        files = os.listdir(txtPath)
        for i, name in enumerate(files):
          xmlBuilder = Document()
          annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
          xmlBuilder.appendChild(annotation)
          txtFile=open(os.path.join(txt_path,name))
          txtList = txtFile.readlines()
          # img = cv2.imread(picPath+name[0:-4]+".jpg")
          # Pheight,Pwidth,Pdepth=img.shape
          source = xmlBuilder.createElement("source")  # folder标签
          filename = xmlBuilder.createElement('filename')
          filenameContent = xmlBuilder.createTextNode(name[:-4] + '.tif')
          filename.appendChild(filenameContent)
          source.appendChild(filename)
          origin = xmlBuilder.createElement('origin')
          originContent = xmlBuilder.createTextNode('GF2/GF3')
          origin.appendChild(originContent)
          source.appendChild(origin)
          annotation.appendChild(source)

          research = xmlBuilder.createElement("research")  # folder标签
          version = xmlBuilder.createElement('version')
          versionContent = xmlBuilder.createTextNode('1.0')
          version.appendChild(versionContent)
          research.appendChild(version)
          provider = xmlBuilder.createElement('provider')
          providerContent = xmlBuilder.createTextNode('School of team')
          provider.appendChild(providerContent)
          research.appendChild(provider)
          author = xmlBuilder.createElement('author')
          authorContent = xmlBuilder.createTextNode('1111')
          author.appendChild(authorContent)
          research.appendChild(author)
          pluginname = xmlBuilder.createElement('pluginname')
          pluginnameContent = xmlBuilder.createTextNode('FAIR1M')
          pluginname.appendChild(pluginnameContent)
          research.appendChild(pluginname)
          pluginclass = xmlBuilder.createElement('pluginclass')
          pluginclassContent = xmlBuilder.createTextNode('object detection')
          pluginclass.appendChild(pluginclassContent)
          research.appendChild(pluginclass)
          time = xmlBuilder.createElement('time')
          timeContent = xmlBuilder.createTextNode('2021-03')
          time.appendChild(timeContent)
          research.appendChild(time)
          annotation.appendChild(research)
          objects = xmlBuilder.createElement("objects")

          for i in txtList:
             oneline = i.strip().split(" ")
             #  加上anchor finetune
             # bbox = [float(point) for point in oneline[0:8]]
             # bbox=anchor_finetune(bbox, min_anchor_size=0, img_size=img_size[name.strip().split('.txt')[0]])
             # bbox = [str(point) for point in bbox]
             # oneline[0:8] = bbox[0:8]
             object = xmlBuilder.createElement("object")
             coordinate = xmlBuilder.createElement("coordinate")
             coordinateContent = xmlBuilder.createTextNode('pixel')
             coordinate.appendChild(coordinateContent)
             object.appendChild(coordinate)
             type = xmlBuilder.createElement("type")
             typeContent = xmlBuilder.createTextNode('rectangle')
             type.appendChild(typeContent)
             object.appendChild(type)
             description = xmlBuilder.createElement("description")
             descriptionContent = xmlBuilder.createTextNode('None')
             description.appendChild(descriptionContent)
             object.appendChild(description)

             possibleresult = xmlBuilder.createElement("possibleresult")
             objname = xmlBuilder.createElement("name")
            #  objnameContent = xmlBuilder.createTextNode(oneline[0])
             objnameContent = xmlBuilder.createTextNode(oneline[8].replace("_"," "))
             objname.appendChild(objnameContent)
             possibleresult.appendChild(objname)
             probability = xmlBuilder.createElement("probability")
            #  probabilityContent = xmlBuilder.createTextNode(oneline[1])
             probabilityContent = xmlBuilder.createTextNode(oneline[9])
             probability.appendChild(probabilityContent)
             possibleresult.appendChild(probability)
             object.appendChild(possibleresult)

             points = xmlBuilder.createElement("points")
             point = xmlBuilder.createElement("point")
            #  pointContent = xmlBuilder.createTextNode(oneline[2]+','+oneline[3])
             pointContent = xmlBuilder.createTextNode(oneline[0]+','+oneline[1])
             point.appendChild(pointContent)
             points.appendChild(point)
             point = xmlBuilder.createElement("point")
            #  pointContent = xmlBuilder.createTextNode(oneline[4]+','+oneline[5])
             pointContent = xmlBuilder.createTextNode(oneline[2]+','+oneline[3])
             point.appendChild(pointContent)
             points.appendChild(point)
             point = xmlBuilder.createElement("point")
            #  pointContent = xmlBuilder.createTextNode(oneline[6]+','+oneline[7])
             pointContent = xmlBuilder.createTextNode(oneline[4]+','+oneline[5])
             point.appendChild(pointContent)
             points.appendChild(point)
             point = xmlBuilder.createElement("point")
            #  pointContent = xmlBuilder.createTextNode(oneline[8]+','+oneline[9])
             pointContent = xmlBuilder.createTextNode(oneline[6]+','+oneline[7])
             point.appendChild(pointContent)
             points.appendChild(point)
             point = xmlBuilder.createElement("point")
            #  pointContent = xmlBuilder.createTextNode(oneline[2]+','+oneline[3])
             pointContent = xmlBuilder.createTextNode(oneline[0]+','+oneline[1])
             point.appendChild(pointContent)
             points.appendChild(point)
             object.appendChild(points)

             objects.appendChild(object)
          annotation.appendChild(objects)
          f = open(os.path.join(xmlPath,name[:-4])+".xml", 'w')
          xmlBuilder.writexml(f, indent='', newl='\n', addindent='\t', encoding='utf-8')
          f.close()

def result2txt(result_path,save_path):
    results = os.listdir(result_path)
    for j,result in enumerate(results):
        r = open(os.path.join(result_path,result),'r')
        lines = r.readlines()
        for line in lines:
            img_name = line.strip().split(' ')[0]
            w = open(os.path.join(save_path,img_name).replace('tif','txt'),'a')
            w.write(result.split(".")[0].split('_')[1]+','+','.join(line.strip().split(' ')[1:])+'\n')
            w.close()

def filltxt(save_path):
    for i in range(8137):  #8287
        if not os.path.exists(os.path.join(save_path,str(i)+'.xml')):
            xmlBuilder = Document()
            annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
            xmlBuilder.appendChild(annotation)
            # img = cv2.imread(picPath+name[0:-4]+".jpg")
            # Pheight,Pwidth,Pdepth=img.shape
            source = xmlBuilder.createElement("source")  # folder标签
            filename = xmlBuilder.createElement('filename')
            filenameContent = xmlBuilder.createTextNode(str(i) + '.tif')
            filename.appendChild(filenameContent)
            source.appendChild(filename)
            origin = xmlBuilder.createElement('origin')
            originContent = xmlBuilder.createTextNode('GF2/GF3')
            origin.appendChild(originContent)
            source.appendChild(origin)
            annotation.appendChild(source)

            research = xmlBuilder.createElement("research")  # folder标签
            version = xmlBuilder.createElement('version')
            versionContent = xmlBuilder.createTextNode('1.0')
            version.appendChild(versionContent)
            research.appendChild(version)
            provider = xmlBuilder.createElement('provider')
            providerContent = xmlBuilder.createTextNode('School of team')
            provider.appendChild(providerContent)
            research.appendChild(provider)
            author = xmlBuilder.createElement('author')
            authorContent = xmlBuilder.createTextNode('YLSD')
            author.appendChild(authorContent)
            research.appendChild(author)
            pluginname = xmlBuilder.createElement('pluginname')
            pluginnameContent = xmlBuilder.createTextNode('FAIR1M')
            pluginname.appendChild(pluginnameContent)
            research.appendChild(pluginname)
            pluginclass = xmlBuilder.createElement('pluginclass')
            pluginclassContent = xmlBuilder.createTextNode('object detection')
            pluginclass.appendChild(pluginclassContent)
            research.appendChild(pluginclass)
            time = xmlBuilder.createElement('time')
            timeContent = xmlBuilder.createTextNode('2021-03')
            time.appendChild(timeContent)
            research.appendChild(time)
            annotation.appendChild(research)
            objects = xmlBuilder.createElement("objects")
            annotation.appendChild(objects)
            f = open(os.path.join(save_path, str(i)) + ".xml", 'w')
            xmlBuilder.writexml(f, indent='', newl='\n', addindent='\t', encoding='utf-8')
            f.close()



if __name__ == '__main__':
    # txt_path = '/media/hnu/2D97AD940A9AD661/Data/gaofeng/FAIR1M/result_redet_1x/result_txt'
    # save_path = '/media/hnu/2D97AD940A9AD661/Data/gaofeng/FAIR1M/result_redet_1x/test'
    # result_path = '/media/hnu/2D97AD940A9AD661/code/ReDet/work_dirs/ReDet_re50_refpn_1x_gaofen/Task1_results_nms'
    txt_path = "/home/hnu1/GGM/OBBDetection/work_dir/oriented_obb_contrast_catbalance/result_txt"
    save_path = "/home/hnu1/GGM/OBBDetection/work_dir/oriented_obb_contrast_catbalance/test"
    # result_path = '/media/hnu/2D97AD940A9AD661/code/ReDet/work_dirs/ReDet_re50_refpn_1x_gaofen/Task1_results_nms'
    # result2txt(result_path,txt_path)
    makexml(txt_path,save_path)
    filltxt(save_path)
