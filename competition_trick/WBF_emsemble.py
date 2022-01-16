import cv2
import os
import math
import argparse
import codecs
import copy
import json
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
from xml.dom.minidom import Document
from xml.dom import minidom
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
#from weighted_boxes_fusion.ensemble_boxes import *
#使用gpu计算iou可大大提高集成速度
def parse_args():
    parser = argparse.ArgumentParser(description='results style change')#'/home/disk/FAIR1M_emsemble/OBB_epoch12_0.05_41','/home/disk/FAIR1M_emsemble/redet_epoch24_0.001_45'
    parser.add_argument('--models_result', help='所有模型结果的路径列表', default=['/home/disk/FAIR1M_emsemble/OBB_epoch12_0.001_43',
                                                                        '/home/disk/FAIR1M_emsemble/redet_epoch24_0.001_45'
])
    parser.add_argument('--save_path', help='ensemble后模型结果列表', default='/home/disk/FAIR1M_emsemble/emsemble')
    parser.add_argument('--weights', help='各模型权重', default=[1, 1])
    parser.add_argument('--IOU_threshold', help='IOU阈值', default=0.5)  #0.7
    parser.add_argument('--conf_threshold', help='置信度阈值', default=0.001)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def mkdirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def prefilter_boxes(class_objs,weights, thr = 0):
    '''
    Args:
        class_objs:  [model][objs]:[conf,bbox]
        thr:       保留bbox的最低置信度值
    Returns:
    '''
    # Create dict with boxes stored by its label
    new_objs = []
    for model_name, objs in class_objs.items():
        weight = weights[model_name]
        for obj in objs:
            obj[0] = float(obj[0]) * weight   #加权
            if obj[0] >= thr:
                new_objs.append(obj)
    #排序
    new_objs = np.array(new_objs)
    if len(new_objs) > 0:
        new_objs = new_objs[np.argsort(-new_objs[:, 0])]  #将序排列
    return new_objs

def rotate_IOU(g,p):
    g = np.asarray(g)
    p = np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2))) #创建一个多边形对象
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union

#确定属于哪一类
def find_matching_box(boxes_list, new_box, match_iou):
    '''
    Args:
        boxes_list: 已有的ensemble出来的框
        new_box:    新框
        match_iou:  IOU
    Returns:
        best_index: 重叠度最大框的索引
        best_iou:   最大重叠度
    '''
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i][1]
        iou = rotate_IOU(box, new_box[1])  #计算IOU
        if iou > best_iou:
            best_index = i
            best_iou = iou
    return best_index, best_iou

#对box进行整理
def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse   box格式[conf, bbox]
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """

    #bbox = np.zeros(8, dtype=np.float32)
    conf = 0
    best_conf = 0
    for b in boxes:
        if b[0] > best_conf:
            bbox = b[1]
            best_conf = b[0]
        # bbox += (b[0] * np.array(b[1]))    #假设bbox中每个点排列顺序都一样
        conf += b[0]
    #box[0] = boxes[0][0]
    if conf_type == 'avg':
        conf_final = conf / len(boxes)
    elif conf_type == 'max':
        conf_final = best_conf
    #bbox /= conf
    box = [conf_final, bbox]
    return box

#对一类对象进行WBF操作
def weighted_boxes_fusion(class_objs, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable 排除得分低于设置值的框
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value  融合bbox置信度的取值策略
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    # if weights is None:
    #     weights = np.ones(len(class_objs))   #如果不设置模型间权重则按等权重取值
    # if len(weights) != len(class_objs):
    #     print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(class_objs)))
    #     weights = np.ones(len(class_objs))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    boxes = prefilter_boxes(class_objs, weights, skip_box_thr)  #返回一个列表   加权值
    if len(boxes) == 0:
        return []   #error
    overall_boxes = []
    new_boxes = []
    weighted_boxes = []
    # Clusterize boxes  # 聚集框
    for j in range(0, len(boxes)):  # 对该类别的每个框
        index, best_iou = find_matching_box(new_boxes, boxes[j], iou_thr)  # 计算该框与已得框是否重合,与哪个框重合
        if index != -1:
            new_boxes[index][0].append(boxes[j][0])
            #weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)  # 边界变换
        else:
            new_boxes.append(boxes[j].copy())
            new_boxes[-1][0] = [new_boxes[-1][0]]
            #weighted_boxes.append(boxes[j].copy())

    # Rescale confidence based on number of models and boxes  #依据预测出来的筐数
    for i in range(len(new_boxes)):
        num_predicts = len(new_boxes[i][0])
        new_boxes[i][0] = np.mean(new_boxes[i][0])
        if not allows_overflow:
            new_boxes[i][0] = new_boxes[i][0] * min(weights.sum(),
                                                    num_predicts) / weights.sum()  # ? 模型总数
        else:
            new_boxes[i][0] = new_boxes[i][0] * num_predicts/weights.sum()
    overall_boxes.append(np.array(new_boxes))
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    return overall_boxes

def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    x1 = coordinate[0]
    y1 = coordinate[1]
    x2 = coordinate[2]
    y2 = coordinate[3]
    x3 = coordinate[4]
    y3 = coordinate[5]
    x4 = coordinate[6]
    y4 = coordinate[7]
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
                     + cal_line_length(combinate[i][1],dst_coordinate[1]) \
                     + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                     + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")

    return [combinate[force_flag][0][0], combinate[force_flag][0][1], combinate[force_flag][1][0], combinate[force_flag][1][1], combinate[force_flag][2][0], combinate[force_flag][2][1], combinate[force_flag][3][0], combinate[force_flag][3][1]]

#微调bbox
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



def WBF(args, iou_thr=0.55, draw_image=True):
    """
    This example shows how to ensemble boxes from 2 models using WBF method
    :return:
    """
    weights = args.weights
    results = defaultdict(lambda: defaultdict(defaultdict))
    print('loading results:')
    # 提取预测结果
    for model_num, model_result in enumerate(args.models_result):  #对每个模型
        result_files = os.listdir(model_result)
        for result_file in result_files:                   #每个图片
            result_path = os.path.join(model_result, result_file)
            tree = ET.parse(result_path)  # 打开xml文档
            root = tree.getroot()  # 获得root节点
            img_name = result_file.strip().split('.')[0]
            for object in root.find('objects').findall('object'):  # 每个结果
                label_name = object.find('possibleresult').find('name').text # 子节点下节点name的值
                conf = float(object.find('possibleresult').find('probability').text)
                bbox = []
                for i, point in enumerate(object.find('points').findall('point')):  # object下每个点
                    if i >= 4:
                        break
                    x, y = point.text.strip().split(',')
                    bbox.append(float(x))
                    bbox.append(float(y))
                #bbox = anchor_finetune(bbox, img_size=img_size[img_name])
                obj_det = [conf, bbox]

                try:
                    results[img_name][label_name][model_num].append(obj_det)
                except:
                    results[img_name][label_name][model_num] = []
                    results[img_name][label_name][model_num].append(obj_det)
            # else:
            #     print('error!')

    # init output xml

    # if draw_image:
    #     show_boxes(boxes_list, scores_list, labels_list)  #画图
    #  输出成DOTA格式再
    print('init emsemble results:')
    filltxt(args.save_path)
    print('emsembleling:')
    i=0
    for img_name, value1 in results.items():
        i=i+1
        print('img_name=',img_name,'  num:',i)
        root = minidom.parse(os.path.join(args.save_path, img_name + '.xml'))
        xmlBuilder = Document()
        objects = root.getElementsByTagName("objects").item(0)
        for label_name, value2 in value1.items():
            boxes = weighted_boxes_fusion(value2, weights=weights, iou_thr=args.IOU_threshold,
                                          skip_box_thr=args.conf_threshold)
            for box in boxes:
                conf = str(box[0])
                bbox = [str(point) for point in box[1]]
                # update_xml(os.path.join(args.save_path, img_name+'.xml'), label_name, conf, bbox)

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
                objnameContent = xmlBuilder.createTextNode(label_name)  # label_name
                objname.appendChild(objnameContent)
                possibleresult.appendChild(objname)
                probability = xmlBuilder.createElement("probability")
                #  probabilityContent = xmlBuilder.createTextNode(oneline[1])
                probabilityContent = xmlBuilder.createTextNode(conf)  # score
                probability.appendChild(probabilityContent)
                possibleresult.appendChild(probability)
                object.appendChild(possibleresult)
                points = xmlBuilder.createElement("points")
                point = xmlBuilder.createElement("point")
                #  pointContent = xmlBuilder.createTextNode(oneline[2]+','+oneline[3])
                pointContent = xmlBuilder.createTextNode(bbox[0] + ',' + bbox[1])  # point1
                point.appendChild(pointContent)
                points.appendChild(point)
                point = xmlBuilder.createElement("point")
                #  pointContent = xmlBuilder.createTextNode(oneline[4]+','+oneline[5])
                pointContent = xmlBuilder.createTextNode(bbox[2] + ',' + bbox[3])  # point2
                point.appendChild(pointContent)
                points.appendChild(point)
                point = xmlBuilder.createElement("point")
                #  pointContent = xmlBuilder.createTextNode(oneline[6]+','+oneline[7])
                pointContent = xmlBuilder.createTextNode(bbox[4] + ',' + bbox[5])  # point3
                point.appendChild(pointContent)
                points.appendChild(point)
                point = xmlBuilder.createElement("point")
                #  pointContent = xmlBuilder.createTextNode(oneline[8]+','+oneline[9])
                pointContent = xmlBuilder.createTextNode(bbox[6] + ',' + bbox[7])  # points
                point.appendChild(pointContent)
                points.appendChild(point)
                point = xmlBuilder.createElement("point")
                #  pointContent = xmlBuilder.createTextNode(oneline[2]+','+oneline[3])
                pointContent = xmlBuilder.createTextNode(bbox[0] + ',' + bbox[1])  #
                point.appendChild(pointContent)
                points.appendChild(point)
                object.appendChild(points)
                objects.appendChild(object)
        #objects.appendChild(object)
        root.writexml(open(os.path.join(args.save_path, img_name + '.xml'), "w"), encoding='utf-8')


def filltxt(save_path):
    for i in range(8287):#
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        # txtFile = open(os.path.join(txt_path, name))
        # txtList = txtFile.readlines()
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
        annotation.appendChild(objects)
        f = open(os.path.join(save_path, str(i)) + ".xml", 'w')
        xmlBuilder.writexml(f, indent='', newl='\n', addindent='\t', encoding='utf-8')
        f.close()


def update_xml(path, label_name, conf, bbox):
    root = minidom.parse(path)
    xmlBuilder = Document()
    objects = root.getElementsByTagName("objects").item(0)#root.find("objects")
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
    objnameContent = xmlBuilder.createTextNode(label_name)  # label_name
    objname.appendChild(objnameContent)
    possibleresult.appendChild(objname)
    probability = xmlBuilder.createElement("probability")
    #  probabilityContent = xmlBuilder.createTextNode(oneline[1])
    probabilityContent = xmlBuilder.createTextNode(conf)  # score
    probability.appendChild(probabilityContent)
    possibleresult.appendChild(probability)
    object.appendChild(possibleresult)
    points = xmlBuilder.createElement("points")
    point = xmlBuilder.createElement("point")
    #  pointContent = xmlBuilder.createTextNode(oneline[2]+','+oneline[3])
    pointContent = xmlBuilder.createTextNode('x1,y1')  # point1
    point.appendChild(pointContent)
    points.appendChild(point)
    point = xmlBuilder.createElement("point")
    #  pointContent = xmlBuilder.createTextNode(oneline[4]+','+oneline[5])
    pointContent = xmlBuilder.createTextNode('x2,y2')  # point2
    point.appendChild(pointContent)
    points.appendChild(point)
    point = xmlBuilder.createElement("point")
    #  pointContent = xmlBuilder.createTextNode(oneline[6]+','+oneline[7])
    pointContent = xmlBuilder.createTextNode('x3,y3')  # point3
    point.appendChild(pointContent)
    points.appendChild(point)
    point = xmlBuilder.createElement("point")
    #  pointContent = xmlBuilder.createTextNode(oneline[8]+','+oneline[9])
    pointContent = xmlBuilder.createTextNode('x4,y4')  # points
    point.appendChild(pointContent)
    points.appendChild(point)
    point = xmlBuilder.createElement("point")
    #  pointContent = xmlBuilder.createTextNode(oneline[2]+','+oneline[3])
    pointContent = xmlBuilder.createTextNode('x1,y1') #
    point.appendChild(pointContent)
    points.appendChild(point)
    object.appendChild(points)
    objects.appendChild(object)
    root.writexml(open(path,"w" ), encoding='utf-8')
    # ET.dump(root)  #打印xml
    # root.write(path)

def get_imgs_size(imgs_dir,save_path):
    imgs_size = {}
    img_names = os.listdir(imgs_dir)
    for img_name in img_names:
        img = cv2.imread(os.path.join(imgs_dir,img_name))
        img_y, img_x, _ = img.shape[:]
        name = img_name.strip().split('.')[0]
        imgs_size[name] = [img_x, img_y]
    with open(save_path, 'w') as file_img:
        json.dump(imgs_size, file_img)


if __name__ == '__main__':
    args = parse_args()
    #mkdirs_if_not_exists(args.save_path)
    WBF(args)
    #filltxt('/home/ggm/GGM/competition_trick/try_em')
    #update_xml('/home/ggm/GGM/competition_trick/try_em/0.xml')
    #get_imgs_size('/home/disk/FAIR1M_FUll/test/images','/home/disk/FAIR1M_emsemble/imgs_size.json')
    print("done!")