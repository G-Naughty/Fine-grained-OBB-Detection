# -*- coding: utf-8 -*-
import argparse
import itertools
import easydict
import os
import pickle
from tkinter import (END, Button, Checkbutton, E, Entry, IntVar, Label,
                     Listbox, Menu, N, S, Scrollbar, StringVar, Tk, W, ttk)
import cv2
import matplotlib
import math
import numpy as np
import platform
import pycocotools.mask as maskUtils
from PIL import Image, ImageTk
from tqdm import trange
from shapely.geometry import Polygon  # 多边形

matplotlib.use('TkAgg')

"""离线推理查看结果
使用过程：
先根据配置文件和权重文件产生验证集结果文件 val.pkl
python tools/test.py user/retina-fair1m.py user/gfl-visdrone/latest.pth --out user/gfl-visdrone/visdrone_val.pkl
再将结果文件作为参数传入本代码

"""

cfg = easydict.EasyDict()
# dataset settings
cfg.dataset_type = 'DOTA_OBB_Dataset'
cfg.data_root = '/home/ggm/GGM/OBBDetection-master/data/FaIR1M/'
cfg.classes = ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919',
               'A220', 'A321', 'A330', 'A350', 'ARJ21',
               'other-airplane', 'Passenger_Ship', 'Motorboat', 'Fishing_Boat', 'Tugboat',
               'Engineering_Ship', 'Liquid_Cargo_Ship', 'Dry_Cargo_Ship', 'Warship', 'other-ship',
               'Small_Car', 'Bus', 'Cargo_Truck', 'Dump_Truck', 'Van',
               'Trailer', 'Tractor', 'Excavator', 'Truck_Tractor', 'other-vehicle',
               'Basketball_Court', 'Tennis_Court', 'Football_Field', 'Baseball_Field', 'Intersection',
               'Roundabout', 'Bridge')

cfg.data = dict(
    train=dict(
        type=cfg.dataset_type,
        ann_file=cfg.data_root + 'trainval/annfiles/patch_annfile.pkl',
        img_prefix=cfg.data_root + 'trainval/images/'),
    val=dict(
        type=cfg.dataset_type,
        ann_file=cfg.data_root + 'trainval/annfiles/patch_annfile.pkl',
        img_prefix=cfg.data_root + 'trainval/images/'),
    test=dict(
        type=cfg.dataset_type,
        img_prefix='/home/disk/Fine-grained Object Recognition in High-Resolution Optical Images/test_benchmark/images/'))

width_det = 2


def parse_args():
    parser = argparse.ArgumentParser(description='DetVisGUI')
    # parser.add_argument('--det_file', default='',
    parser.add_argument('--det_file', default='/home/ggm/GGM/OBBDetection-master/work_dir/oriented_obb_ori/dets.pkl',
                        help='detection results file path')
    parser.add_argument('--stage', default='test', choices=['train', 'val', 'test'], help='stage')
    parser.add_argument('--no_gt', action='store_true', help='test images without groundtruth')
    parser.add_argument('--det_box_color', default=(255, 255, 0), help='detection box color')
    parser.add_argument('--gt_box_color', default=(0, 255, 0), help='groundtruth box color')
    parser.add_argument('--output', default='output', help='image save folder')
    args = parser.parse_args()
    return args


args = parse_args()


def obb2poly(obboxes):
    center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)

    vector1 = np.concatenate([w / 2 * Cos, -w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, -h / 2 * Cos], axis=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    return np.concatenate([point1, point2, point3, point4], axis=-1)


def rotate_IOU(g, p):
    g = np.asarray(g)
    p = np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))  # 创建一个多边形对象
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def calc_square_iou(box, boxes_gt):
    # 计算预测框box与所有真值间的正框iou
    box = box.reshape(-1, 4, 2)  # 转成矩形坐标。偷懒计算
    boxes_gt = boxes_gt.reshape(-1, 4, 2)
    BB = np.zeros(4)
    BBGT = np.zeros((len(boxes_gt), 4))
    BB[0] = box[:, :, 0].min()
    BB[1] = box[:, :, 1].min()
    BB[2] = box[:, :, 0].max()
    BB[3] = box[:, :, 1].max()
    for i in range(len(boxes_gt)):
        BBGT[i, 0] = boxes_gt[i, :, 0].min()
        BBGT[i, 1] = boxes_gt[i, :, 1].min()
        BBGT[i, 2] = boxes_gt[i, :, 0].max()
        BBGT[i, 3] = boxes_gt[i, :, 1].max()
    # intersection
    ixmin = np.maximum(BBGT[:, 0], BB[0])
    iymin = np.maximum(BBGT[:, 1], BB[1])
    ixmax = np.minimum(BBGT[:, 2], BB[2])
    iymax = np.minimum(BBGT[:, 3], BB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    # union
    uni = ((BB[2] - BB[0] + 1.) * (BB[3] - BB[1] + 1.) + (BBGT[:, 2] - BBGT[:, 0] + 1.) * (
            BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
    overlaps = inters / uni
    return overlaps


# dota dataset
class DOTA_obb_dataset:
    def __init__(self, cfg, args):
        self.dataset = 'DOTA OBB'
        self.det_file = args.det_file if args.stage == 'test' else ''
        if hasattr(getattr(cfg.data, args.stage), 'ann_file'):
            self.anno_file = os.path.join(getattr(cfg.data, args.stage).ann_file)
        else:
            self.anno_file = None
            args.no_gt = True
        self.img_root = os.path.join(getattr(cfg.data, args.stage).img_prefix)
        self.has_anno = not args.no_gt
        self.mask = False

        self.img_list = self.get_img_list()

        self.anno_list = self.get_anno_list() if self.anno_file else None

        self.results = self.get_det_results() if self.det_file != '' else None
        self.aug_category = aug_category(cfg.classes)

        if self.det_file != '':
            self.img_det = {
                self.img_list[i]: self.results[:, i]
                for i in range(len(self.img_list))
            }

    def get_img_list(self):  # 根据pkl注释文件获取图片顺序v
        if self.det_file == '':
            with open(self.anno_file, 'rb') as f:
                anns = pickle.load(f)
            ann_list = np.array([ann['filename'] for ann in anns['content']])
        else:
            with open(self.det_file, 'rb') as f:
                anns = pickle.load(f)
            ann_list = [ann[0] + '.tif' for ann in anns]
        return ann_list
        # return os.listdir(self.img_root)

    def get_anno_list(self):
        with open(self.anno_file, 'rb') as f:
            annos = pickle.load(f)  # [(bg + cls), images]
        return {dic['filename']: dic['ann'] for dic in annos['content']}

    def get_det_results(self):
        with open(self.det_file, 'rb') as f:
            det_results = pickle.load(f)  # [(bg + cls), images]
        # det_results = np.array(det_results)
        det_results = np.array([det_result[1] for det_result in det_results])
        det_results = np.transpose(det_results, (1, 0))
        return det_results

    def get_img_by_name(self, name):
        img = Image.open(os.path.join(self.img_root, name)).convert('RGB')
        return img

    def get_img_by_index(self, idx):
        img = Image.open(os.path.join(self.img_root, self.img_list[idx])).convert('RGB')
        return img

    def get_singleImg_gt(self, name):  # get annotations by image name
        img_anns = []
        objs = self.anno_list[name]
        for i in range(len(objs['labels'])):  # 遍历所有目标
            name = cfg.classes[objs['labels'][i]]
            box = objs['bboxes'][i]
            bbox = [  # x1,y1,x2,y2,x3,y3,x4,y4
                int(float(box[0])), int(float(box[1])),
                int(float(box[2])), int(float(box[3])),
                int(float(box[4])), int(float(box[5])),
                int(float(box[6])), int(float(box[7])),
            ]
            diff = objs['diffs'][i]  # 困难程度
            img_anns.append([name, bbox, diff])
        return img_anns

    def get_singleImg_dets(self, name):
        return self.img_det[name]


def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
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
                     + cal_line_length(combinate[i][1], dst_coordinate[1]) \
                     + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                     + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass  # ???代码冗余
        # print("choose one direction!")
    return np.array(combinate[force_flag]).reshape(8)


def get_best_begin_point(coordinates):
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates


# main GUI
class vis_tool:
    def __init__(self):
        self.args = parse_args()
        self.window = Tk()
        self.menubar = Menu(self.window)

        self.info = StringVar()
        self.info_label = Label(
            self.window, bg='yellow', width=4, textvariable=self.info)

        self.listBox_img = Listbox(
            self.window, width=50, height=25, font=('Times New Roman', 10))
        self.listBox_obj = Listbox(
            self.window, width=50, height=12, font=('Times New Roman', 10))

        self.scrollbar_img = Scrollbar(
            self.window, width=15, orient='vertical')
        self.scrollbar_obj = Scrollbar(
            self.window, width=15, orient='vertical')

        self.listBox_img_info = StringVar()
        self.listBox_img_label = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=4,
            height=1,
            textvariable=self.listBox_img_info)

        self.listBox_obj_info = StringVar()
        self.listBox_obj_label1 = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=4,
            height=1,
            textvariable=self.listBox_obj_info)
        self.listBox_obj_label2 = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=4,
            height=1,
            text='Object Class : Score (IoU)')

        if cfg.dataset_type == 'DOTA_OBB_Dataset':
            self.data_info = DOTA_obb_dataset(cfg, self.args)
        else:
            raise RuntimeError(cfg.data_info + ' is not support now')
        self.info.set('DATASET: {}'.format(self.data_info.dataset))

        # load image and show it on the window
        self.img = self.data_info.get_img_by_index(0)
        self.photo = ImageTk.PhotoImage(self.img)
        self.label_img = Label(self.window, image=self.photo)

        self.show_det_txt = IntVar(value=1)
        self.checkbn_det_txt = Checkbutton(
            self.window,
            text='Text',
            font=('Arial', 10, 'bold'),
            variable=self.show_det_txt,
            command=self.change_img,
            fg='#0000FF')

        self.show_dets = IntVar(value=1)
        self.checkbn_det = Checkbutton(
            self.window,
            text='Detections',
            font=('Arial', 10, 'bold'),
            variable=self.show_dets,
            command=self.change_img,
            fg='#0000FF')

        self.show_gt_txt = IntVar(value=1)
        self.checkbn_gt_txt = Checkbutton(
            self.window,
            text='Text',
            font=('Arial', 10, 'bold'),
            variable=self.show_gt_txt,
            command=self.change_img,
            fg='#FF8C00')

        self.show_gts = IntVar(value=1)
        self.checkbn_gt = Checkbutton(
            self.window,
            text='Groundtruth',
            font=('Arial', 10, 'bold'),
            variable=self.show_gts,
            command=self.change_img,
            fg='#FF8C00')

        self.combo_label = Label(
            self.window,
            bg='yellow',
            width=10,
            height=1,
            text='Show Category',
            font=('Arial', 11))
        self.combo_category = ttk.Combobox(
            self.window,
            font=('Arial', 11),
            values=self.data_info.aug_category.combo_list)
        self.combo_category.current(0)
        self.th_label = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=10,
            height=1,
            text='Score Threshold')
        self.threshold = np.float32(0.5)
        self.th_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=10,
            textvariable=StringVar(self.window, value=str(self.threshold)))
        self.th_button = Button(
            self.window, text='Enter', height=1, command=self.change_threshold)

        self.iou_th_label = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=10,
            height=1,
            text='IoU Threshold')
        self.iou_threshold = np.float32(0.5)
        self.iou_th_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=10,
            textvariable=StringVar(self.window, value=str(self.iou_threshold)))
        self.iou_th_button = Button(
            self.window, text='Enter', height=1, command=self.change_iou_threshold)

        self.find_label = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=10,
            height=1,
            text='find')
        self.find_name = ''
        self.find_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=10,
            textvariable=StringVar(self.window, value=str(self.find_name)))
        self.find_button = Button(
            self.window, text='Enter', height=1, command=self.findname)

        self.listBox_img_idx = 0

        # ====== ohter attribute ======
        self.img_name = ''
        self.show_img = None

        self.output = self.args.output

        if not os.path.isdir(self.output):
            os.makedirs(self.output)

        self.img_list = self.data_info.img_list

        # flag for find/threshold button switch focused element
        self.button_clicked = False

    def change_threshold(self, event=None):
        try:
            self.threshold = np.float32(self.th_entry.get())
            self.change_img()

            # after changing threshold, focus on listBox for easy control
            if self.window.focus_get() == self.listBox_obj:
                self.listBox_obj.focus()
            else:
                self.listBox_img.focus()

            self.button_clicked = True

        except ValueError:
            self.window.title('Please enter a number as score threshold.')

    def change_iou_threshold(self, event=None):

        try:
            self.iou_threshold = np.float32(self.iou_th_entry.get())
            self.change_img()

            # after changing threshold, focus on listBox for easy control
            if self.window.focus_get() == self.listBox_obj:
                self.listBox_obj.focus()
            else:
                self.listBox_img.focus()

            self.button_clicked = True

        except ValueError:
            self.window.title("Please enter a number as IoU threshold.")

    # draw groundtruth
    def draw_gt_boxes(self, img, objs):
        for obj in objs:
            cls_name = obj[0]

            # according combobox to decide whether to plot this category
            if self.combo_category.get() == 'All':
                show_category = self.data_info.aug_category.category
            else:
                show_category = [self.combo_category.get()]

            if cls_name not in show_category:
                # raise RuntimeError(cls_name+' is not in category,please check')
                continue

            box = np.array(obj[1]).reshape(-1, 4, 2)
            diff = obj[2]
            box[box < 0] = 0  # 避免异常值
            # box[box > self.img_width] = self.img_width
            box[:, :, 0][box[:, :, 0] > self.img_width] = self.img_width
            box[:, :, 1][box[:, :, 1] > self.img_height] = self.img_height

            xmin, ymin, xmax, ymax = \
                box[:, :, 0].min(), box[:, :, 1].min(), box[:, :, 0].max(), box[:, :, 1].max()

            font = cv2.FONT_HERSHEY_SIMPLEX
            if self.show_gt_txt.get():  # 绘制ground——truth
                cls_name = cls_name  # + ' ' + str(obj[1][0])
                if ymax + 30 >= self.img_height:
                    cv2.rectangle(img, (xmin, ymin),
                                  (xmin + len(cls_name) * 10, int(ymin - 20)),
                                  (255, 140, 0), cv2.FILLED)
                    cv2.putText(img, cls_name, (xmin, int(ymin - 5)), font,
                                0.5, (255, 255, 255), 1)
                else:
                    cv2.rectangle(img, (xmin, ymax),
                                  (xmin + len(cls_name) * 10, int(ymax + 20)),
                                  (255, 140, 0), cv2.FILLED)
                    cv2.putText(img, cls_name, (xmin, int(ymax + 15)), font,
                                0.5, (255, 255, 255), 1)

            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
            #               self.args.gt_box_color, 1)
            cv2.polylines(img, box, True, self.args.gt_box_color, 1)

        return img

    def get_iou(self, det):

        iou = np.zeros_like(det)
        GT = self.data_info.get_singleImg_gt(self.img_name)  # 提取出标签

        for idx, cls_objs in enumerate(det):  # 遍历每个类的检测结果

            category = self.data_info.aug_category.category[idx]
            BBGT = []
            for t in GT:
                if not t[0] == category: continue
                BBGT.append(t[1])
            BBGT = np.asarray(BBGT)
            d = [0] * len(BBGT)  # for check 1 GT map to several det

            confidence = cls_objs[:, 5]
            BB = cls_objs[:, 0:5]  # bounding box

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]

            # for returning original order
            ind_table = {i: sorted_ind[i] for i in range(len(sorted_ind))}

            iou[idx] = np.zeros(len(BB))

            if len(BBGT) > 0:
                BB = obb2poly(BB)  # 坐标转换
                for i in range(len(BB)):  # 遍历每个预测框
                    overlaps = np.zeros(len(BBGT))
                    positive_iou = calc_square_iou(BB[i], BBGT)  # 把预测框与每个GT做计算正框iou
                    for j in range(len(BBGT)):  # 把预测框与每个GT做计算旋转框iou
                        if positive_iou[j] > 0:
                            overlaps[j] = rotate_IOU(BBGT[j], BB[i])
                    ovmax = np.max(overlaps)  # max overlaps with all gt
                    jmax = np.argmax(overlaps)  # 最终计算出与第三个gt的iou高达0.89

                    if ovmax > self.iou_threshold:
                        if not d[jmax]:
                            d[jmax] = 1
                        else:  # multiple bounding boxes map to one gt
                            ovmax = -ovmax

                    iou[idx][ind_table[i]] = ovmax  # return to unsorted order
        return iou

    def draw_all_det_boxes(self, img, single_detection):

        if self.data_info.has_anno:
            self.iou = self.get_iou(single_detection)

        for idx, cls_objs in enumerate(single_detection):
            category = self.data_info.aug_category.category[idx]

            if self.combo_category.get() == 'All':
                show_category = self.data_info.aug_category.category
            else:
                show_category = [self.combo_category.get()]

            if category not in show_category:
                continue

            for obj_idx, obj in enumerate(cls_objs):
                [score, box] = [round(obj[5], 2), obj[:5]]

                if score >= self.threshold:
                    # box = list(map(int, list(map(round, box[:-1]))))
                    xmin = int(max(obj[0] - obj[2] / 2, 0))
                    ymin = int(max(obj[1] - obj[3] / 2, 0))
                    xmax = int(min(obj[0] + obj[2] / 2, self.img_width))
                    ymax = int(min(obj[1] + obj[3] / 2, self.img_height))

                    if not self.data_info.has_anno or \
                            self.iou[idx][obj_idx] >= self.iou_threshold:
                        color = self.args.det_box_color
                    else:
                        color = (255, 0, 0)

                    if self.show_det_txt.get():
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = category + ' : ' + str(score)

                        if ymax + 30 >= self.img_height:
                            cv2.rectangle(img, (xmin, ymin), (xmin + len(text) * 9, int(ymin - 20)), (0, 0, 255),
                                          cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymin - 5)), font, 0.5, (255, 255, 255), 1)
                        else:
                            cv2.rectangle(img, (xmin, ymax), (xmin + len(text) * 9, int(ymax + 20)), (0, 0, 255),
                                          cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymax + 15)), font, 0.5, (255, 255, 255), 1)

                    # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, width_det)
                    box = obb2poly(box).reshape(-1, 4, 2).astype('int')  # 坐标转换
                    # box = xywha2xyxy(np.expand_dims(obj[:5], axis=0)).reshape(-1, 4, 2).astype('int')
                    cv2.polylines(img, box, True, color, width_det)
        return img

    def draw_all_det_boxes_masks(self, img, single_detection):
        img = np.require(img, requirements=['W'])
        boxes, masks = single_detection

        # draw segmentation masks
        # reference mmdetection/mmdet/models/detectors/base.py
        if self.combo_category.get() != 'All':
            show_idx = self.data_info.aug_category.category.index(
                self.combo_category.get())
            masks = np.asarray([masks[show_idx]])
            boxes = np.asarray([boxes[show_idx]])
            category = self.data_info.aug_category.category[show_idx]

        segms = list(itertools.chain(*masks))
        bboxes = np.vstack(boxes)

        inds = np.where(np.round(bboxes[:, -1], 2) >= self.threshold)[0]

        self.color_list = []
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            if type(segms[0]) == np.ndarray:
                mask = segms[i]
            elif type(segms[0]) == dict:
                mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
            self.color_list.append('#%02x%02x%02x' % tuple(color_mask[0]))

        if self.data_info.has_anno:
            boxes2, _ = single_detection
            self.iou = self.get_iou(boxes2)
            if self.combo_category.get() != 'All':
                iou = np.asarray([self.iou[show_idx]])
            else:
                iou = self.iou

        # draw bounding box
        for idx, cls_objs in enumerate(boxes):
            if self.combo_category.get() == 'All':
                category = self.data_info.aug_category.category[idx]

            for obj_idx, obj in enumerate(cls_objs):
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    box = list(map(int, list(map(round, box))))
                    xmin = max(box[0], 0)
                    ymin = max(box[1], 0)
                    xmax = min(box[2], self.img_width)
                    ymax = min(box[3], self.img_height)

                    if not self.data_info.has_anno or \
                            iou[idx][obj_idx] >= self.iou_threshold:
                        color = self.args.det_box_color
                    else:
                        color = (255, 0, 0)

                    if self.show_det_txt.get():
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = category + ' : ' + str(score)

                        if ymax + 30 >= self.img_height:
                            cv2.rectangle(
                                img, (xmin, ymin),
                                (xmin + len(text) * 9, int(ymin - 20)),
                                (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymin - 5)), font,
                                        0.5, (255, 255, 255), 1)
                        else:
                            cv2.rectangle(
                                img, (xmin, ymax),
                                (xmin + len(text) * 9, int(ymax + 20)),
                                (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymax + 15)),
                                        font, 0.5, (255, 255, 255), 1)

                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

        return img

    def change_img(self, event=None):
        if len(self.listBox_img.curselection()) != 0:
            self.listBox_img_idx = self.listBox_img.curselection()[0]

        self.listBox_img_info.set('Image  {:6}  / {:6}'.format(
            self.listBox_img_idx + 1, self.listBox_img.size()))

        name = self.listBox_img.get(self.listBox_img_idx)
        self.window.title('DATASET : ' + self.data_info.dataset + '   ' + name)

        img = self.data_info.get_img_by_name(name)
        self.img_width, self.img_height = img.width, img.height

        img = np.asarray(img)

        self.img_name = name
        self.img = img

        if self.data_info.has_anno and self.show_gts.get():
            objs = self.data_info.get_singleImg_gt(name)
            img = self.draw_gt_boxes(img, objs)

        if self.data_info.results is not None and self.show_dets.get():
            if self.data_info.mask is False:
                dets = self.data_info.get_singleImg_dets(name)
                img = self.draw_all_det_boxes(img, dets)
            else:
                dets = self.data_info.get_singleImg_dets(name).transpose(
                    (1, 0))
                img = self.draw_all_det_boxes_masks(img, dets)

            self.clear_add_listBox_obj()

        self.show_img = img
        img = Image.fromarray(img)
        img = self.scale_img(img)
        self.photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=self.photo)
        self.window.update_idletasks()

        if self.img_name in os.listdir(self.output):
            self.listBox_img_label.config(bg='#CCFF99')
        else:
            self.listBox_img_label.config(bg='yellow')

    def draw_one_det_boxes(self, img, single_detection, selected_idx=-1):
        idx_counter = 0
        for idx, cls_objs in enumerate(single_detection):

            category = self.data_info.aug_category.category[idx]
            if self.combo_category.get() == 'All':
                show_category = self.data_info.aug_category.category
            else:
                show_category = [self.combo_category.get()]

            if category not in show_category:
                continue

            for obj_idx, obj in enumerate(cls_objs):
                [score, box] = [round(obj[5], 2), obj[:5]]

                if score >= self.threshold:
                    if idx_counter == selected_idx:
                        # box = list(map(int, list(map(round, box[:-1]))))
                        xmin = int(max(obj[0] - obj[2] / 2, 0))
                        ymin = int(max(obj[1] - obj[3] / 2, 0))
                        xmax = int(min(obj[0] + obj[2] / 2, self.img_width))
                        ymax = int(min(obj[1] + obj[3] / 2, self.img_height))

                        if not self.data_info.has_anno or \
                                self.iou[idx][obj_idx] >= self.iou_threshold:
                            color = self.args.det_box_color
                        else:
                            color = (255, 0, 0)

                        if self.show_det_txt.get():
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text = category + ' : ' + str(score)

                            if ymax + 30 >= self.img_height:
                                cv2.rectangle(
                                    img, (xmin, ymin),
                                    (xmin + len(text) * 9, int(ymin - 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymin - 5)),
                                            font, 0.5, (255, 255, 255), 1)
                            else:
                                cv2.rectangle(
                                    img, (xmin, ymax),
                                    (xmin + len(text) * 9, int(ymax + 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymax + 15)),
                                            font, 0.5, (255, 255, 255), 1)

                        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, width_det)
                        # box = xywha2xyxy(np.expand_dims(obj[:5], axis=0)).reshape(-1, 4, 2).astype('int')
                        box = obb2poly(box).reshape(-1, 4, 2).astype('int')  # 坐标转换
                        cv2.polylines(img, box, True, color, width_det)

                        return img
                    else:
                        idx_counter += 1

    def draw_one_det_boxes_masks(self, img, single_detection, selected_idx=-1):
        img = np.require(img, requirements=['W'])
        boxes, masks = single_detection

        # draw segmentation masks
        # reference mmdetection/mmdet/models/detectors/base.py
        if self.combo_category.get() != 'All':
            show_idx = self.data_info.aug_category.category.index(
                self.combo_category.get())
            category = self.data_info.aug_category.category[
                show_idx]  # fixed category
            masks = np.asarray([masks[show_idx]])
            boxes = np.asarray([boxes[show_idx]])

        segms = list(itertools.chain(*masks))
        bboxes = np.vstack(boxes)

        inds = np.where(np.round(bboxes[:, -1], 2) >= self.threshold)[0]

        self.color_list = []
        for inds_idx, i in enumerate(inds):
            if inds_idx == selected_idx:
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)

                if type(segms[0]) == np.ndarray:
                    mask = segms[i]
                elif type(segms[0]) == dict:
                    mask = maskUtils.decode(segms[i]).astype(np.bool)

                img[mask] = img[mask] * 0.5 + color_mask * 0.5
                self.color_list.append('#%02x%02x%02x' % tuple(color_mask[0]))

        if self.data_info.has_anno:
            if self.combo_category.get() != 'All':
                iou = np.asarray([self.iou[show_idx]])
            else:
                iou = self.iou

        # draw bounding box
        idx_counter = 0
        for idx, cls_objs in enumerate(boxes):
            if self.combo_category.get() == 'All':
                category = self.data_info.aug_category.category[idx]

            for obj_idx, obj in enumerate(cls_objs):
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    if idx_counter == selected_idx:
                        box = list(map(int, list(map(round, box))))
                        xmin = max(box[0], 0)
                        ymin = max(box[1], 0)
                        xmax = min(box[2], self.img_width)
                        ymax = min(box[3], self.img_height)

                        if not self.data_info.has_anno or \
                                iou[idx][obj_idx] >= self.iou_threshold:
                            color = self.args.det_box_color
                        else:
                            color = (255, 0, 0)

                        if self.show_det_txt.get():
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text = category + ' : ' + str(score)

                            if ymax + 30 >= self.img_height:
                                cv2.rectangle(
                                    img, (xmin, ymin),
                                    (xmin + len(text) * 9, int(ymin - 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymin - 5)),
                                            font, 0.5, (255, 255, 255), 1)
                            else:
                                cv2.rectangle(
                                    img, (xmin, ymax),
                                    (xmin + len(text) * 9, int(ymax + 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymax + 15)),
                                            font, 0.5, (255, 255, 255), 1)

                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                                      color, 2)

                        return img
                    else:
                        idx_counter += 1

    # plot only one object
    def change_obj(self, event=None):
        if len(self.listBox_obj.curselection()) == 0:
            self.listBox_img.focus()
            return
        else:
            listBox_obj_idx = self.listBox_obj.curselection()[0]

        self.listBox_obj_info.set('Detected Object : {:4}  / {:4}'.format(
            listBox_obj_idx + 1, self.listBox_obj.size()))

        name = self.listBox_img.get(self.listBox_img_idx)
        img = self.data_info.get_img_by_name(name)
        self.img_width, self.img_height = img.width, img.height
        img = np.asarray(img)
        self.img_name = name
        self.img = img

        if self.data_info.has_anno and self.show_gts.get():
            objs = self.data_info.get_singleImg_gt(name)
            img = self.draw_gt_boxes(img, objs)

        if self.data_info.results is not None and self.show_dets.get():

            if self.data_info.mask is False:
                dets = self.data_info.get_singleImg_dets(name)
                img = self.draw_one_det_boxes(img, dets, listBox_obj_idx)
            else:
                dets = self.data_info.get_singleImg_dets(name).transpose(
                    (1, 0))
                img = self.draw_one_det_boxes_masks(img, dets, listBox_obj_idx)

        self.show_img = img
        img = Image.fromarray(img)
        img = self.scale_img(img)
        self.photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=self.photo)
        self.window.update_idletasks()

        if self.img_name in os.listdir(self.output):
            self.listBox_img_label.config(bg='#CCFF99')
        else:
            self.listBox_img_label.config(bg='yellow')

    # ============================================

    def scale_img(self, img):
        [s_w, s_h] = [1, 1]

        # if window size is (1920, 1080),
        # the default max image size is (1440, 810)
        # (fix_width, fix_height) = (1920, 1080)
        (fix_width, fix_height) = (1440, 810)

        # change image size according to window size
        if self.window.winfo_width() != 1:
            fix_width = (
                    self.window.winfo_width() - self.listBox_img.winfo_width() -
                    self.scrollbar_img.winfo_width() - 5)
            # fix_height = int(fix_width * 9 / 16)
            fix_height = 750

        # handle image size is too big
        if img.width > fix_width:
            s_w = fix_width / img.width
        if img.height > fix_height:
            s_h = fix_height / img.height

        scale = min(s_w, s_h)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.ANTIALIAS)
        return img

    def clear_add_listBox_obj(self):
        self.listBox_obj.delete(0, 'end')

        if self.data_info.mask is False:
            single_detection = self.data_info.get_singleImg_dets(
                self.img_list[self.listBox_img_idx])
        else:
            single_detection, single_mask = self.data_info.get_singleImg_dets(
                self.img_list[self.listBox_img_idx]).transpose((1, 0))

        if self.combo_category.get() == 'All':
            show_category = self.data_info.aug_category.category
        else:
            show_category = [self.combo_category.get()]

        num = 0
        for idx, cls_objs in enumerate(single_detection):
            category = self.data_info.aug_category.category[idx]

            if category not in show_category:
                continue

            for obj_idx, obj in enumerate(cls_objs):
                score = np.round(obj[-1], 2)
                if score >= self.threshold:
                    if not self.data_info.has_anno:
                        self.listBox_obj.insert('end', category + " : " + str(score))
                    elif self.iou[idx][obj_idx] > self.iou_threshold:
                        s = "{:9}:{:4.3}({:4.3}){}".format(category, score,
                                                           abs(round(self.iou[idx][obj_idx], 2)),
                                                           str(obb2poly(obj[:5]).astype('int')))
                        self.listBox_obj.insert('end', s)
                        self.listBox_obj.itemconfig(num, fg="green")
                    else:
                        s = "{:9}:{:4.3}({:4.3}){}".format(category, score,
                                                           abs(round(self.iou[idx][obj_idx], 2)),
                                                           str(obb2poly(obj[:5]).astype('int')))
                        self.listBox_obj.insert('end', s)
                        self.listBox_obj.itemconfig(num, fg="red")

                    num += 1

        self.listBox_obj_info.set('Detected Object : {:3}'.format(num))

    def change_threshold_button(self, v):
        self.threshold += v

        if self.threshold <= 0:
            self.threshold = 0
        elif self.threshold >= 1:
            self.threshold = 1

        self.th_entry.delete(0, END)
        self.th_entry.insert(0, str(round(self.threshold, 2)))
        self.change_threshold()

    def change_iou_threshold_button(self, v):
        self.iou_threshold += v

        if self.iou_threshold <= 0:
            self.iou_threshold = 0
        elif self.iou_threshold >= 1:
            self.iou_threshold = 1

        self.iou_th_entry.delete(0, END)
        self.iou_th_entry.insert(0, str(round(self.iou_threshold, 2)))
        self.change_iou_threshold()

    def save_img(self):
        print('Save image to ' + os.path.join(self.output, self.img_name))
        cv2.imwrite(
            os.path.join(self.output, self.img_name),
            cv2.cvtColor(self.show_img, cv2.COLOR_BGR2RGB))
        self.listBox_img_label.config(bg='#CCFF99')

    def save_all_images(self):
        print('plot all images ... ')

        for listBox_img_idx in trange(len(self.data_info.img_list)):

            name = self.listBox_img.get(listBox_img_idx)
            self.img_name = name

            img = np.asarray(self.data_info.get_img_by_name(name))
            self.img = img

            if self.data_info.has_anno and self.show_gts.get():
                objs = self.data_info.get_singleImg_gt(name)
                img = self.draw_gt_boxes(img, objs)

            if self.data_info.results is not None and self.show_dets.get():
                if self.data_info.mask is False:
                    dets = self.data_info.get_singleImg_dets(name)
                    img = self.draw_all_det_boxes(img, dets)
                else:
                    dets = self.data_info.get_singleImg_dets(name).transpose(
                        (1, 0))
                    img = self.draw_all_det_boxes_masks(img, dets)

            cv2.imwrite(os.path.join(self.output, name),
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def eventhandler(self, event):
        entry_list = [self.find_entry, self.th_entry, self.iou_th_entry]
        if self.window.focus_get() not in entry_list:
            if platform.system() == 'Windows':
                state_1key = 8
                state_2key = 12
            else:  # 'Linux'
                state_1key = 16
                state_2key = 20

            if event.state == state_2key and event.keysym == 'Left':
                self.change_iou_threshold_button(-0.1)
            elif event.state == state_2key and event.keysym == 'Right':
                self.change_iou_threshold_button(0.1)

            elif event.state == state_1key and event.keysym == 'Left':
                self.change_threshold_button(-0.1)
            elif event.state == state_1key and event.keysym == 'Right':
                self.change_threshold_button(0.1)
            elif event.keysym == 'q':
                self.window.quit()
            elif event.keysym == 's':
                self.save_img()

            if self.button_clicked:
                self.button_clicked = False
            else:
                if event.keysym in ['KP_Enter', 'Return']:
                    self.listBox_obj.focus()
                    self.listBox_obj.select_set(0)
                elif event.keysym == 'Escape':
                    self.change_img()
                    self.listBox_img.focus()

    def combobox_change(self, event=None):
        self.listBox_img.focus()
        self.change_img()

    def clear_add_listBox_img(self):
        self.listBox_img.delete(0, 'end')  # delete listBox_img 0 ~ end items

        # add image name to listBox_img
        for item in self.img_list:
            self.listBox_img.insert('end', item)

        self.listBox_img.select_set(0)
        self.listBox_img.focus()
        self.change_img()

    def findname(self, event=None):
        self.find_name = self.find_entry.get()
        new_list = []

        if self.find_name == '':
            new_list = self.data_info.img_list
        else:
            for img_name in self.data_info.img_list:
                if self.find_name[0] == '!':
                    if self.find_name[1:] not in img_name:
                        new_list.append(img_name)
                else:
                    if self.find_name in img_name:
                        new_list.append(img_name)

        if len(new_list) != 0:
            self.img_list = new_list
            self.clear_add_listBox_img()
            self.clear_add_listBox_obj()
            self.button_clicked = True
        else:
            self.window.title("Can't find any image about '{}'".format(
                self.find_name))

    def run(self):
        self.window.title('DATASET : ' + self.data_info.dataset)
        self.window.geometry('1280x800+350+100')  # 初始界面宽高 左上角的坐标
        self.menubar.add_command(label='Save-All', command=self.save_all_images)
        self.menubar.add_command(label='Quit', command=self.window.quit)
        self.window.config(menu=self.menubar)  # display the menu
        self.scrollbar_img.config(command=self.listBox_img.yview)
        self.listBox_img.config(yscrollcommand=self.scrollbar_img.set)
        self.scrollbar_obj.config(command=self.listBox_obj.yview)
        self.listBox_obj.config(yscrollcommand=self.scrollbar_obj.set)

        layer1 = 0
        layer2 = 50

        # ======================= layer 1 =========================

        # combobox
        self.combo_label.grid(
            row=layer1 + 30,
            column=0,
            sticky=W + E + N + S,
            padx=3,
            pady=3,
            columnspan=6)
        self.combo_category.grid(
            row=layer1 + 30,
            column=6,
            sticky=W + E + N + S,
            padx=3,
            pady=3,
            columnspan=6)

        if self.data_info.det_file != '':
            # show det
            self.checkbn_det.grid(
                row=layer1 + 40,
                column=0,
                sticky=N + S,
                padx=3,
                pady=3,
                columnspan=4)
            # show det text
            self.checkbn_det_txt.grid(
                row=layer1 + 40,
                column=4,
                sticky=N + S,
                padx=3,
                pady=3,
                columnspan=2)
        if self.data_info.has_anno != False:
            # show gt
            self.checkbn_gt.grid(
                row=layer1 + 40,
                column=6,
                sticky=N + S,
                padx=3,
                pady=3,
                columnspan=4)
            # show gt text
            self.checkbn_gt_txt.grid(
                row=layer1 + 40,
                column=10,
                sticky=N + S,
                padx=3,
                pady=3,
                columnspan=2)

        # ======================= layer 2 =========================

        self.listBox_img_label.grid(
            row=layer2 + 0, column=0, sticky=N + S + E + W, columnspan=12)

        # find name
        self.find_label.grid(
            row=layer2 + 20, column=0, sticky=E + W, columnspan=4)
        self.find_entry.grid(
            row=layer2 + 20, column=4, sticky=E + W, columnspan=4)
        self.find_button.grid(
            row=layer2 + 20, column=8, sticky=E + W, pady=3, columnspan=4)

        self.scrollbar_img.grid(row=layer2 + 30, column=11, sticky=N + S + W)
        self.label_img.place(x=375, y=3, anchor=N + W)
        self.listBox_img.grid(
            row=layer2 + 30,
            column=0,
            sticky=N + S + E + W,
            pady=3,
            columnspan=11)

        if self.data_info.det_file != '':
            self.th_label.grid(
                row=layer2 + 40, column=0, sticky=E + W, columnspan=6)
            self.th_entry.grid(
                row=layer2 + 40, column=6, sticky=E + W, columnspan=3)
            self.th_button.grid(
                row=layer2 + 40, column=9, sticky=E + W, columnspan=3)

            if self.data_info.has_anno != False:
                self.iou_th_label.grid(
                    row=layer2 + 50, column=0, sticky=E + W, columnspan=6)
                self.iou_th_entry.grid(
                    row=layer2 + 50, column=6, sticky=E + W, columnspan=3)
                self.iou_th_button.grid(
                    row=layer2 + 50, column=9, sticky=E + W, columnspan=3)

            self.listBox_obj_label1.grid(
                row=layer2 + 60, column=0, sticky=E + W, pady=3, columnspan=12)

            if self.data_info.has_anno != False:
                self.listBox_obj_label2.grid(
                    row=layer2 + 70,
                    column=0,
                    sticky=E + W,
                    pady=2,
                    columnspan=12)

            self.scrollbar_obj.grid(
                row=layer2 + 80, column=11, sticky=N + S + W, pady=3)
            self.listBox_obj.grid(
                row=layer2 + 80,
                column=0,
                sticky=N + S + E + W,
                pady=3,
                columnspan=11)

        self.clear_add_listBox_img()
        self.listBox_img.bind('<<ListboxSelect>>', self.change_img)
        self.listBox_img.bind_all('<KeyRelease>', self.eventhandler)

        self.listBox_obj.bind('<<ListboxSelect>>', self.change_obj)

        self.th_entry.bind('<Return>', self.change_threshold)
        self.th_entry.bind('<KP_Enter>', self.change_threshold)
        self.iou_th_entry.bind('<Return>', self.change_iou_threshold)
        self.iou_th_entry.bind('<KP_Enter>', self.change_iou_threshold)
        self.find_entry.bind('<Return>', self.findname)
        self.find_entry.bind('<KP_Enter>', self.findname)

        self.combo_category.bind('<<ComboboxSelected>>', self.combobox_change)

        self.window.mainloop()


class aug_category:
    def __init__(self, categories):
        self.category = categories
        self.combo_list = categories.copy()
        self.combo_list.insert(0, 'All')
        self.all = True


if __name__ == '__main__':
    vis_tool().run()