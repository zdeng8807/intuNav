# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree

class DataSet(Dataset):
    def __init__(self, voc_root, transforms, train_set='train.txt'):
        self.root = os.path.join(voc_root, "dataset")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")
        txt_list = os.path.join(self.root, "ImageSets", "Main", train_set)
        with open(txt_list) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines()]
        json_file_box = "./intu_boxConifdences.json"
        json_file_name ="./intu_names.json"
        assert os.path.exists(json_file_box), "{} file not exist.".format(json_file_box)
        assert os.path.exists(json_file_name), "{} file not exist.".format(json_file_name)
        json_file_box = open(json_file_box, 'r')
        json_file_name = open(json_file_name, 'r')
        self.class_dict_box = json.load(json_file_box)
        self.class_dict_name = json.load(json_file_name)
        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        height_width = [data_height, data_width]
        img_path = os.path.join(self.img_root, data["filename"])

        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))
        assert "object" in data, "{} lack of object information.".format(xml_path)

        if data["object"][0]["name"] == "nosed" or data["object"][0]["name"] == "node" or data["object"][0]["name"] == 'd':
            data["object"][0]["name"] = 'nose'
        name = self.class_dict_name[data["object"][0]["name"]]

        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin, xmax = obj["bndbox"]["xmin"], obj["bndbox"]["xmax"]
            ymin, ymax = obj["bndbox"]["ymin"], obj["bndbox"]["ymax"]

            xmin, xmax, ymin, ymax = self.cycle_to_rectangle(xmin, xmax, ymin, ymax)

            xmin = float(xmin) / data_width
            xmax = float(xmax) / data_width
            ymin = float(ymin) / data_height
            ymax = float(ymax) / data_height

            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict_box[data["boxconfs"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        name = torch.as_tensor(name, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        height_width = torch.as_tensor(height_width, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["name"] = name
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["height_width"] = height_width
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def get_height_and_width(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        height_width = [data_height, data_width]
        if data["object"][0]["name"]=="nosed" or "node":
            data["object"][0]["name"] = 'nose'
        name = self.class_dict_name[data["object"][0]["name"]]
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin, xmax = obj["bndbox"]["xmin"], obj["bndbox"]["xmax"]
            ymin, ymax = obj["bndbox"]["ymin"], obj["bndbox"]["ymax"]
            xmin, xmax, ymin, ymax = self.cycle_to_rectangle(xmin, xmax, ymin, ymax)
            xmin = float(xmin) / data_width
            xmax = float(xmax) / data_width
            ymin = float(ymin) / data_height
            ymax = float(ymax) / data_height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict_box[data["boxconfs"]])
            iscrowd.append(int(obj["difficult"]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        name = torch.as_tensor(name, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        height_width = torch.as_tensor(height_width, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["name"] = name
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["height_width"] = height_width

        return target
    def cycle_to_rectangle(self, xmin, xmax, ymin, ymax):
        xmin, xmax, ymin, ymax = float(xmin), float(xmax), float(ymin), float(ymax)
        radius = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)

        left_x = xmin - radius
        left_y = ymin - radius
        right_x = xmin + radius
        right_y = ymin + radius
        xmin, xmax, ymin, ymax = left_x, right_x, left_y, right_y
        return xmin, xmax, ymin, ymax


    @staticmethod
    def collate_fn(batch):
        images, targets = tuple(zip(*batch))
        return images, targets
