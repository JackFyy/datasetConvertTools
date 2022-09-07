import os
import json
# import cv2
from lxml import etree
import xml.etree.cElementTree as ET
import time
import pandas as pd
from tqdm import tqdm
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/tong/coco.json', help='Directory to save dataset')
parser.add_argument('--saveroot', default='/tong/xml', help='Directory to save bbox results')

parser.add_argument('--score_data_root', default='/my_data/Cutting_tool_211109_D.json', help='Directory to save dataset')
parser.add_argument('--score_saveroot', default='/my_data/score_xml_label', help='Directory to save bbox results')

FLAGS = parser.parse_args()


root = os.getcwd()

xml_saveroot = os.path.join(root + FLAGS.saveroot)
json_label = os.path.join(root+FLAGS.data_root)

score_xml_saveroot = os.path.join(root + FLAGS.score_saveroot)
score_json_label = os.path.join(root + FLAGS.score_data_root)

def coco2voc():
    anno = json_label
    xml_dir = xml_saveroot

    with open(anno, 'r', encoding='utf-8') as load_f:
        f = json.load(load_f)


    imgs = f['images']

    df_cate = pd.DataFrame(f['categories'])
    _ = df_cate.sort_values(["id"],ascending=True)
    df_anno = pd.DataFrame(f['annotations'])
    categories = dict(zip(df_cate.id.values, df_cate.name.values))

    for i in tqdm(range(len(imgs))):
        xml_content = []
        file_name = imgs[i]['file_name']
        height = imgs[i]['height']
        img_id = imgs[i]['id']
        width = imgs[i]['width']

        xml_content.append("<annotation>")
        xml_content.append("    <folder>cutting2021</folder>")
        xml_content.append("	<filename>"+file_name+"</filename>")
        xml_content.append("	<size>")
        xml_content.append("		<width>"+str(width)+"</width>")
        xml_content.append("		<height>"+str(height)+"</height>")
        xml_content.append("	</size>")
        xml_content.append("	<segmented>0</segmented>")
        
        annos = df_anno[df_anno["image_id"].isin([img_id])]
        count = 0
        for index, row in annos.iterrows():
            count += 1
            bbox = row["bbox"]
            category_id = row["category_id"]
            cate_name = categories[category_id]

            
            xml_content.append("	<object>")
            xml_content.append("		<name>"+cate_name+"</name>")
            xml_content.append("		<pose>Unspecified</pose>")
            xml_content.append("		<truncated>0</truncated>")
            xml_content.append("		<difficult>0</difficult>")
            xml_content.append("		<bndbox>")
            xml_content.append("			<xmin>"+str(int(bbox[0]))+"</xmin>")
            xml_content.append("			<ymin>"+str(int(bbox[1]))+"</ymin>")
            xml_content.append("			<xmax>"+str(int(bbox[0]+bbox[2]))+"</xmax>")
            xml_content.append("			<ymax>"+str(int(bbox[1]+bbox[3]))+"</ymax>")
            xml_content.append("		</bndbox>")
            xml_content.append("	</object>")
        xml_content.append("    <object_number>"+str(count)+"</object_number>")
        xml_content.append("</annotation>")

        x = xml_content
        xml_content=[x[i] for i in range(0,len(x)) if x[i]!="\n"]


        xml_path = os.path.join(xml_dir,file_name.split('.')[-2] + '.xml')
        print("xml_path", xml_path)
        with open(xml_path, 'w+',encoding="utf8") as f:
            f.write('\n'.join(xml_content))
        xml_content[:]=[]


def score_map():
    anno = score_json_label
    xml_dir = score_xml_saveroot

    with open(anno, 'r', encoding='utf-8') as load_f:
        f = json.load(load_f)


    imgs = f['images']

    df_cate = pd.DataFrame(f['categories'])
    _ = df_cate.sort_values(["id"],ascending=True)
    df_anno = pd.DataFrame(f['annotations'])
    categories = dict(zip(df_cate.id.values, df_cate.name.values))

    for i in tqdm(range(len(imgs))):
        xml_content = []
        file_name = imgs[i]['file_name']
        print("file_name", file_name)
        height = imgs[i]['height']
        img_id = imgs[i]['id']
        width = imgs[i]['width']

        xml_content.append("<annotation>")
        xml_content.append("    <folder>cutting2021</folder>")
        xml_content.append("	<filename>"+file_name+"</filename>")
        xml_content.append("	<size>")
        xml_content.append("		<width>"+str(width)+"</width>")
        xml_content.append("		<height>"+str(height)+"</height>")
        xml_content.append("	</size>")
        xml_content.append("	<segmented>0</segmented>")
        
        annos = df_anno[df_anno["image_id"].isin([img_id])]
        count = 0
        for index, row in annos.iterrows():
            count += 1
            segmentation = row["segmentation"]
            category_id = row["category_id"]
            cate_name = categories[category_id]
            print("cate_name", cate_name)
            print("segmentation", segmentation)
            
            xml_content.append("	<object>")
            xml_content.append("		<name>"+cate_name+"</name>")
            xml_content.append("		<pose>Unspecified</pose>")
            xml_content.append("		<truncated>0</truncated>")
            xml_content.append("		<difficult>0</difficult>")
            xml_content.append("		<bndbox>")

            print("len(segmentation[0])", len(segmentation[0]))
            ss = [[segmentation[0][k], segmentation[0][k+1]] for k in range(0, len(segmentation[0])-1, 2)]
            ss = np.array(ss)
            x = ss[:, 0]
            y = ss[:, 1]
            print("ss", ss)
            print("x", x)
            print("y", y)
            xml_content.append("			<segmentation_x>"+str(x[:])+"</segmentation_x>")
            xml_content.append("			<segmentation_y>"+str(y)+"</segmentation_y>")
            # for j in range(0, len(segmentation[0])-1, 2):
            #     xml_content.append("			<points>"+str(int(segmentation[0][j]))+","+str(int(segmentation[0][j+1]))+"</points>")
                # xml_content.append("			<xmin>"+str(int(bbox[0]))+"</xmin>")
                # xml_content.append("			<ymin>"+str(int(bbox[1]))+"</ymin>")
                # xml_content.append("			<xmax>"+str(int(bbox[0]+bbox[2]))+"</xmax>")
                # xml_content.append("			<ymax>"+str(int(bbox[1]+bbox[3]))+"</ymax>")

            xml_content.append("		</bndbox>")
            xml_content.append("	</object>")
        xml_content.append("    <object_number>"+str(count)+"</object_number>")
        xml_content.append("</annotation>")

        x = xml_content
        xml_content=[x[i] for i in range(0,len(x)) if x[i]!="\n"]


        xml_path = os.path.join(xml_dir,file_name.split('.')[-2] + '.xml')
        print("xml_path", xml_path)
        with open(xml_path, 'w+',encoding="utf8") as f:
            f.write('\n'.join(xml_content))
        xml_content[:]=[]


coco2voc()