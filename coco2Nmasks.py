  
import os
import json
from unicodedata import category
from lxml import etree
import xml.etree.cElementTree as ET
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

# coco有80类，这里写要进行二值化的类的名字
# 其他没写的会被当做背景变成黑色
classes_names = ['pocketknife','rantene','bluerivet','Head&Shoulders','Cylindricalcleanser','sod',
                 'plum', 'bluecup', 'toy']


def seg_point(seg):
    points = []
    for i in range(0,len(seg[0])-1,2):
        x = int(seg[0][i])
        y = int(seg[0][i+1])
        points.append([x,y])
    return points
    
def read_coco():
    anno = 'annotations_masks.json'
    with open(anno, 'r', encoding='utf-8') as load_f:
        f = json.load(load_f)
    
    imgs = f['images']
    df_cate = pd.DataFrame(f['categories'])
    _ = df_cate.sort_values(["id"], ascending=True)
    df_anno = pd.DataFrame(f['annotations'])
    categories = dict(zip(df_cate.id.values, df_cate.name.values))
    #print("imgs", imgs)
    # print("df_cate",df_cate)
    # print("df_anno",df_anno)
    # print("categories", categories)
    
    for i in tqdm(range(len(imgs))):
        
        img_name = imgs[i]['file_name'].split('/')[1]
        print("img_name",img_name)
        height = imgs[i]['height']
        width = imgs[i]['width']
        img_id = imgs[i]['id']

        annos = df_anno[df_anno["image_id"].isin([img_id])]
        #print("annos",annos)
        if annos.empty:
            continue
        
        image = cv2.imread("rgb/"+img_name.split('.')[0]+".png")
        depth = cv2.imread("ori_depth/"+img_name.split('.')[0]+".png")
        depth_path = "depth/"+img_name
        cv2.imwrite(depth_path, depth)
        for index, row in annos.iterrows():
            cur_mask = np.zeros((height,width,3),np.uint8)
            #print("row", row)
            seg = row['segmentation']
            bbox = row['bbox']
            category_id = row["category_id"]
            cate_name = categories[category_id]
            print("seg",seg)
            points = seg_point(seg)
            cv2.fillPoly(cur_mask, [np.array(points)], (255, 255, 255))
            index = classes_names.index(cate_name)
            img_path = "masks/"+str(index)+'/'+img_name
            cv2.imwrite(img_path, cur_mask)
        

if __name__ == '__main__':
    read_coco()





