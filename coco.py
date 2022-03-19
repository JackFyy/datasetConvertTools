import os
import json
from unicodedata import category
from lxml import etree
import xml.etree.cElementTree as ET
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

def seg_point(seg):
    points = []
    for i in range(0,len(seg[0])-1,2):
        x = int(seg[0][i])
        y = int(seg[0][i+1])
        points.append([x,y])
    return points
    
def read_coco():
    anno = 'instances_default.json'
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
        data = dict(
            
            version=None,
            flags=dict(),
                
            shapes = [],
            imagePath=None,
            imageData = None,
            imageHeight=550,
            imageWidth=900,
        )
        img_name = imgs[i]['file_name'].split('/')[1]
        print("img_name",img_name)
        height = imgs[i]['height']
        width = imgs[i]['width']
        img_id = imgs[i]['id']

        annos = df_anno[df_anno["image_id"].isin([img_id])]
        #print("annos",annos)
        if annos.empty:
            continue
        
        image = cv2.imread("cnc_damaged/data/"+img_name)
        for index, row in annos.iterrows():
            #print("row", row)
            seg = row['segmentation']
            bbox = row['bbox']
            category_id = row["category_id"]
            cate_name = categories[category_id]
            print("seg",seg)
            points = seg_point(seg)
            data['shapes'].append(
                dict(
                    label=cate_name,
                    points=points,
                    group_id=None,
                    shape_type="polygon",
                    flags=dict(),
                )
            )
        data['imagePath']=img_name
        img_path = "rgb/"+img_name
        cv2.imwrite(img_path, image)
        json_name = img_path.split('.')[0]+".json"
        with open(json_name,"w") as f:
            json.dump(data,f)
            print("write json down")

    


if __name__ == '__main__':
    read_coco()