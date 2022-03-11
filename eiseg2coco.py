import os
import json
import cv2
from lxml import etree
import xml.etree.cElementTree as ET
import time
import pandas as pd
from tqdm import tqdm
import json
import argparse
import numpy as np
import glob

cropleft = 500
cropright= 1700
croptop=140
cropbottom=1070
def seg_crop(seg):
    points = []
    for i in range(len(seg)):
        # x = int(seg[0][i])-450
        # y = int(seg[0][i+1])-200
        x = int(seg[i][0]-cropleft)
        y = int(seg[i][1]-croptop)
        if x<0 or y<0:
            return 0
        points.append([x,y])
    return points

data_label = ["spoon_black","spoon_argent","Headphone_case","blue_box","orange_box", "yellow_box",
              "chewing_gum", "paper_towel","cup"]

def eiseg2coco():

    rootpath = "/home/vision/project/DATASET/myt_data/annotate/annotated"
    imgfile=os.path.join(rootpath,"*.jpg")
    imglist=glob.glob(imgfile)
    depthfile = os.path.join(rootpath,"*.png")
    depthlist=glob.glob(depthfile)

    for i, depthimgpath in enumerate(depthlist):
        #print(depthimgpath)
        depth = cv2.imread(depthimgpath)
        depth = depth[croptop:cropbottom, cropleft:cropright]
        depth_path ="cropedDepth/"+depthimgpath.split('/')[-1]
        print(depth_path)
        #raise
        cv2.imwrite(depth_path, depth)

    #读取每张图像和标签json文件
    for i, curimgpath in enumerate(imglist):
        curimgname = curimgpath.split('/')[-1]
        curimgID = curimgname.split('_')[1].split('.')[0]
        curjsonname = curimgname.split('.')[0]+".json"
        curjsonpath = rootpath+"/"+curjsonname
        print(curimgpath)
        print(curjsonpath)
        print(curjsonname)
        img = cv2.imread(curimgpath)
        # cv2.namedWindow("img",cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("img",1070,760)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        img = img[croptop:cropbottom, cropleft:cropright]
        data = dict(
            
            version=None,
            flags=dict(),
                
            shapes = [],
            imagePath=None,
            imageData = None,
            imageHeight=cropbottom-croptop,
            imageWidth=cropright-cropleft,
        )
        with open(curjsonpath, 'r', encoding='utf-8') as load_f:
            f = json.load(load_f)

        #处理每张图像中的每个目标物体标签
        for j,item in enumerate(f):
            cate_name = item['name']
            seg = item['points']
            if seg_crop(seg)==0:
                continue
            points = seg_crop(seg)
        
            data['shapes'].append(
                    dict(
                        label=cate_name,
                        points=points,
                        group_id=None,
                        shape_type="polygon",
                        flags=dict(),
                    )
                )
        data['imagePath']=curimgname
        imgp = "cropedRGB/"+curimgname
        cv2.imwrite(imgp, img)
        json_name = "cropedJson/"+curjsonname
        with open(json_name,"w") as f:
            json.dump(data,f)
            print("write json down")
        #raise







    # anno = "Cutting_tool_20211228.json"
    

    # with open(anno, 'r', encoding='utf-8') as load_f:
    #     f = json.load(load_f)

    
    # imgs = f['images']

    # df_cate = pd.DataFrame(f['categories'])
    # _ = df_cate.sort_values(["id"],ascending=True)
    # df_anno = pd.DataFrame(f['annotations'])
    # categories = dict(zip(df_cate.id.values, df_cate.name.values))
    # count = 0
    # for i in tqdm(range(len(imgs))):
    #     data = dict(
            
    #         version=None,
    #         flags=dict(),
                
    #         shapes = [],
    #         imagePath=None,
    #         imageData = None,
    #         imageHeight=700,
    #         imageWidth=1000,
    #     )

    #     xml_content = []
    #     file_name = imgs[i]['file_name']
    #     height = imgs[i]['height']
    #     img_id = imgs[i]['id']
    #     width = imgs[i]['width']
        
    #     annos = df_anno[df_anno["image_id"].isin([img_id])]
    #     count = 0
    #     #print("file_name",file_name)
    #     img = cv2.imread(file_name)
    #     img = img[300:1000, 650:1550]
    #     for index, row in annos.iterrows():
    #         count += 1
            
    #         seg = row['segmentation']
    #         bbox = row['bbox']
    #         print("bbox",bbox)
    #         center_x = bbox[0]+bbox[2]/2
    #         center_y = bbox[1]+bbox[3]/2
    #         if center_x<450 or center_x>1650 or center_y<200 or center_y>900:
    #             continue
            
    #         category_id = row["category_id"]
    #         cate_name = categories[category_id]
    #         if cate_name not in data_label:
    #             cate_name = "RassM18x1.5"
    #         #print("cate_name", cate_name)
    #         #print("seg", seg)
    #         points = seg_crop(seg)
    #         data['shapes'].append(
    #             dict(
    #                 label=cate_name,
    #                 points=points,
    #                 group_id=None,
    #                 shape_type="polygon",
    #                 flags=dict(),
    #             )
    #         )
    #     data['imagePath']=file_name.split('/')[1]
    #     img_path = "crop-RGB/"+file_name.split('/')[1]
    #     cv2.imwrite(img_path, img)
    #     json_name = img_path.split('.')[0]+".json"
    #     with open(json_name,"w") as f:
    #         json.dump(data,f)
    #         print("write json down")



eiseg2coco()