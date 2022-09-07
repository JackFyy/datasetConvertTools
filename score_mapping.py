import os
import numpy as np
from PIL import Image
#from utils.xmlhandler import xmlReader
import scipy.io as scio
from PIL import Image
import cv2
#from transforms3d.euler import euler2mat, quat2mat
from multiprocessing import Process
import argparse
import xml.etree.ElementTree as ET
import json
import pandas as pd
from tqdm import tqdm

from skimage import io,data


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/my_data/score_xml_label/Cutting_tool_211109', help='Directory to save dataset')
parser.add_argument('--saveroot', default='/my_data/deeplab', help='Directory to save bbox results')
parser.add_argument('--score_data_root', default='/my_data/Cutting_tool_211109_D.json', help='Directory to save dataset')
FLAGS = parser.parse_args()

root = os.getcwd()
print("root", root)

DATASET_ROOT = root + FLAGS.data_root
score_saveroot = os.path.join(root + FLAGS.saveroot, 'score_anno')
visu_saveroot = os.path.join(root + FLAGS.saveroot, 'mask')
json_label = os.path.join(root+FLAGS.data_root)
score_json_label = os.path.join(root + FLAGS.score_data_root)


def score_mapping(img_path):

    anno = score_json_label
    with open(anno, 'r', encoding='utf-8') as load_f:
        f = json.load(load_f)


    imgs = f['images']

    df_cate = pd.DataFrame(f['categories'])
    _ = df_cate.sort_values(["id"],ascending=True)
    df_anno = pd.DataFrame(f['annotations'])
    categories = dict(zip(df_cate.id.values, df_cate.name.values))

    for i in tqdm(range(len(imgs))):
        file_name = imgs[i]['file_name']
        img_name = file_name.split('/')[1]
        
        height = imgs[i]['height']
        img_id = imgs[i]['id']
        width = imgs[i]['width']

        rgb_dir = os.path.join(img_path, img_name)
        rgb_image = np.array(Image.open(rgb_dir), dtype=np.float32)

        # k_size = 31
        # sigma = 5
        score_image = np.zeros([height, width], dtype=np.float32)
        
        annos = df_anno[df_anno["image_id"].isin([img_id])]
        count = 0
        for index, row in annos.iterrows():
            print("index",index)
            count += 1
            segmentation = row["segmentation"]
            category_id = row["category_id"]
            cate_name = categories[category_id]
            
            ss = [[segmentation[0][k], segmentation[0][k+1]] for k in range(0, len(segmentation[0])-1, 2)]
            
            ss = np.array(ss)
            
            x_cor = ss[:, 0]
            y_cor = ss[:, 1]
            
            x_cor = np.array(list(map(int, x_cor[:]))).reshape(x_cor.shape[0],1)
            y_cor = np.array(list(map(int, y_cor[:]))).reshape(y_cor.shape[0],1)
            # im = np.zeros(图像对应尺寸, dtype="uint8")
            print("x_cor", x_cor)
            print("y_cor", y_cor)
            cor_xy = np.hstack((x_cor, y_cor))
            print("cor_xy", cor_xy)
            cv2.fillPoly(score_image, [cor_xy], 1)
            mask_array = score_image

            
        
        #保存结果
        # print('Saving:', score_saveroot + '/%s'%img_name.split('.')[0]+'.npz')
        # np.savez(score_saveroot + '/%s'%img_name.split('.')[0]+'.npz', score_image)

        # #io.imsave("/home/vision/project/suctionnet/my_data/score_maps/score_anno/mask_"+img_name,score_image)
        # #可视化
        # score_image *= 255
            
        # score_image = score_image.clip(0, 255)
        score_image = score_image.astype(np.uint8)
        # score_image = cv2.applyColorMap(score_image, cv2.COLORMAP_RAINBOW)
        # rgb_image = 0.5 * rgb_image + 0.5 * score_image
        # rgb_image = rgb_image.astype(np.uint8)
        im = Image.fromarray(score_image)
        
        # #visu_dir = os.path.join(visu_saveroot, 'scene_'+str(scene_idx), camera, 'visu')
        # os.makedirs(visu_saveroot, exist_ok=True)
        print('Saving:', visu_saveroot+'/'+img_name)
        saname = img_name.split('.')[0]
        print("#########", visu_saveroot+'/'+saname + '.png')
        
        im.save(visu_saveroot+'/'+saname + '.png')

       

score_mapping(DATASET_ROOT)