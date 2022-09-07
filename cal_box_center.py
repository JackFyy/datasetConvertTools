import os
import numpy as np
from PIL import Image
#import open3d as o3d
#from utils.xmlhandler import xmlReader
import scipy.io as scio
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
#from multiprocessing import Process
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/my_data/bbox_xml_label/Cutting_tool_211109', help='Directory to save dataset')
parser.add_argument('--saveroot', default='/my_data/center_maps', help='Directory to save bbox results')
parser.add_argument('--save_visu', default='True', action='store_true', help='Whether to save visualizations')
parser.add_argument('--camera', default='realsense', help='camera to use [default: realsense]')
parser.add_argument('--pool_size', type=int, default=10, help='How many threads to use')
FLAGS = parser.parse_args()

root = os.getcwd()
print("root", root)

DATASET_ROOT = root + FLAGS.data_root
scenedir = root + FLAGS.data_root + '/scenes/scene_{}/{}'
bbox_saveroot = os.path.join(root + FLAGS.saveroot, 'bbox_anno')
center_saveroot = os.path.join(root + FLAGS.saveroot, 'center_anno')
visu_saveroot = os.path.join(root + FLAGS.saveroot, 'visu')
# mask_saveroot = ''
print("DATASET_ROOT", DATASET_ROOT)
print("scenedir", scenedir)
print("bbox_saveroot", bbox_saveroot)
print("center_saveroot", center_saveroot)
print("visu_saveroot", visu_saveroot)

# root_path = os.path.abspath(os.getcwd())
# img_path = os.path.join(root_path, "label/Cutting_tool_20210705")
# print("img_path", img_path)


def get_center_bbox(img_path):
    #$print('Scene {}, {}'.format('scene_%04d'%scene_idx, camera))
    img_list = os.listdir(img_path)
    print(img_list)
    
    image_name = [i for i in(img_list) if i.endswith(".jpg")]
    print("image_name", image_name)
    
    
    
    # bbox_list_scene = []
    # center_list_scene = []
    # mask_list_scene = []
    
    countID = 0
    for i, img_name in enumerate(image_name):
        # camera_pose = camera_poses[anno_idx]
        # # print('camera pose')
        # # print(camera_pose)
        # if align:
        #     align_mat = np.load(os.path.join(DATASET_ROOT, 'scenes', 'scene_%04d'%scene_idx, camera, 'cam0_wrt_table.npy'))
        #     camera_pose = align_mat.dot(camera_pose)
        
        #rgb_dir = os.path.join(scenedir.format('%04d'%scene_idx, camera), 'rgb', '%04d'%anno_idx+'.png')

        rgb_dir = os.path.join(img_path, img_name)
        img = Image.open(rgb_dir)
        rgb_image = np.array(img, dtype=np.float32)
        
        
        # model_list, _, _ = generate_scene_model(DATASET_ROOT, 'scene_%04d'%scene_idx, anno_idx, return_poses=True, camera=camera)
        xml_name = img_name.split('.')[0] + '.xml'
        print("os.path.join(img_path, xml_name)", os.path.join(img_path, xml_name))
        label_xml = open(os.path.join(img_path, xml_name))
        
        tree = ET.parse(label_xml)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        bbox_list_single = []
        center_list_single = []
        mask_list_single = []

        #for i, model in enumerate(model_list):
        for obj in root.iter('object'):
           
            
            #points = np.array(model.points)
            # print('points:', points.shape)
            #center = np.mean(points, keepdims=True, axis=0)

            #x, y, _ = points2depth(points, scene_idx, camera)
            # print('center:', center.shape)
            #center_x, center_y, _ = points2depth(center, scene_idx, camera)

            # valid_y = y
            # valid_x = x

            # min_x = valid_x.min()
            # min_y = valid_y.min()
            
            # max_x = valid_x.max()
            # max_y = valid_y.max()
            xmlbox = obj.find('bndbox')
            min_x = int(xmlbox.find('xmin').text)
            min_y = int(xmlbox.find('ymin').text)
            max_x = int(xmlbox.find('xmax').text)
            max_y = int(xmlbox.find('ymax').text)

            center_x = [(min_x + max_x)//2]
            center_y = [(min_y + max_y)//2]
            print("center_x", center_x)

            assert center_x[0] > min_x and center_x[0] < max_x, 'center x out of bbox'
            assert center_y[0] > min_y and center_y[0] < max_y, 'center y out of bbox'

            if not ((center_y[0] >= 0 ) & (center_y[0] < h) & (center_x[0] >= 0 ) & (center_x[0] < w)):
                mask_list_single.append(0)
            else:
                mask_list_single.append(1)

            bbox = np.array([min_y, min_x, max_y, max_x], dtype=np.int32)
            bbox_list_single.append(bbox[np.newaxis, :])

            
            

            center_pix = np.concatenate([center_y, center_x], axis=0)[np.newaxis, :]
            print('center_pix:', center_pix)
            center_list_single.append(center_pix)
            print("center_list_single", center_list_single)
            #raise

            ############### my ########### 
            # cropped_depth = depth_img.crop((min_x-10, min_y-10, max_x+10, max_y+10))

            
            # cropped = img.crop((min_x-10, min_y-10, max_x+10, max_y+10))
            # print(cropped.size)
            # w = cropped.size[0]
            # h = cropped.size[1]
            # c_x = w//2
            # c_y = h//2
            # cc = [c_y, c_x]
            # print("cc", cc)
            # print('Saving:/home/vision/project/suctionnet/my_data/new_bbox/img'+'/%d'%countID+'.jpg')
            # cropped.save('/home/vision/project/suctionnet/my_data/new_bbox/img'+'/%d'%countID+'.jpg')
            # print('Saving:/home/vision/project/suctionnet/my_data/new_bbox/label'+'/%d'%countID+'.npz')
            # np.savez('/home/vision/project/suctionnet/my_data/new_bbox/label'+'/%d'%countID+'.npz', cc)

            # cropped_depth.save('/home/vision/project/suctionnet/my_data/new_bbox/depth'+'/%d'%countID+'.png')
            
            if FLAGS.save_visu:
                rgb_image[max(min_y, 0): min(max_y, h), max(min_x, 0): min(max_x, w), :] *= 0.4
                cv2.circle(rgb_image, (center_x[0], center_y[0]), 10, (255,0,0), -1)
                
            
            ############### my ########### 

        bbox_single = np.concatenate(bbox_list_single, axis=0)[np.newaxis, :, :]                
        mask_single = np.array(mask_list_single, dtype=np.bool)[np.newaxis, :]
        
        center_single = np.concatenate(center_list_single, axis=0)[np.newaxis, :, :]
        

        if FLAGS.save_visu:
            if (mask_single == 0).sum() == 0:
                rgb_image = rgb_image.astype(np.uint8)
                im = Image.fromarray(rgb_image)
                
                os.makedirs(visu_saveroot, exist_ok=True)
                print('Saving:', visu_saveroot+'/%s'%img_name.split('.')[0]+'.jpg')
                im.save(visu_saveroot+'/%s'%img_name.split('.')[0]+'.jpg')
        
    
       
        
        bbox_dir = bbox_saveroot
        os.makedirs(bbox_dir, exist_ok=True)
        print('Saving:', bbox_dir + '/%s'%img_name.split('.')[0]+'.npz')
        np.savez(bbox_dir + '/%s'%img_name.split('.')[0]+'.npz', bbox_single)

        center_dir = center_saveroot
        os.makedirs(center_dir, exist_ok=True)
        print('Saving:', center_dir + '/%s'%img_name.split('.')[0]+'.npz')
        np.savez(center_dir + '/%s'%img_name.split('.')[0] +'.npz', center_single)

    

if __name__ == "__main__":
    
    
    get_center_bbox(DATASET_ROOT) 

    