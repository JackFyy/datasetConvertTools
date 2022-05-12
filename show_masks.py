import os
import cv2
import numpy as np
import json 
import glob


labels = ['pocketknife','rantene','bluerivet','Head&Shoulders','Cylindricalcleanser','sod',
                 'plum', 'bluecup', 'toy']

def seg_point(seg):
    points = []
    for i in range(0,len(seg[0])-1,2):
        x = int(seg[0][i])
        y = int(seg[0][i+1])
        points.append([x,y])
    return points
# color_masks = [
#     np.random.randint(0,256,(1,3), dtype=np.uint8)
#     for _ in range(9)
# ]
color_masks = [
    [[10,13,165]],[[14,28, 152]],[[104, 225, 158]],[[50, 158, 149]],[[184, 157, 222]],[[103,  40, 106]],
    [[ 80, 225,  66]],[[177, 120, 130]],[[119, 202, 157]]
]
color_masks = np.array(color_masks).astype(int)
print("color_masks",color_masks)

def show():
    rootpath = "/home/vision/project/DATASET/mygraspdataset/segmentattion/"
    imgfile=os.path.join(rootpath,"*.png")
    imglist=glob.glob(imgfile)
    #print("imgfile",imgfile)
    for img_p in imglist:
        #print("img_p",img_p.split('/')[-1])
        
        cur_json = img_p.split('.')[0] + '.json'
        # print(img_p)
        # print(cur_json)
        with open(cur_json, 'r', encoding='utf-8') as load_f:
            f = json.load(load_f)
        #print(f['shapes'])
        image = cv2.imread(img_p)
        #print("rootpath+img_p",rootpath+img_p)
        imageHeight = f['imageHeight']
        imageWidth = f['imageWidth']
        for curPt in f['shapes']:
            curSeg = curPt['points']
            curlabel = curPt['label']
            color_mask = color_masks[labels.index(curlabel)]
            cur_mask = np.zeros((imageHeight,imageWidth), dtype=np.uint8)
            polygon = np.array(curSeg, np.int32)
            #cv2.fillConvexPoly(cur_mask, polygon, (255, 255, 255))
            points = seg_point(curSeg)
            cv2.fillPoly(cur_mask,[polygon], (255, 255, 255))
            # cv2.imshow("img", cur_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #print("cur_mask",cur_mask.shape)
            cur_mask_bool = cur_mask.astype(bool)
            # print("cur_mask_bool",cur_mask_bool.shape)
            # print("cur_mask_bool[cur_mask_bool]",cur_mask_bool[cur_mask_bool].shape)
            image[cur_mask_bool] = image[cur_mask_bool]*0.4+color_mask*0.6
        image = cv2.resize(image,(640,384))
        cv2.imwrite("/home/vision/project/DATASET/mygraspdataset/gt_mask/"+img_p.split('/')[-1],image)


if __name__ == '__main__':
    show()