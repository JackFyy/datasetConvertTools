import cv2
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def drawGaussian(img, pt, score, sigma=1):
    """Draw 2d gaussian on input image.
    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    pt: list or tuple
        A point: (x, y).
    sigma: int
        Sigma of gaussian distribution.
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.
    """
    # img = to_numpy(img)
    tmp_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.float32)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    # print('x:', x.shape)
    y = x[:, np.newaxis]
    # print('x:', x.shape)
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    
    # print('g:', g.shape)
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) * score
    g = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    # g = np.concatenate([g[..., np.newaxis], np.zeros([g.shape[0], g.shape[1], 2], dtype=np.float32)], axis=-1)
    
    tmp_img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g
    # img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g
    
    img += tmp_img
    tmp_img = tmp_img*255
    return tmp_img

def calcentermap():
    rootpath = "/home/vision/project/DATASET/myt_data/annotate/"
    imgfile=os.path.join(rootpath,"centermap/*.jpg")
    imglist=glob.glob(imgfile)
    
    for img_p in imglist:
        cur_json = img_p.split('.')[0] + '.json'
        print(img_p)
        print(cur_json)
        with open(cur_json, 'r', encoding='utf-8') as load_f:
            f = json.load(load_f)
        print(f['shapes'])
        height = f['imageHeight']
        width = f['imageWidth']
        center_map = np.zeros([height, width], dtype=np.float32)
        for curPt in f['shapes']:
            curPt = curPt['points'][0]
            print("curPt", curPt)
            center_map += drawGaussian(center_map, curPt, 1, sigma=20)
            print(curPt)
        imgp = rootpath + 'centermap_mask/'+ img_p.split('/')[-1]
        # print("imgp",imgp)
        # img = mpimg.imread(img_p)
        # plt.subplot(1,2,1)
        # plt.imshow(img)
        # plt.subplot(1,2,2)
        # plt.imshow(center_map)
        # plt.show()
        # raise
        #center_map = center_map.astype(np.uint8)
        cv2.imwrite(imgp, center_map)
        


if __name__ == "__main__":
    calcentermap()