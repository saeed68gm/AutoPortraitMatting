import numpy as np
import scipy.io as sio
import os
from PIL import Image
import math
from scipy import misc

class TestDataset:
    imgs = []
    max_batch = 0
    batch_size = 0
    cur_batch = 0 # index of batch generated
    cur_ind = -1 # index of current image in imgs
    img_width = 600
    img_height = 800

    def __init__(self, imgs_path, batch_size=1):
        self.imgs = sio.loadmat(imgs_path)['list'][0]
        self.batch_size = batch_size

    def next_batch(self):
        cur_imgs = []
        cur_orgs = []
        while len(cur_imgs) < self.batch_size: # if not enough, get the next image
            self.cur_ind += 1
            if self.cur_ind >= len(self.imgs):
                break
            cur_name = self.imgs[self.cur_ind]
            print ("getting image : ", cur_name)
            tmp_img, tmp_org = self.get_images(cur_name)
            if tmp_img is not None:
                cur_imgs.append(tmp_img)
                cur_orgs.append(tmp_org)
        if len(cur_imgs) == self.batch_size:
            rimat = np.zeros((self.batch_size, self.img_height, self.img_width, 6), dtype=np.float)
            org_mat = np.zeros((self.batch_size, self.img_height, self.img_width, 3), dtype=np.int)
            self.cur_batch += 1 # output a new batch
            for i in range(self.batch_size):
                rimat[i] = cur_imgs.pop(0)
                org_mat[i] = cur_orgs.pop(0)
            return rimat, org_mat
        return [], []

    def get_images(self, img_name):
        stp = str(img_name)
        img_path = 'data/output/' + stp + '.mat'
        import pdb; pdb.set_trace()  # breakpoint 6a66f711 //
        if os.path.exists(img_path):
            imat = sio.loadmat(img_path)['img']
            nimat = np.array(imat, dtype=np.float)
            h, w, _ = nimat.shape
            org_mat = np.zeros((h, w, 3), dtype=np.int)
            for i in range(h):
                for j in range(w):
                    org_mat[i][j][0] = round(nimat[i][j][2] * 255 + 122.675)
                    org_mat[i][j][1] = round(nimat[i][j][1] * 255 + 116.669)
                    org_mat[i][j][2] = round(nimat[i][j][0] * 255 + 104.008)
            return nimat, org_mat
        return None, None