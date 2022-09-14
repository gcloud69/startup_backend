
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pifuhd')))

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

from lib.mesh_util import save_obj_mesh_with_color, reconstruction
from lib.model import HGPIFuNetwNML, HGPIFuMRNet
from lib.options import BaseOptions



def crop_image(img, rect):
    x, y, w, h = rect

    left = abs(x) if x < 0 else 0
    top = abs(y) if y < 0 else 0
    right = abs(img.shape[1]-(x+w)) if x + w >= img.shape[1] else 0
    bottom = abs(img.shape[0]-(y+h)) if y + h >= img.shape[0] else 0
    
    if img.shape[2] == 4:
        color = [0, 0, 0, 0]
    else:
        color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    x = x + left
    y = y + top

    return new_img[y:(y+h),x:(x+w),:]

def fullbody_crop(pts):
    flags = pts[:,2] > 0.5      #openpose
    # flags = pts[:,2] > 0.2  #detectron
    check_id = [11,19,21,22]
    cnt = sum(flags[check_id])

    if cnt == 0:
        center = pts[8,:2].astype(np.int)
        pts = pts[pts[:,2] > 0.5][:,:2]
        radius = int(1.45*np.sqrt(((center[None,:] - pts)**2).sum(1)).max(0))
        center[1] += int(0.05*radius)
    else:
        pts = pts[pts[:,2] > 0.2]
        pmax = pts.max(0)
        pmin = pts.min(0)

        center = (0.5 * (pmax[:2] + pmin[:2])).astype(np.int)
        radius = int(0.65 * max(pmax[0]-pmin[0], pmax[1]-pmin[1]))

    x1 = center[0] - radius
    x2 = center[0] + radius
    y1 = center[1] - radius
    y2 = center[1] + radius

    return (x1, y1, x2-x1, y2-y1)

to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def prepare_input(image, keypoints, load_size=1024):
    # Calib
    keypoints = np.array(keypoints).reshape(-1,3)

    if image.shape[2] == 4:
        image = image / 255.0
        image[:,:,:3] /= image[:,:,3:] + 1e-8
        image = image[:,:,3:] * image[:,:,:3] + 0.5 * (1.0 - image[:,:,3:])
        image = (255.0 * image).astype(np.uint8)
    h, w = image.shape[:2]
    
    intrinsic = np.identity(4)
    trans_mat = np.identity(4)

    rect = fullbody_crop(keypoints)
    image = crop_image(image, rect)

    scale_im2ndc = 1.0 / float(w // 2)
    scale = w / rect[2]
    trans_mat *= scale
    trans_mat[3,3] = 1.0
    trans_mat[0, 3] = -scale*(rect[0] + rect[2]//2 - w//2) * scale_im2ndc
    trans_mat[1, 3] = scale*(rect[1] + rect[3]//2 - h//2) * scale_im2ndc
    
    intrinsic = np.matmul(trans_mat, intrinsic)
    im_512 = cv2.resize(image, (512, 512))
    image = cv2.resize(image, (load_size, load_size))

    image_512 = Image.fromarray(im_512[:,:,::-1]).convert('RGB')
    image = Image.fromarray(image[:,:,::-1]).convert('RGB')
    
    B_MIN = np.array([-1, -1, -1])
    B_MAX = np.array([1, 1, 1])
    projection_matrix = np.identity(4)
    projection_matrix[1, 1] = -1
    calib = torch.Tensor(projection_matrix).float()

    calib_world = torch.Tensor(intrinsic).float()

    # image
    image_512 = to_tensor(image_512)
    image = to_tensor(image)
    return {
        'name': 'test_image',
        'img': image.unsqueeze(0),
        'img_512': image_512.unsqueeze(0),
        'calib': calib.unsqueeze(0),
        'calib_world': calib_world.unsqueeze(0),
        'b_min': B_MIN,
        'b_max': B_MAX,
    }
    
def gen_mesh(res, net, cuda, data, save_path, thresh=0.5, use_octree=True, components=False):
    image_tensor_global = data['img_512'].to(device=cuda)
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter_global(image_tensor_global)
    net.filter_local(image_tensor[:,None])

    try:
        if net.netG.netF is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlF], 0)
        if net.netG.netB is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlB], 0)
    except:
        pass
    
    b_min = data['b_min']
    b_max = data['b_max']

    # save_img_path = save_path[:-4] + '.png'
    # save_img_list = []
    # for v in range(image_tensor_global.shape[0]):
    #     save_img = (np.transpose(image_tensor_global[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
    #     save_img_list.append(save_img)
    # save_img = np.concatenate(save_img_list, axis=1)
    # cv2.imwrite(save_img_path, save_img)

    verts, faces, _, _ = reconstruction(
        net, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=1000)
    verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
    color = np.zeros(verts.shape)
    interval = 1000
    for i in range(len(color) // interval + 1):
        left = i * interval
        if i == len(color) // interval:
            right = -1
        else:
            right = (i + 1) * interval
        net.calc_normal(verts_tensor[:, None, :, left:right], calib_tensor[:,None], calib_tensor)
        nml = net.nmls.detach().cpu().numpy()[0] * 0.5 + 0.5
        color[left:right] = nml.T

    save_obj_mesh_with_color(save_path, verts, faces, color)
    return save_path


class Pifu3DMGenerator():
    def __init__(self, checkpoint_path: str):
        self.cuda = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')

        state_dict = torch.load(checkpoint_path, map_location=self.cuda)   
        opt_netG = state_dict['opt_netG']
        self.opt = BaseOptions().parse()
        self.opt = state_dict['opt']
        self.opt.resolution = 256
        self.opt.loadSize = 512

        projection_mode =  'orthogonal'
        self.netG = HGPIFuNetwNML(opt_netG, projection_mode).to(device=self.cuda)
        self.netMR = HGPIFuMRNet(self.opt, self.netG, projection_mode).to(device=self.cuda)
        # load checkpoints
        self.netMR.load_state_dict(state_dict['model_state_dict'])


    def recon(self, image, keypoints, save_path):
        with torch.no_grad():
            self.netG.eval()
            self.netMR.eval()
            data = prepare_input(image, keypoints)
            return gen_mesh(self.opt.resolution, self.netMR, self.cuda, data, save_path, components=False)