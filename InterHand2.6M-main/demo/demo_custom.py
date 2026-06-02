import sys
import os
import os.path as osp
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image, transform_input_to_output_space, trans_point2d
from utils.vis import vis_keypoints, vis_3d_keypoints

# GPU setup
gpu_ids = '0'
cfg.set_args(gpu_ids)
cudnn.benchmark = True

# joint info
joint_num = 21
root_joint_idx = {'right': 20, 'left': 41}
joint_type = {'right': np.arange(0, joint_num), 'left': np.arange(joint_num, joint_num * 2)}
skeleton = load_skeleton(osp.join('../data/InterHand2.6M/annotations/skeleton.txt'), joint_num * 2)

# Load latest snapshot
test_epoch = 19
model_path = osp.join('..', 'output', 'model_dump', 'snapshot_%d.pth.tar' % test_epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test', joint_num)
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# Load image
transform = transforms.ToTensor()
img_path = 'input.jpg'
original_img = load_img(img_path)
original_img_height, original_img_width = original_img.shape[:2]
print('Image size: {} x {}'.format(original_img_width, original_img_height))

# Use bbox covering most of the image (hand assumed to be the main subject)
margin_x = 20
margin_y = 30
bbox = [margin_x, margin_y,
        original_img_width - 2 * margin_x,
        original_img_height - 2 * margin_y]
print('Bbox: [xmin={}, ymin={}, w={}, h={}]'.format(bbox[0], bbox[1], bbox[2], bbox[3]))

bbox = process_bbox(bbox, (original_img_height, original_img_width, original_img_height))
img, trans, inv_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, cfg.input_img_shape)
img = transform(img.astype(np.float32)) / 255
img = img.cuda()[None, :, :, :]

# Forward pass
inputs = {'img': img}
targets = {}
meta_info = {}
with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')

img = img[0].cpu().numpy().transpose(1, 2, 0)
joint_coord = out['joint_coord'][0].cpu().numpy()
rel_root_depth = out['rel_root_depth'][0].cpu().numpy()
hand_type = out['hand_type'][0].cpu().numpy()

# Restore joint coords to original image space
joint_coord[:, 0] = joint_coord[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
joint_coord[:, 1] = joint_coord[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
joint_coord[:, :2] = np.dot(inv_trans,
                            np.concatenate((joint_coord[:, :2], np.ones_like(joint_coord[:, :1])), 1).transpose(1, 0)).transpose(1, 0)
joint_coord[:, 2] = (joint_coord[:, 2] / cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size / 2)
rel_root_depth = (rel_root_depth / cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root / 2)
joint_coord[joint_type['left'], 2] += rel_root_depth

# Handedness
joint_valid = np.zeros((joint_num * 2), dtype=np.float32)
right_exist = hand_type[0] > 0.5
left_exist = hand_type[1] > 0.5
if right_exist:
    joint_valid[joint_type['right']] = 1
if left_exist:
    joint_valid[joint_type['left']] = 1
print('Right hand: {}, Left hand: {}'.format(right_exist, left_exist))

# Visualize 2D
filename = 'result_2d.jpg'
vis_img = original_img.copy()[:, :, ::-1].transpose(2, 0, 1)
vis_img = vis_keypoints(vis_img, joint_coord, joint_valid, skeleton, filename, save_path='.')
print('2D result saved to {}'.format(osp.join(os.getcwd(), filename)))

# Visualize 3D
filename_3d = 'result_3d'
vis_3d_keypoints(joint_coord, joint_valid, skeleton, filename_3d)
print('3D result saved as {}'.format(osp.join(os.getcwd(), filename_3d + '.jpg')))
