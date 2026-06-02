"""Run InterHand2.6M inference on a custom image."""
import sys
import os
import os.path as osp
import argparse
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
from utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image
from utils.vis import vis_keypoints, vis_3d_keypoints

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
parser.add_argument('--test_epoch', type=str, default='19')
parser.add_argument('--img', type=str, required=True)
parser.add_argument('--out_dir', type=str, default='.')
args = parser.parse_args()

cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

joint_num = 21
root_joint_idx = {'right': 20, 'left': 41}
joint_type = {'right': np.arange(0, joint_num), 'left': np.arange(joint_num, joint_num * 2)}
skeleton = load_skeleton(osp.join('..', 'data', 'InterHand2.6M', 'annotations', 'skeleton.txt'), joint_num * 2)

# Load model
model_path = osp.join('..', 'output', 'model_dump', 'snapshot_%d.pth.tar' % int(args.test_epoch))
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test', joint_num)
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# Load image
transform = transforms.ToTensor()
img_path = args.img
original_img = load_img(img_path)
h, w = original_img.shape[:2]
print('Image size: {} x {}'.format(w, h))

# Use full image center area as bbox (square crop)
size = min(w, h)
bbox = [(w - size) // 2, (h - size) // 2, size, size]
bbox = process_bbox(bbox, (h, w))
img, trans, inv_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, cfg.input_img_shape)
img = transform(img.astype(np.float32)) / 255
img = img.cuda()[None, :, :, :]

# Forward
inputs = {'img': img}
targets = {}
meta_info = {}
with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')

joint_coord = out['joint_coord'][0].cpu().numpy()
rel_root_depth = out['rel_root_depth'][0].cpu().numpy()
hand_type = out['hand_type'][0].cpu().numpy()

# Restore to original image space
joint_coord[:, 0] = joint_coord[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
joint_coord[:, 1] = joint_coord[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
joint_coord[:, :2] = np.dot(inv_trans, np.concatenate((joint_coord[:, :2], np.ones_like(joint_coord[:, :1])), 1).transpose(1, 0)).transpose(1, 0)
joint_coord[:, 2] = (joint_coord[:, 2] / cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size / 2)

# Relative root depth
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

print('Hand type probs: Right={:.4f}, Left={:.4f}'.format(hand_type[0], hand_type[1]))
print('Right hand exist: {} | Left hand exist: {}'.format(right_exist, left_exist))

# 2D visualization
vis_img = original_img.copy()[:, :, ::-1].transpose(2, 0, 1)
vis_keypoints(vis_img, joint_coord, joint_valid, skeleton, 'result_2d.jpg', save_path=args.out_dir)

# 3D visualization
vis_3d_keypoints(joint_coord, joint_valid, skeleton, 'result_3d.png', score_thr=0.0)
print('3D result saved to output/vis/result_3d.png')

# Print key joint predictions
print('\n' + '='*60)
print('Predicted Joint Coordinates (pixel x, pixel y, depth z-mm)')
print('='*60)
for i, sk in enumerate(skeleton):
    name = sk['name']
    x, y, z = joint_coord[i]
    valid = 'V' if joint_valid[i] else 'X'
    print(f'  [{valid}] {name:14s}  x={x:7.1f}  y={y:7.1f}  z={z:7.1f}')

print('\nDone. Results saved to {}'.format(args.out_dir))
