import sys, os, os.path as osp
import numpy as np
import cv2, torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image

cfg.set_args('0')
cudnn.benchmark = True

joint_num = 21
root_joint_idx = {'right': 20, 'left': 41}
joint_type = {'right': np.arange(0, joint_num), 'left': np.arange(joint_num, joint_num * 2)}
skeleton = load_skeleton(osp.join('..', 'data', 'InterHand2.6M', 'annotations', 'skeleton.txt'), joint_num * 2)

model_path = osp.join('..', 'output', 'model_dump', 'snapshot_19.pth.tar')
model = get_model('test', joint_num)
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

transform = transforms.ToTensor()
original_img = load_img('input.jpg')
oh, ow = original_img.shape[:2]
print('Image: {} x {}'.format(ow, oh))

margin_x, margin_y = 20, 30
bbox = [margin_x, margin_y, ow - 2 * margin_x, oh - 2 * margin_y]
bbox = process_bbox(bbox, (oh, ow, oh))
img, trans, inv_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, cfg.input_img_shape)
img = transform(img.astype(np.float32)) / 255
img = img.cuda()[None, :, :, :]

with torch.no_grad():
    out = model({'img': img}, {}, {}, 'test')

joint_coord = out['joint_coord'][0].cpu().numpy()
rel_root_depth = out['rel_root_depth'][0].cpu().numpy()
hand_type = out['hand_type'][0].cpu().numpy()

# Restore to original image space
joint_coord[:, 0] = joint_coord[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
joint_coord[:, 1] = joint_coord[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
joint_coord[:, :2] = np.dot(inv_trans, np.concatenate((joint_coord[:, :2], np.ones_like(joint_coord[:, :1])), 1).transpose(1, 0)).transpose(1, 0)
joint_coord[:, 2] = (joint_coord[:, 2] / cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size / 2)
rel_root_depth = (rel_root_depth / cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root / 2)
joint_coord[joint_type['left'], 2] += rel_root_depth

print('')
print('=== Hand type probabilities ===')
print('Right hand: {:.4f}'.format(hand_type[0]))
print('Left hand:  {:.4f}'.format(hand_type[1]))

print('')
print('=== Right hand keypoints (x, y pixel | z depth mm) ===')
for i in range(joint_num):
    name = skeleton[i]['name']
    x, y, z = joint_coord[i]
    print('  {:20s}: x={:7.1f}  y={:7.1f}  z={:7.1f}'.format(name, x, y, z))

print('')
print('=== Left hand keypoints (x, y pixel | z depth mm) ===')
for i in range(joint_num, joint_num * 2):
    name = skeleton[i]['name']
    x, y, z = joint_coord[i]
    print('  {:20s}: x={:7.1f}  y={:7.1f}  z={:7.1f}'.format(name, x, y, z))

print('')
print('Relative root depth (left vs right): {:.1f}'.format(float(rel_root_depth)))
