#!/usr/bin/env python3
"""
An example script to run JAANet sub-networks on a real-time or video input,
using RetinaFace (face detection) + face-alignment (68 landmarks) + 68->49
mapping + crop/resize for JAANet.
"""
import sys
sys.path.append("/app/Pytorch_Retinaface")  # DockerでCOPYしたパス
# ↑ これで "import models" や "import utils" がトップとして認識される


import os
import cv2
import numpy as np
import torch
import argparse
from PIL import Image

# ---------- For face detection / alignment (install these separately) ----------
# pip install git+https://github.com/serengil/retinaface.git
# pip install face-alignment
# ---------- Pytorch_Retinaface Imports ----------
# (assuming you cloned https://github.com/biubug6/Pytorch_Retinaface.git
#  and placed it so that "from Pytorch_Retinaface.models.retinaface import RetinaFace" works)
from Pytorch_Retinaface.models.retinaface import RetinaFace
from Pytorch_Retinaface.data import cfg_mnet, cfg_re50
from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms


import face_alignment

################################################################################
# 1) PyTorch RetinaFace model load
################################################################################
def load_retinaface_model(trained_model_path, network='mobile0.25', cpu=False):
    """Load PyTorch RetinaFace with specified backbone and weights."""
    if network == 'mobile0.25':
        cfg = cfg_mnet
    else:
        cfg = cfg_re50
    net = RetinaFace(cfg=cfg, phase='test')
    device = torch.device("cpu" if cpu else "cuda")

    # load pretrained
    print(f"Loading pretrained model from {trained_model_path}")
    pretrained_dict = torch.load(trained_model_path, map_location=device)
    # Some old checkpoints have 'module.' prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if k.startswith('module.'):
            k2 = k[7:]
        else:
            k2 = k
        new_state_dict[k2] = v
    net.load_state_dict(new_state_dict, strict=False)

    net = net.to(device)
    net.eval()
    return net, cfg

################################################################################
# 2) detect_faces_pytorch_retinaface
#    (adapted from test_widerface.py but simplified for single-frame usage)
################################################################################
def detect_faces_pytorch_retinaface(net, cfg, frame_bgr,
                                    confidence_threshold=0.5,
                                    nms_threshold=0.4,
                                    top_k=5000,
                                    keep_top_k=750,
                                    use_cpu=False):
    """
    Args:
      net: loaded RetinaFace model (phase='test', eval mode)
      cfg: cfg_mnet or cfg_re50
      frame_bgr: numpy (H,W,3)
      confidence_threshold: filter out low scores
      nms_threshold: for py_cpu_nms
      ...
    Returns:
      A list of dict, e.g.:
      [
        {
          'facial_area': [x1,y1,x2,y2],
          'score': float,
          'landmarks': [[lx0,ly0], [lx1,ly1], ...], # 5 points if needed
        },
        ...
      ]
    """
    device = torch.device("cpu" if use_cpu else "cuda")

    img_raw = frame_bgr.copy()
    # Convert to float
    img = np.float32(img_raw)

    # no fancy resizing here, or follow exactly test_widerface style:
    # e.g. subtract mean
    img -= (104, 117, 123)
    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    # forward
    loc, conf, landms = net(img)
    # decode
    im_height, im_width = frame_bgr.shape[:2]
    scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)
    boxes = decode(loc.data.squeeze(0), priors.data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()

    scores = conf.data.squeeze(0).cpu().numpy()[:,1]

    landms = decode_landm(landms.data.squeeze(0), priors.data, cfg['variance'])
    scale1 = torch.Tensor([
        im_width, im_height, im_width, im_height,
        im_width, im_height, im_width, im_height,
        im_width, im_height
    ]).to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # filter low score
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # sort
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # limit top_k
    dets = dets[:keep_top_k,:]
    landms = landms[:keep_top_k,:]

    # package results
    results = []
    for i in range(dets.shape[0]):
        xmin = int(dets[i,0])
        ymin = int(dets[i,1])
        xmax = int(dets[i,2])
        ymax = int(dets[i,3])
        score = float(dets[i,4])

        lmk = landms[i].reshape((5,2))
        lmk = lmk.astype(int)

        results.append({
            'facial_area': [xmin,ymin,xmax,ymax],
            'score': score,
            'landmarks': lmk.tolist()
        })

    return results

# ---------- JAANet Imports (network definitions) ----------
import network  # from your JAANet code
# e.g. "network_dict" must contain the sub-networks: region_learning, align_net, local_attention_refine, ...
# Also, ensure you have an __init__.py or similar so that "import network" works.

# ---------- Torch transforms for normalization ----------
import torchvision.transforms as T

###############################################################################
# 68 -> 49 Landmark Mapping (Example)
###############################################################################

# 既存の align_face_49pts (face_transform.py) を貼り付ける
# ※ ここでは元コードから関数部分だけ抜き出し
def align_face_49pts(img_bgr, img_land_49, box_enlarge, img_size):
    """
    img_bgr: OpenCVのBGR画像 (H,W,3)
    img_land_49: shape=(49*2,) のフラットランドマーク (0-based)
    box_enlarge: 2.9 (学習時)
    img_size: 200 (学習時)
    """
    import math
    
    # 例：19,20,21,22,23,24 が左目, 25..30 が右目 etc.
    # 下記は学習時コードのまま
    leftEye0 = (img_land_49[2*19] + img_land_49[2*20] + img_land_49[2*21] + 
                img_land_49[2*22] + img_land_49[2*23] + img_land_49[2*24]) / 6.0
    leftEye1 = (img_land_49[2*19+1] + img_land_49[2*20+1] + img_land_49[2*21+1] + 
                img_land_49[2*22+1] + img_land_49[2*23+1] + img_land_49[2*24+1]) / 6.0
    rightEye0 = (img_land_49[2*25] + img_land_49[2*26] + img_land_49[2*27] + 
                 img_land_49[2*28] + img_land_49[2*29] + img_land_49[2*30]) / 6.0
    rightEye1 = (img_land_49[2*25+1] + img_land_49[2*26+1] + img_land_49[2*27+1] + 
                 img_land_49[2*28+1] + img_land_49[2*29+1] + img_land_49[2*30+1]) / 6.0

    deltaX = (rightEye0 - leftEye0)
    deltaY = (rightEye1 - leftEye1)
    l = math.sqrt(deltaX**2 + deltaY**2)
    sinVal = deltaY / l
    cosVal = deltaX / l

    mat1 = np.mat([[cosVal, sinVal, 0],
                   [-sinVal, cosVal, 0],
                   [0, 0, 1]])

    # 追加で参照している点(13,31,37など)をまとめて座標変換下見
    mat2 = np.mat([
        [leftEye0, leftEye1, 1],
        [rightEye0, rightEye1, 1],
        [img_land_49[2*13], img_land_49[2*13+1], 1],
        [img_land_49[2*31], img_land_49[2*31+1], 1],
        [img_land_49[2*37], img_land_49[2*37+1], 1]
    ])
    mat2 = (mat1 * mat2.T).T

    cx = float((max(mat2[:, 0]) + min(mat2[:, 0])) * 0.5)
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1])) * 0.5)

    width_range = float(max(mat2[:, 0]) - min(mat2[:, 0]))
    height_range = float(max(mat2[:, 1]) - min(mat2[:, 1]))
    if width_range > height_range:
        halfSize = 0.5 * box_enlarge * width_range
    else:
        halfSize = 0.5 * box_enlarge * height_range

    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.mat([[scale, 0, scale*(halfSize - cx)],
                   [0, scale, scale*(halfSize - cy)],
                   [0, 0, 1]])
    mat = mat3 * mat1

    # warpAffine
    aligned_img = cv2.warpAffine(img_bgr, mat[0:2, :], (img_size, img_size),
                                 flags=cv2.INTER_LINEAR, borderValue=(128,128,128))

    # ランドマークのアフィン変換
    N_points = int(len(img_land_49)//2)
    land_3d = np.ones((N_points, 3))
    for i in range(N_points):
        land_3d[i,0] = img_land_49[2*i]
        land_3d[i,1] = img_land_49[2*i+1]

    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = new_land[:, 0:2].reshape(-1)  # shape(49*2,)

    return aligned_img, new_land


###############################################
# 66->49 用マッピング (仮例) 
# reflect_66 = [18..66]
###############################################
import numpy as np
reflect_66 = np.array([18,19,20,21,22,23,24,25,26,27,28,29,
                       30,31,32,33,34,35,36,37,38,39,40,41,
                       42,43,44,45,46,47,48,49,50,51,52,53,
                       54,55,56,57,58,59,60,61,62,63,64,65,66])

def map_66_to_49(land66):
    """
    land66: shape (66,2)
    reflect_66: 1-based => 18..66
    output: (49,2)
    """
    land49 = np.zeros((49,2), dtype=np.float32)
    for i in range(49):
        idx_66 = reflect_66[i] - 1  # 0-based
        land49[i] = land66[idx_66]
    return land49


###############################################
# 新しい preprocess_for_jaanet
# 学習時と同じアフィン変換を踏襲
###############################################
def preprocess_for_jaanet(frame_rgb, land66, device='cpu',
                          box_enlarge=2.9, img_size=200,
                          final_crop_size=176):
    """
    1) 66->49 (reflect_66)
    2) align_face_49pts (目を水平にし、box_enlarge=2.9で200x200にwarp)
    3) 200x200 -> 176x176 (学習コードに合わせるか任意)
    4) Normalize(mean=0.5, std=0.5)
    return: (1,3,176,176) tensor, new_land(49*2)
    """
    # 1) 66->49
    land49_2d = map_66_to_49(land66)
    # flatten => shape(98,)
    land49_flat = land49_2d.reshape(-1)

    # Convert frame_rgb (H,W,3, RGB) => BGR if face_transform.py expects BGR
    # face_transform.py uses cv2.imread => BGR. 
    # So we do color channel swap if needed:
    img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # 2) align_face_49pts => (200x200)
    aligned_img_bgr, new_land_49 = align_face_49pts(
        img_bgr, land49_flat,
        box_enlarge=box_enlarge,
        img_size=img_size
    )
    # new_land_49 shape(98,) => 49点 x,y

    # 3) 200->176
    # ここで一度 PIL に変換
    pil_aligned = Image.fromarray(aligned_img_bgr[:,:,::-1])  # BGR->RGB

    if final_crop_size < img_size:
        pil_aligned = pil_aligned.resize((final_crop_size, final_crop_size), Image.BILINEAR)

    # 4) Torch Transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5],
                    std=[0.5,0.5,0.5])
    ])
    tensor_img = transform(pil_aligned).unsqueeze(0).to(device)  # (1,3,176,176) or (1,3,200,200)

    return tensor_img, new_land_49


###############################################################################
# Load JAANet Subnetworks
###############################################################################
def load_jaanet_subnetworks(config, device):
    """
    This replicates the 'test_JAAv2.py' approach:
    1) Create subnetwork instances
    2) Load state_dict from .pth
    3) Set them to eval()
    4) Return as a dict
    """
    # region_learning
    region_learning = network.network_dict[config.region_learning](input_dim=3, unit_dim=config.unit_dim)
    align_net = network.network_dict[config.align_net](
        crop_size=config.crop_size,
        map_size=config.map_size,
        au_num=config.au_num,
        land_num=config.land_num,
        input_dim=config.unit_dim * 8
    )
    local_attention_refine = network.network_dict[config.local_attention_refine](au_num=config.au_num, unit_dim=config.unit_dim)
    local_au_net = network.network_dict[config.local_au_net](au_num=config.au_num, input_dim=config.unit_dim*8, unit_dim=config.unit_dim)
    global_au_feat = network.network_dict[config.global_au_feat](input_dim=config.unit_dim*8, unit_dim=config.unit_dim)
    au_net = network.network_dict[config.au_net](au_num=config.au_num, input_dim=12000, unit_dim=config.unit_dim)

    # Move to device
    region_learning = region_learning.to(device)
    align_net = align_net.to(device)
    local_attention_refine = local_attention_refine.to(device)
    local_au_net = local_au_net.to(device)
    global_au_feat = global_au_feat.to(device)
    au_net = au_net.to(device)

    # Load weights from epoch
    epoch = config.start_epoch  # must > 0
    checkpoint_dir = os.path.join(config.write_path_prefix, config.run_name)
    if epoch <= 0:
        raise RuntimeError("start_epoch should be larger than 0")

    region_learning.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"region_learning_{epoch}.pth")))
    align_net.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"align_net_{epoch}.pth")))
    local_attention_refine.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"local_attention_refine_{epoch}.pth")))
    local_au_net.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"local_au_net_{epoch}.pth")))
    global_au_feat.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"global_au_feat_{epoch}.pth")))
    au_net.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"au_net_{epoch}.pth")))

    region_learning.eval()
    align_net.eval()
    local_attention_refine.eval()
    local_au_net.eval()
    global_au_feat.eval()
    au_net.eval()

    return {
        'region_learning': region_learning,
        'align_net': align_net,
        'local_attention_refine': local_attention_refine,
        'local_au_net': local_au_net,
        'global_au_feat': global_au_feat,
        'au_net': au_net,
    }

###############################################################################
# Forward pass: region_learning -> align_net -> local_attention_refine -> ...
###############################################################################
@torch.no_grad()
def forward_jaanet_subnetworks(tensor_img, nets):
    """
    Args:
      tensor_img: shape(1,3,176,176)
      nets: dict of sub-networks
    Return:
      au_scores: shape(1, AU_num)
    """
    # region_feat: e.g. (1, unit_dim*8, H, W)
    region_feat = nets['region_learning'](tensor_img)
    
    # align_feat, align_output, aus_map
    align_feat, align_output, aus_map = nets['align_net'](region_feat)

    output_aus_map = nets['local_attention_refine'](aus_map)

    local_au_out_feat, local_aus_output = nets['local_au_net'](region_feat, output_aus_map)

    global_au_out_feat = nets['global_au_feat'](region_feat)

    # concat_au_feat shape = [1, (unit_dim*8 + unit_dim*8 + unit_dim*8)]? => 24*8=192?
    concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat), 1)
    aus_output = nets['au_net'](concat_au_feat)

    return aus_output  # shape (1, au_num)

###############################################################################
# Main function
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    # from test_JAAv2.py
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--retina_weights', type=str, default='Pytorch_Retinaface/weights/mobilenet0.25_Final.pth')
    parser.add_argument('--retina_network', type=str, default='mobile0.25')
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--nms_threshold', type=float, default=0.4)

    # JAANet + face_alignment + etc
    parser.add_argument('--crop_size', type=int, default=176)
    parser.add_argument('--map_size', type=int, default=44)
    parser.add_argument('--au_num', type=int, default=12)
    parser.add_argument('--land_num', type=int, default=49)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--start_epoch', type=int, default=5, help='which epoch to load')
    parser.add_argument('--n_epochs', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--run_name', type=str, default='JAAv2')
    parser.add_argument('--write_path_prefix', type=str, default='data/snapshots/')
    parser.add_argument('--write_res_prefix', type=str, default='data/res/')
    parser.add_argument('--region_learning', type=str, default='HMRegionLearning')
    parser.add_argument('--align_net', type=str, default='AlignNet')
    parser.add_argument('--local_attention_refine', type=str, default='LocalAttentionRefine')
    parser.add_argument('--local_au_net', type=str, default='LocalAUNetv2')
    parser.add_argument('--global_au_feat', type=str, default='HLFeatExtractor')
    parser.add_argument('--au_net', type=str, default='AUNet')
    parser.add_argument('--unit_dim', type=int, default=8, help='unit dims')
    # Additional for this script
    parser.add_argument('--input_video', type=str, default='my_Docker_app/IMG_7721.mp4', help='path to video or 0 for camera')
    parser.add_argument('--csv_out', type=str, default='au_result.csv', help='CSV output')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Load PyTorch RetinaFace
    retina_net, cfg_retina = load_retinaface_model(
        trained_model_path=args.retina_weights,
        network=args.retina_network,
        cpu=(device.type=='cpu')
    )

    # 2) Load JAANet
    nets = load_jaanet_subnetworks(args, device)
    # after load_jaanet_subnetworks
    for net_name, mod in nets.items():
        for param_name, param in mod.named_parameters():
            print(f"{net_name}.{param_name} -> {param.device}")

    # 3) face-alignment
    import face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device.type)

    # 4) open CSV
    csv_file = open(args.csv_out, "w")

    # au_header を neg/pos 含めて拡張:
    au_header = []
    for i in range(args.au_num):
        au_header.append(f"AU{i+1}-neg")
        au_header.append(f"AU{i+1}-pos")

    csv_file.write("frame," + ",".join(au_header) + "\n")

    # 5) open video
    if args.input_video == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.input_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 1

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 6) Detect faces w/ PyTorch RetinaFace
        faces = detect_faces_pytorch_retinaface(
            net=retina_net,
            cfg=cfg_retina,
            frame_bgr=frame_bgr,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold,
            use_cpu=(device.type=='cpu')
        )

        if faces and len(faces)>0:
            # pick first face
            face_info = faces[0]
            [x1,y1,x2,y2] = face_info['facial_area']

            # 7) Face-alignment => 68 landmarks
            all_landmarks = fa.get_landmarks(frame_rgb)
            if all_landmarks and len(all_landmarks)>0:
                land68 = all_landmarks[0]  # pick first face
                # 8) Preprocess => (1,3,176,176)
                input_tensor, land49 = preprocess_for_jaanet(
                    frame_rgb, land68,
                    device=device
                )
                
                # 9) JAANet Forward
                aus_output = forward_jaanet_subnetworks(input_tensor, nets)
                # aus_output shape: (batch=1, 2, au_num) => (2, au_num)
                #   where 0-axis=neg,1-axis=pos

                au_scores = aus_output[0].cpu().numpy()  # shape => (2, au_num)
                #print("au_scores shape:", au_scores.shape)
                print("frame index", frame_idx)

                # 10) CSV
                # au_scores[0,i] = neg, au_scores[1,i] = pos
                row_data = []
                for i in range(args.au_num):
                    neg_val = au_scores[0, i]  # shape (1,2,au_num) => (2, au_num)
                    pos_val = au_scores[1, i]
                    row_data.extend([neg_val, pos_val])

                # row_data 長さ= 2*au_num
                csv_line = ",".join(f"{v:.4f}" for v in row_data)
                csv_file.write(f"{frame_idx},{csv_line}\n")

        frame_idx += 1

    cap.release()
    csv_file.close()
    print("Done. Saved CSV to", args.csv_out)


if __name__ == "__main__":
    main()