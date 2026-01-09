# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.
import numpy as np
import cv2

from ppocrdet import PPOCRDet

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
]



def visualize(image, pts, thickness=2):
    output = image.copy()

    for rect in pts:
        if(len(rect) ==2):
          output = cv2.rectangle(output, rect[0], rect[1], (0, 0, 255), thickness)
        if(len(rect) == 4):
          output = cv2.polylines(output, pts, isClosed=True, color=(0, 255, 0), thickness=thickness)
    return output

backend_id = backend_target_pairs[0][0]
target_id = backend_target_pairs[0][1]

    # Instantiate model
model = PPOCRDet(modelPath="./model/text_detection_en_ppocrv3_2023may.onnx",
               inputSize=[736, 736],
               binaryThreshold=0.3,
               polygonThreshold=0.3,
               maxCandidates=20,
               unclipRatio=2.0,
               backendId=backend_id,
               targetId=target_id)

def letterbox(img, target_size=(640, 640), color=(116, 116, 116)):
    """
    对图像进行 letterbox 处理，保持宽高比
    返回：letterboxed 图像, 缩放比例, (dx, dy)
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size

    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放图像
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建画布
    padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    
    # 居中放置
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    padded[dy:dy+new_h, dx:dx+new_w] = resized

    return padded, scale

def find_top_right_and_bottom_right_rects(polys, min_width=20):
    """
    查找：
      - 右上：先取 y_min 最小的多边形（最靠上），若有多个，取其中 x_max 最大的（最靠右）
      - 右下：x_max + y_max 最大（保持不变）
    然后分别用宽度 >= min_width 过滤。
    """
    if len(polys) == 0:
        return None, None

    x_min = np.min(polys[..., 0], axis=1)
    y_min = np.min(polys[..., 1], axis=1)
    x_max = np.max(polys[..., 0], axis=1)
    y_max = np.max(polys[..., 1], axis=1)
    widths = x_max - x_min

    # ====== 右上：先按 y_min 升序，再按 x_max 降序 ======
    min_y = np.min(y_min)
    top_candidates = (y_min == min_y)  # 布尔掩码
    if np.sum(top_candidates) == 1:
        tr_idx = np.where(top_candidates)[0][0]
    else:
        # 多个最上 → 从中选 x_max 最大的
        candidate_xmax = x_max[top_candidates]
        best_local_idx = np.argmax(candidate_xmax)
        tr_idx = np.where(top_candidates)[0][best_local_idx]

    # ====== 右下：x_max + y_max 最大（不变）======
    br_idx = np.argmax(x_max + y_max)

    def make_rect(i):
        return ((int(x_min[i]), int(y_min[i])), (int(x_max[i]), int(y_max[i])))

    tr_rect = make_rect(tr_idx) if widths[tr_idx] >= min_width else None
    br_rect = make_rect(br_idx) if widths[br_idx] >= min_width else None

    return tr_rect, br_rect

def text_dect(input=None, visi=False):
    img = cv2.imread(input) if isinstance(input, str) else input
       
    paddle_img, scale = letterbox(img, target_size=(736, 736))

    # Inference
    results = model.infer(paddle_img)

    rlen = len(results[0])
    print(f'{rlen} texts detected.')
    tr_rect, br_rect = find_top_right_and_bottom_right_rects(np.array(results[0]),min_width=40)
    if tr_rect is None:
        tr_rect = ([130, 195], [300, 235])
    print(f"top right {tr_rect}")
    print(f"bottom right {br_rect}")
    if(visi):
      paddle_img = visualize(paddle_img, [tr_rect, br_rect])
      # paddle_img = visualize(paddle_img, results[0])
      # cv2.imshow('img', paddle_img)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
    # Save results if save is true
    return paddle_img, tr_rect, br_rect
