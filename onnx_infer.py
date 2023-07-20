import onnxruntime, onnx
import cv2
import numpy as np


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

import copy
def apply_coords(coords, original_size):
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(old_h, old_w, 1024)
    coords = copy.deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords

def preprocess(x, image_encoder_img_size=(1024, 1024)):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    target_size = get_preprocess_shape(int(x.shape[0]), int(x.shape[1]), 1024)
    x = cv2.resize(x, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    pixel_mean = [123.675, 116.28, 103.53],
    pixel_std = [58.395, 57.12, 57.375]
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[:2]
    padh = image_encoder_img_size[0] - h
    padw = image_encoder_img_size[1] - w
    x = np.pad(x, ((0, padh), (0, padw), (0, 0)))
    return x

image = cv2.imread('/home/zhengzhimin/segment-anything-main/notebooks/images/kashi-small2.tif')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (1024, 1024))
orig_im_size = image.shape[:2]
image = preprocess(image).transpose(2, 0, 1)[None, ...].astype(np.float32)

ort_inputs = {'image': image}
ort_session = onnxruntime.InferenceSession(r'/home/zhengzhimin/segment-anything-main/onnx_weight/vit_b.onnx',
                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
image_embedding = ort_session.run(None, ort_inputs)

input_point = np.array([[472, 506]])
input_label = np.array([1])
onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

onnx_coord = apply_coords(onnx_coord, orig_im_size).astype(np.float32)
onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)


# import torch
# from segment_anything import build_sam, build_sam_vit_b, build_sam_vit_l
# from segment_anything.utils.onnx import SamOnnxModel, EncoderModel
# ort_inputs = {
#     "image_embeddings": torch.from_numpy(image_embedding[0]),
#     "point_coords": torch.from_numpy(onnx_coord),
#     "point_labels": torch.from_numpy(onnx_label),
#     "mask_input": torch.from_numpy(onnx_mask_input),
#     "has_mask_input": torch.from_numpy(onnx_has_mask_input),
#     "orig_im_size": torch.from_numpy(np.array(orig_im_size, dtype=np.float32))
# }
#
# sam = build_sam(r'sam_vit_h_4b8939.pth')
# onnx_model = SamOnnxModel(
#         model=sam,
#         return_single_mask=False
#     )
# _ = onnx_model(**ort_inputs)

ort_inputs = {
    "image_embeddings": image_embedding[0],
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(orig_im_size, dtype=np.float32)
}

ort_session_interaction = onnxruntime.InferenceSession(r'/home/zhengzhimin/segment-anything-main/interaction_vit_b.onnx')

masks, scores, low_res_logits = ort_session_interaction.run(None, ort_inputs)
masks = masks > 0

