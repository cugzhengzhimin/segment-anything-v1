import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import random
def percent_maxmin_aug(correctImage, img_max=255):
    min = np.min(correctImage)
    max = np.max(correctImage)
    if img_max != 255:
        max = max
    else:
        max = img_max
    low_val = random.randint(1, 5) / 1000
    high_val = random.randint(1, 5) / 1000
    if np.max(correctImage) == np.min(correctImage):
        min_hist = -9999
        max_hist = 9999
    else:
        hist, bins = np.histogram(correctImage.ravel(), max, [0, max])
        min_hist = -9999
        if min == 0:
            hist[0] = 0
        for i in range(len(hist)):
            if np.sum(hist[:i]) / np.sum(hist) > low_val:
                min_hist = bins[i]
                break
        max_hist = 9999
        for j in range(len(hist), 1, -1):
            if np.sum(hist[j:]) / np.sum(hist) > high_val:
                max_hist = bins[j]
                break
    correctImage = np.where(correctImage > max_hist, max_hist, correctImage)
    correctImage = np.where((correctImage < min_hist) & (correctImage != 0), min_hist, correctImage)
    if np.max(correctImage) == np.min(correctImage):
        correctImage = correctImage
    else:
        correctImage = max * ((correctImage - np.min(correctImage)) / (np.max(correctImage) - np.min(correctImage)))
    if max == 255:
        correctImage = correctImage.astype(np.uint8)
    else:
        correctImage = correctImage.astype(np.uint16)
    return 255 * correctImage


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def CalContoursNum(masks):
    num_contours = []
    for j in range(masks.shape[0]):
        contours, _ = cv2.findContours(masks[j].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        num_contours.append(len(contours))
    # index = np.argmin(np.array(num_contours))
    return num_contours


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

label = cv2.imread(r'/home/zhengzhimin/segment-anything-main/notebooks/images/shandong_80_y1.tif', 8)
# label = (label>0.1).astype(np.uint8)

# x_arange = np.arange(0, label.shape[1], 50)
# y_arange = np.arange(0, label.shape[0], 50)
# xv, yv = np.meshgrid(x_arange, y_arange)
# xv = xv.reshape(-1)
# yv = yv.reshape(-1)
# point_list = np.concatenate([xv[..., None], yv[..., None]], axis=1)


contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
point_list = []
contours_filt = []
for c in contours:
    m = cv2.moments(c)
    cx = int(m['m10'] / (m['m00'] + 1e-6))
    cy = int(m['m01'] / (m['m00'] + 1e-6))
    if label[cy, cx] == 0:
        continue
    point_list.append([cx, cy])
    contours_filt.append(c)

image = cv2.imread('/home/zhengzhimin/segment-anything-main/notebooks/images/shandong_80.tif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (label.shape[0], label.shape[1]))

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

import time
s = time.time()
predictor = SamPredictor(sam)
predictor.set_image(image)
print(time.time()-s)

# input_point = np.array([[500, 375]])
input_point = np.array(point_list)
input_label = np.array([1])
# input_label = np.zeros((len(point_list), ))
result = np.zeros_like(label).astype(np.uint16)
mask_all = []
score_all = []
for i in range(0, len(point_list)):
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point[i:i+1],
        point_labels=input_label,
        multimask_output=True,
    )

    # mask_input = logits[np.argmax(scores), :, :]
    # input_label = np.zeros((len(point_list),))
    # input_label[i] = 1
    # masks, scores, logits = predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     mask_input=mask_input[None, :, :],
    #     multimask_output=False,
    # )
    if i + 1 == 39:
        print('')
    num_contours = CalContoursNum(masks)
    index = np.argmin(np.array(num_contours))

    point_coords = input_point[i:i + 1]
    point_labels = input_label

    # 负点的迭代添加
    point_mask = np.zeros_like(masks[0])
    point_mask[input_point[:, 1], input_point[:, 0]] = 1
    thr = (point_mask * masks[index]).sum() / point_mask.sum()
    n = 0
    while thr > 0.5:
        temp_neg = np.concatenate([input_point[:i], input_point[i+1:]])
        random_index = np.random.randint(0, temp_neg.shape[0])
        point_coords = np.concatenate([point_coords, input_point[random_index:random_index+1]], axis=0)
        point_labels = np.concatenate([point_labels, np.array([0])], axis=0)
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=logits[index, :, :][None, :, :],
            multimask_output=True,
        )
        num_contours = CalContoursNum(masks)
        index = np.argmin(np.array(num_contours))
        thr = (point_mask * masks[index]).sum() / point_mask.sum()
        n += 1
    if (point_mask * masks[index]).sum() / point_mask.sum() > 0.5:
        print('')

    num_min_contours = num_contours[index]

    # # 正点的迭代添加
    # point_coords = input_point[i:i+1]
    # point_labels = input_label
    # n=0
    # while num_min_contours > 3 and n < 5:
    #     temp_mask = np.zeros_like(masks[index]).astype(np.uint8)
    #     cv2.drawContours(temp_mask, [contours_filt[i]], -1, 1, -1)
    #     points = np.array(np.where(temp_mask == 1)).T[:, [1, 0]]
    #     random_index = np.random.randint(0, points.shape[0])
    #     point_coords = np.concatenate([point_coords, points[random_index:random_index + 1]], axis=0)
    #     point_labels = np.concatenate([point_labels, input_label], axis=0)
    #     masks, scores, logits = predictor.predict(
    #         point_coords=point_coords,
    #         point_labels=point_labels,
    #         mask_input=logits[index, :, :][None, :, :],
    #         multimask_output=True,
    #     )
    #     num_contours = []
    #     for j in range(masks.shape[0]):
    #         contours, _ = cv2.findContours(masks[j].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #         num_contours.append(len(contours))
    #     index = np.argmin(np.array(num_contours))
    #     num_min_contours = num_contours[index]
    #     n += 1
    if num_min_contours > 3:
        continue



    # index = np.argmax(scores)
    # logits_max = logits[index, :, :]
    # po_mean = np.mean(1 / (1 + np.exp(-logits_max[logits_max > 0])))
    # if po_mean < 0.9:
    #     continue

    # po_mean = np.array([np.mean(1 / (1 + np.exp(-logits[i][logits[i] > 0]))) for i in range(scores.shape[0])])
    # cost = scores * po_mean
    # index = np.argmax(cost)
    # if cost[index] < 0.85 and po_mean[index] < 0.9:
    #     continue

    # contours1, _ = cv2.findContours(masks[index].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # n = 0
    # point_coords = input_point[i:i+1]
    # point_labels = input_label
    # while len(contours1) > 3 and n < 5:
    #     temp_mask = np.zeros_like(masks[index]).astype(np.uint8)
    #     cv2.drawContours(temp_mask, [contours_filt[i]], -1, 1, -1)
    #     points = np.array(np.where(temp_mask == 1)).T[:, [1, 0]]
    #     random_index = np.random.randint(0, points.shape[0])
    #     point_coords = np.concatenate([point_coords, points[random_index:random_index+1]], axis=0)
    #     point_labels = np.concatenate([point_labels, input_label], axis=0)
    #     masks, scores, logits = predictor.predict(
    #         point_coords=point_coords,
    #         point_labels=point_labels,
    #         mask_input=logits[index, :, :][None, :, :],
    #         multimask_output=True,
    #     )
    #     index = np.argmax(scores)
    #     logits_max = logits[index, :, :]
    #     po_mean = np.mean(1 / (1 + np.exp(-logits_max[logits_max > 0])))
    #     if po_mean < 0.9:
    #         continue
    #     contours1, _ = cv2.findContours(masks[index].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #     n += 1

    # po_mean = np.array([np.mean(1 / (1 + np.exp(-logits[i][logits[i] > 0]))) for i in range(logits.shape[0])])
    # area = np.array([masks[i].sum() for i in range(logits.shape[0])])
    # index_mean = np.where(po_mean>0.9)
    # if len(po_mean[po_mean > 0.9]) == 0:
    #     continue
    # print(po_mean > 0.9)
    # index = index_mean[0][np.argmin(area[po_mean > 0.9])]


    # if masks[index].sum() / (1280*1280) > 0.05:
    #     continue
    # print(scores[index], masks[index].sum() / (1280*1280))

    if ((result > 0) * masks[index]).sum() / (masks[index].sum()) > 0.9:
        continue
    temp = (1 - (result > 0)) * masks[index]
    result[temp > 0] = i+1


cv2.imwrite(r'./1.tif', result.astype(np.uint16))