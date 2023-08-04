import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything.matrix_nms import mask_matrix_nms
from sobel import Sobel
device = "cuda"

def model_load(image, weight_path="sam_vit_h_4b8939.pth", model_type="vit_h"):
    sam = sam_model_registry[model_type](checkpoint=weight_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    return predictor


def CalContoursNum(masks):
    num_contours = []
    area = []
    _, h, w = masks.shape
    for j in range(masks.shape[0]):
        contours, _ = cv2.findContours(masks[j].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        num_contours.append(len(contours))
        area.append(masks[j].sum() / (h*w))
    # index = np.argmin(np.array(num_contours))
    return num_contours, area


def CalContoursNumJIT(masks):
    # result = 0
    # for i in range(masks.shape[0]):
    #     for j in range(masks.shape[1]):
    #         result += masks[i, j]
    all_num_contours = []
    for i in range(masks.shape[0]):
        mask = masks[i]
        num_contours = []
        for j in range(mask.shape[0]):
            contours, _ = cv2.findContours(mask[j].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_contours.append(len(contours))
        all_num_contours.append(num_contours)
    return all_num_contours


def predict_v1(predictor, label):
    contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    point_list = []
    for c in contours:
        m = cv2.moments(c)
        cx = int(m['m10'] / (m['m00'] + 1e-6))
        cy = int(m['m01'] / (m['m00'] + 1e-6))
        if label[cy, cx] == 0:
            continue
        point_list.append([cx, cy])

    mask_all = []
    score_all = []
    input_point = np.array(point_list)
    input_label = np.array([1])
    result = np.zeros_like(label)
    for i in range(0, len(point_list)):
        masks, scores, logits = predictor.predict(
            point_coords=input_point[i:i + 1],
            point_labels=input_label,
            multimask_output=True,
        )
        index = torch.argmax(scores)
        logits_max = logits[index, :, :]
        pred = logits_max.sigmoid()
        po_mean = pred[pred > 0.5].mean()
        if po_mean < 0.9:
            continue
        mask_all.append(masks[index:index+1])
        score_all.append(scores[index:index+1])
    mask_all = torch.cat(mask_all, dim=0)
    score_all = torch.cat(score_all, dim=0)
    label_all = torch.ones_like(score_all)
    scores, labels, masks, keep_inds = mask_matrix_nms(mask_all, label_all, score_all, filter_thr=0.2)
    masks = masks.cpu().data.numpy()
    for j in range(masks.shape[0]):
        result[masks[j]>0] = j+1
    cv2.imwrite(r'./1.tif', result.astype(np.uint8))
    return result

def predict_v2(predictor, label):
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

    mask_all = []
    score_all = []
    input_point = np.array(point_list)
    result = np.zeros_like(label)
    for i in range(0, len(point_list)):
        # 初步预测
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point[i:i + 1],
            point_labels=input_label,
            multimask_output=True,
        )
        num_contours = CalContoursNum(masks)
        index = np.argmin(np.array(num_contours))

        # 如果单此迭代的点比较好，则跳过负点迭代
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
        if num_min_contours > 3:
            continue
        if ((result > 0) * masks[index]).sum() / (masks[index].sum()) > 0.9:
            continue
        temp = (1 - (result > 0)) * masks[index]
        result[temp > 0] = i + 1
    return result

def predict_everything(predictor):
    model_size = predictor.model.image_encoder.img_size
    x_arange = np.arange(0, model_size, 50)
    y_arange = np.arange(0, model_size, 50)
    xv, yv = np.meshgrid(x_arange, y_arange)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    point_list = np.concatenate([xv[..., None], yv[..., None]], axis=1)

    input_point = np.array(point_list)
    result = np.zeros((predictor.original_size[0], predictor.original_size[1])).astype(np.uint16)
    for i in range(0, len(point_list)):
        # 初步预测
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point[i:i + 1],
            point_labels=input_label,
            multimask_output=True,
        )

        num_contours = CalContoursNum(masks)
        index = np.argmin(np.array(num_contours))
        num_min_contours = num_contours[index]
        if num_min_contours > 3:
            continue
        # if masks[index].sum() / (1280*1280) > 0.1:
        #     continue

        if ((result > 0) * masks[index]).sum() / (masks[index].sum()) > 0.9:
            continue
        temp = (1 - (result > 0)) * masks[index]
        # temp = masks[index]
        result[temp > 0] = i + 1
    return result


import multiprocessing as mp
def predict_fast_everything(predictor):
    model_size = predictor.model.image_encoder.img_size
    x_arange = np.arange(0, model_size, 50)
    y_arange = np.arange(0, model_size, 50)
    xv, yv = np.meshgrid(x_arange, y_arange)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    point_list = np.concatenate([xv[..., None], yv[..., None]], axis=1)

    input_point = np.array(point_list)
    # result = np.zeros((predictor.original_size[0], predictor.original_size[1])).astype(np.uint16)
    result = torch.zeros((predictor.original_size[0], predictor.original_size[1]), dtype=torch.int16).cuda()


    # 加四角负点
    # fu_point = np.array([[[model_size, 0], [0, model_size], [model_size, model_size]]])
    # fu_point = np.repeat(fu_point, int(input_point.shape[0]), axis=0)
    # point_coords = np.concatenate([input_point[:, None, :], fu_point], axis=1)

    # 不加四角负点
    point_coords = input_point[:, None, :]

    # point_labels = np.concatenate([np.ones((input_point.shape[0], 1)), np.zeros((input_point.shape[0], 3))], axis=-1)
    point_labels = np.ones((input_point.shape[0], 1))
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    # masks, scores, logits = predictor.predict(
    #     point_coords=np.array([[382, 368]]),
    #     point_labels=np.array([1]),
    #     multimask_output=True,
    # )

    low_masks = logits > 0
    low_masks_cpu = low_masks.cpu().data.numpy()
    # masks = masks.cpu().data.numpy()
    # masks_cuda = torch.from_numpy(masks.astype(np.uint8)).cuda()

    keep = []
    for i in range(masks.shape[0]):
        num_contours, area = CalContoursNum(low_masks_cpu[i])

        index = np.argmin(np.array(num_contours))

        num_min_contours = num_contours[index]
        if num_min_contours > 3:
            continue

        if masks[i][index].sum() / (masks[i][index].shape[0]*masks[i][index].shape[1]) > 0.6:
            continue

        if ((result > 0) * masks[i][index]).sum() / (masks[i][index].sum()) > 0.9:
            continue

        keep.append([i, index, area[index]])
        # temp = (1 - (result > 0)) * masks[i][index]
        # result[temp > 0] = i + 1

    import time
    s = time.time()

    keep = np.array(keep)
    # keep = torch.tensor(keep).cuda()
    keep_mask = low_masks_cpu[keep[:, 0].astype(np.uint16), keep[:, 1].astype(np.uint16), ...]
    # keep_mask = low_masks[keep[:, 0].long(), keep[:, 1].long(), ...]
    keep_mask = torch.from_numpy(keep_mask).cuda()
    label_all = torch.ones(keep_mask.shape[0]).cuda()
    _, _, _, keep_inds = mask_matrix_nms(keep_mask, label_all, label_all, filter_thr=0.2)
    keep_inds = keep_inds.cpu().data.numpy()
    keep_end = keep[keep_inds, ...]
    max_to_min = np.argsort(keep[keep_inds, -1])[::-1]
    # _, max_to_min = torch.sort(keep[keep_inds, -1], descending=True)

    keep_end = keep_end[max_to_min, :]
    for k in range(keep_end.shape[0]):
        i, index, _ = keep_end[k]
        # result[masks[int(i.item())][int(index.item())] > 0] = i.item() + 1
        result[masks[int(i)][int(index)] > 0] = i + 1

    print(time.time()-s)
    return result


def predict_fast_gpu_everything(predictor, divice="cuda:0"):
    sobel = Sobel().to(divice)
    model_size = predictor.model.image_encoder.img_size
    x_arange = np.arange(0, model_size, 50)
    y_arange = np.arange(0, model_size, 50)
    xv, yv = np.meshgrid(x_arange, y_arange)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    point_list = np.concatenate([xv[..., None], yv[..., None]], axis=1)

    input_point = np.array(point_list)

    result = torch.zeros((predictor.original_size[0], predictor.original_size[1]), dtype=torch.int16).to(divice)


    # 加四角负点
    # fu_point = np.array([[[model_size, 0], [0, model_size], [model_size, model_size]]])
    # fu_point = np.repeat(fu_point, int(input_point.shape[0]), axis=0)
    # point_coords = np.concatenate([input_point[:, None, :], fu_point], axis=1)

    # 不加四角负点
    point_coords = input_point[:, None, :]

    # point_labels = np.concatenate([np.ones((input_point.shape[0], 1)), np.zeros((input_point.shape[0], 3))], axis=-1)
    point_labels = np.ones((input_point.shape[0], 1))
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    logits_p = logits.sigmoid()
    low_masks = logits > 0
    low_masks_cpu = low_masks.cpu().data.numpy()

    keep = []
    for i in range(masks.shape[0]):
        num_contours, area = CalContoursNum(low_masks_cpu[i])

        index = np.argmin(np.array(num_contours))

        num_min_contours = num_contours[index]
        if not (num_min_contours < 3 or logits_p[i][index][logits_p[i][index] > 0.5].mean() > 0.96):
            continue

        # if masks[i][index].sum() / (masks[i][index].shape[0]*masks[i][index].shape[1]) > 0.6:
        #     continue

        if ((result > 0) * masks[i][index]).sum() / (masks[i][index].sum()) > 0.9:
            continue

        keep.append([i, index, area[index]])

    import time
    s = time.time()
    keep = torch.tensor(keep).to(divice)
    keep_mask = low_masks[keep[:, 0].long(), keep[:, 1].long(), ...]
    label_all = torch.ones(keep_mask.shape[0]).to(divice)
    _, _, _, keep_inds = mask_matrix_nms(keep_mask, label_all, label_all, filter_thr=0.3)
    keep_end = keep[keep_inds, ...]
    _, max_to_min = torch.sort(keep[keep_inds, -1], descending=True)
    keep_end = keep_end[max_to_min, :]
    for k in range(keep_end.shape[0]):
        i, index, _ = keep_end[k]
        result[masks[int(i.item())][int(index.item())] > 0] = i.item() + 1
    g = sobel(result[None, None, ...].float())
    g_mask = torch.sum(g[0].abs(), dim=0) == 0
    result = (result > 0) * g_mask
    print(time.time()-s)
    return result.cpu().data.numpy()


def main(image_path, lable_path, weight_path="sam_vit_h_4b8939.pth", model_type="vit_h", task_type=0):
    img = cv2.imread(image_path)[..., [2, 1, 0]]
    lab = cv2.imread(lable_path, 0)
    img = cv2.resize(img, (1024, 1024))
    predictor = model_load(img, weight_path=weight_path, model_type=model_type)
    if task_type == 0:
        print('auto sam')
        result = predict_v2(predictor, lab)
    elif task_type == 1:
        print('sam everything')
        result = predict_fast_gpu_everything(predictor)

    cv2.imwrite(r'./1.tif', 255 * result.astype(np.uint8))

if __name__ == '__main__':
    image_path = r''
    label_path = r''
    main(image_path, label_path, weight_path='./sam_vit_h_4b8939.pth', model_type='vit_h', task_type=1)