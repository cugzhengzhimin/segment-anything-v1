# Copyright (c) OpenMMLab. All rights reserved.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import cv2
import torch
import albumentations as A
import gdal, osr
import numpy as np
from segment_anything.matrix_nms import mask_matrix_nms
import json


class GwEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def all_path(dir_path):
    file_list = []
    for maindir, subdir, file_name_list in os.walk(dir_path):
        for filename in file_name_list:
            if os.path.splitext(filename)[1] in ['.tif', '.img']:
                apath = os.path.join(maindir, filename)
                file_list.append(apath)
    return file_list


def static_img(input_path):
    img_means = []
    img_devs = []
    img_raster = gdal.Open(input_path, gdal.GA_ReadOnly)
    channel_count = img_raster.RasterCount
    for channel in range(channel_count):
        band = img_raster.GetRasterBand(channel + 1)
        # print(band.GetStatistics( True, True ))
        stat = band.GetStatistics(True, True)
        img_means.append(stat[2])
        img_devs.append(stat[3])
    return np.array(img_means), np.array(img_devs)


def whole_dodgong(src_means, src_devs, des_means, des_devs, buffer):
    mask = (buffer > 0).astype('uint8')
    buffer = (buffer - src_means) * des_devs / src_devs + des_means
    buffer = buffer * mask
    buffer *= 255
    buffer = np.where(buffer > 0, buffer, 0)
    buffer = np.where(buffer < 255, buffer, 255)
    buffer = buffer.astype('uint8')
    return buffer


def to_norm(img, mean2):
    # guanmu的[127.97172403, 121.11262576, 105.96691204]
    # tantu的[120.52805287, 107.64715255,  81.06614085]
    mean1 = [np.mean(img[..., i]) for i in range(img.shape[-1])]
    # mean2 = [145, 133, 103]
    mean2 = [88.86, 86.22, 89.15]
    std1 = np.std(np.std(img, axis=0), axis=0)
    std2 = [28.77, 31.09, 24.46]
    image = (std2/std1) * img + mean2 - (std2/std1) * mean1
    # image = (mean2 / mean1) * img
    return image


def write(input_path1):
    img_raster1 = gdal.Open(input_path1, gdal.GA_ReadOnly)
    data = img_raster1.ReadAsArray(xoff=30000, yoff=33000, xsize=10000, ysize=10000)
    target = gdal.GetDriverByName('GTiff').Create(r'./1.tif', 10000, 10000, 3, gdal.GDT_Byte)
    target.WriteRaster(0, 0, 10000, 10000, data.tostring())


def clip_img(input_path1, input_path2, clip_size, clip_crop):
    img_raster1 = gdal.Open(input_path1, gdal.GA_ReadOnly)
    img_raster2 = gdal.Open(input_path2, gdal.GA_ReadOnly)
    ref_transform1 = img_raster1.GetGeoTransform()
    ref_transform2 = img_raster2.GetGeoTransform()
    im_chanel = img_raster1.RasterCount
    im_width = img_raster1.RasterXSize  # 栅格矩阵的列数
    im_height = img_raster1.RasterYSize  # 栅格矩阵的行数

    im_width2 = img_raster2.RasterXSize  # 栅格矩阵的列数
    im_height2 = img_raster2.RasterYSize  # 栅格矩阵的行数

    true_clip_size = clip_size - 1 * clip_crop
    if im_height % true_clip_size > clip_crop:
        row_num = int(im_height / true_clip_size) + 1
    else:
        row_num = int(im_height / true_clip_size)

    if im_width % true_clip_size > clip_crop:
        col_num = int(im_width / true_clip_size) + 1
    else:
        col_num = int(im_width / true_clip_size)

    for h in range(0, row_num):
        for w in range(0, col_num):
            # print(row_num, col_num, h, w)
            begin_x = w * true_clip_size
            begin_y = h * true_clip_size
            write_width = clip_size - clip_crop // 2
            write_height = clip_size - clip_crop // 2
            true_begin_x = begin_x + clip_crop // 2 if w != 0 else begin_x
            true_begin_y = begin_y + clip_crop // 2 if h != 0 else begin_y
            end_x = begin_x + clip_size - 1
            end_y = begin_y + clip_size - 1
            if end_x >= im_width - 1:
                end_x = im_width - 1
                write_width = end_x - true_begin_x
            if end_y >= im_height - 1:
                end_y = im_height - 1
                write_height = end_y - true_begin_y

            readwidth = end_x - begin_x + 1
            readheight = end_y - begin_y + 1

            try:
                img_buffer1 = img_raster1.ReadAsArray(xoff=begin_x, yoff=begin_y, xsize=readwidth, ysize=readheight)
                img_buffer2 = img_raster2.ReadAsArray(xoff=begin_x, yoff=begin_y, xsize=readwidth, ysize=readheight)
                if im_chanel == 1:
                    img_buffer1 = np.reshape(img_buffer1, [readheight, readwidth, 1]).repeat(3, axis=-1)
                    if readwidth != clip_size or readheight != clip_size:
                        img_buffer1_pad = np.zeros((clip_size, clip_size, img_buffer1.shape[-1]))
                        img_buffer1_pad[:img_buffer1.shape[0], :img_buffer1.shape[1], :] = img_buffer1
                        img_buffer1 = img_buffer1_pad

                else:
                    img_buffer1 = np.transpose(img_buffer1, [1, 2, 0])[..., [0, 1, 2]]
                    img_buffer2 = img_buffer2[..., None]
                    # if readwidth != clip_size or readheight != clip_size:
                    #     img_buffer1_pad = 0 * np.ones((clip_size, clip_size, 3))
                    #     img_buffer1_pad[:img_buffer1.shape[0], :img_buffer1.shape[1], :] = img_buffer1
                    #     img_buffer1 = img_buffer1_pad
                # img_buffer1 = percent_maxmin_norm(img_buffer1)
            except Exception as e:
                print(e)
                # print(begin_x, begin_y, readwidth, readheight)

            print(true_begin_x, true_begin_y, true_begin_x - begin_x, true_begin_y - begin_y, write_width, write_height,
                  end_x, end_y)
            index_xy = [w, h]
            yield img_buffer1, img_buffer2, true_begin_x, true_begin_y, true_begin_x - begin_x, true_begin_y - begin_y, write_width, write_height, index_xy


def percent(correctImage):
    min = np.min(correctImage)
    max = np.max(correctImage)
    if min == max:
        min_hist = -9999
        max_hist = 9999
        return correctImage
    else:
        hist, bins = np.histogram(correctImage.ravel(), max, [min, max])
        min_hist = -9999
        for i in range(len(hist)):
            if np.sum(hist[:i]) / np.sum(hist) > 0.001:
                min_hist = bins[i]
                break
        max_hist = 9999
        for j in range(len(hist), 1, -1):
            if np.sum(hist[j:]) / np.sum(hist) > 0.001:
                max_hist = bins[j]
                break
    correctImage = np.where(correctImage > max_hist, max_hist, correctImage)
    correctImage = np.where(correctImage < min_hist, min_hist, correctImage)
    correctImage = 1 * ((correctImage - np.min(correctImage)) / (
            np.max(correctImage) - np.min(correctImage)))
    return correctImage


def percent_maxmin_norm(Image):
    _, _, c = Image.shape
    new_image = []
    for i in range(c):
        correctImage = Image[..., i]
        min = np.min(correctImage)+1
        max = np.max(correctImage)
        if min == max:
            min_hist = -9999
            max_hist = 9999
        else:
            hist, bins = np.histogram(correctImage.ravel(), int(max), [min, max])
            min_hist = -9999
            for i in range(len(hist)):
                if np.sum(hist[:i]) / np.sum(hist) > 0.0001:
                    min_hist = bins[i]
                    break
            max_hist = 9999
            for j in range(len(hist), 1, -1):
                if np.sum(hist[j:]) / np.sum(hist) > 0.0001:
                    max_hist = bins[j]
                    break
        correctImage = np.where(correctImage > max_hist, max_hist, correctImage)
        correctImage = np.where(correctImage < min_hist, min_hist, correctImage)
        if np.max(correctImage) == np.min(correctImage):
            correctImage = correctImage / 255
        else:
            correctImage = 255 * ((correctImage - np.min(correctImage)) / (np.max(correctImage) - np.min(correctImage)))
        correctImage = correctImage[..., None]
        new_image.append(correctImage)
    new_image = np.concatenate(new_image, axis=-1)
    return new_image

def maxmin_norm(correctImage):
    # if np.max(correctImage) == np.min(correctImage):
    #     correctImage = correctImage / 255
    # else:
    #     correctImage = 1 * ((correctImage - np.min(correctImage)) / (np.max(correctImage) - np.min(correctImage)))
    correctImage = 1 * ((correctImage - np.min(correctImage)) / (np.max(correctImage) - np.min(correctImage) + 1e-5))
    return correctImage

def gwnorm(img, norm, mean):
    if mean:
        img = to_norm(img, mean)
        # img = 255 * percent_maxmin_norm(img)

    if norm == 'max_min':
        img = maxmin_norm(img)
    elif norm == 'percent_max_min':
        img = percent_maxmin_norm(img)

    elif norm == 'div_255':
        img = img / 255
    else:
        img = img
    # img = torch.from_numpy(np.moveaxis(img, -1, 0).astype(np.float32))
    return img

from segment_anything import sam_model_registry, SamPredictor
def model_load(weight_path="sam_vit_h_4b8939.pth", model_type="vit_h", device='cuda:0'):
    sam = sam_model_registry[model_type](checkpoint=weight_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    # predictor.set_image(image)
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


def predict_v2(predictor, label):
    contours, _ = cv2.findContours(label.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        masks = masks.cpu().data.numpy()
        logits = logits.cpu().data.numpy()
        num_contours, _ = CalContoursNum(masks[0])
        index = np.argmin(np.array(num_contours))

        # 如果单此迭代的点比较好，则跳过负点迭代
        point_coords = input_point[i:i + 1]
        point_labels = input_label
        # 负点的迭代添加
        point_mask = np.zeros_like(masks[0][0])
        point_mask[input_point[:, 1], input_point[:, 0]] = 1
        thr = (point_mask * masks[0][index]).sum() / point_mask.sum()
        n = 0
        while thr > 0.2 or num_contours[index] > 3:
            temp_neg = np.concatenate([input_point[:i], input_point[i+1:]])
            random_index = np.random.randint(0, temp_neg.shape[0])
            point_coords = np.concatenate([point_coords, input_point[random_index:random_index+1]], axis=0)
            point_labels = np.concatenate([point_labels, np.array([0])], axis=0)
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=logits[0][index, :, :][None, :, :],
                multimask_output=True,
            )
            masks = masks.cpu().data.numpy()
            logits = logits.cpu().data.numpy()
            num_contours, _ = CalContoursNum(masks[0])
            index = np.argmin(np.array(num_contours))
            thr = (point_mask * masks[0][index]).sum() / point_mask.sum()
            n += 1
        if (point_mask * masks[0][index]).sum() / point_mask.sum() > 0.5:
            print('')

        num_min_contours = num_contours[index]
        if num_min_contours > 3:
            continue
        if ((result > 0) * masks[0][index]).sum() / (masks[0][index].sum()) > 0.9:
            continue
        temp = (1 - (result > 0)) * masks[0][index]
        result[temp > 0] = i + 1
    return result


def random_sample(labels_gpu, num_labels):
    """
    采样正点和负点
    """
    p_point_list = []
    p_label_list = []
    n_point_list = []
    n_label_list = []
    # labels_gpu = torch.from_numpy(labels).cuda()
    for i in range(1, num_labels):
        all_point = torch.stack(torch.where(labels_gpu == i), dim=1)[:, [1, 0]]
        random_point = torch.randint(0, all_point.shape[0], (4,))
        p_point = all_point[random_point]
        p_label = torch.ones((len(p_point),))
        if num_labels==2:
            neg_point = torch.stack(torch.where(labels_gpu==0), dim=1)[:, [1, 0]]
        else:
            neg_point = torch.stack(torch.where((labels_gpu != i) & (labels_gpu != 0)), dim=1)[:, [1, 0]]
        random_neg_point = torch.randint(0, neg_point.shape[0], (100,))
        n_point = neg_point[random_neg_point]
        n_label = torch.zeros((len(n_point),))

        p_point_list.append(p_point)
        p_label_list.append(p_label)
        n_point_list.append(n_point)
        n_label_list.append(n_label)
    p_point_coords = torch.stack(p_point_list, dim=0).cpu().data.numpy()
    p_point_labels = torch.stack(p_label_list, dim=0).cpu().data.numpy()
    n_point_coords = torch.stack(n_point_list, dim=0).cpu().data.numpy()
    n_point_labels = torch.stack(n_label_list, dim=0).cpu().data.numpy()
    return p_point_coords, p_point_labels, n_point_coords, n_point_labels


def iter_interaction_infer(predictor,
                           p_point_coords,
                           p_point_labels,
                           n_point_coords,
                           n_point_labels,
                           all_mask,
                           all_area,
                           labels_gpu):
    point_mask = torch.zeros((all_mask.shape[1], all_mask.shape[2])).to(all_mask.device)
    point_mask[p_point_coords[:, 0, 1], p_point_coords[:, 0, 0]] = 1

    # point_mask = labels_gpu

    masks, scores, logits = predictor.predict(
        point_coords=p_point_coords[:, :1, :],
        point_labels=p_point_labels[:, :1],
        multimask_output=True,
    )

    low_masks = logits > 0
    low_masks_cpu = low_masks.cpu().data.numpy()

    keep = []
    keep_logits = []
    for i in range(masks.shape[0]):
        num_contours, area = CalContoursNum(low_masks_cpu[i])

        index = np.argmin(np.array(num_contours))

        num_min_contours = num_contours[index]
        if num_min_contours > 3:
            keep.append(0)
            keep_logits.append(logits[i][index])
            continue

        intersect_point_num = (point_mask * masks[i][index]).sum()
        thr = intersect_point_num / point_mask.sum()
        if (thr > 0.2 and intersect_point_num > 1) or (labels_gpu[i].sum() / masks[i][index].sum()) < 10/(500*500):
            keep.append(0)
            keep_logits.append(logits[i][index])
            continue

        # thr = labels_gpu[i].sum() / masks[i][index].sum()
        # if thr < 0.2:
        #     keep.append(0)
        #     keep_logits.append(logits[i][index])
        #     continue

        keep.append(1)
        all_mask[i, ...] = masks[i][index]
        all_area[i] = area[index]

    keep = np.array(keep)
    # 取出分割得不好的点，进行二次分割，加负点进行迭代
    index1 = np.where(keep == 0)[0]
    n = 0
    while np.sum(1 - keep) > 0 and n < 10:
        point_coords = np.concatenate([p_point_coords[index1, :1, :], n_point_coords[index1, :n + 1, :]], axis=1)
        point_labels = np.concatenate([p_point_labels[index1, :1], n_point_labels[index1, :n + 1]], axis=1)
        mask_input = torch.stack(keep_logits, dim=0)[:, None, ...].cpu().data.numpy()
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=True,
        )
        low_masks = logits > 0
        low_masks_cpu = low_masks.cpu().data.numpy()

        keep_logits = []
        for i in range(masks.shape[0]):
            num_contours, area = CalContoursNum(low_masks_cpu[i])

            index = np.argmin(np.array(num_contours))

            num_min_contours = num_contours[index]
            if num_min_contours > 3:
                keep_logits.append(logits[i][index])
                continue

            intersect_point_num = (point_mask * masks[i][index]).sum()
            thr = intersect_point_num / point_mask.sum()
            if (thr > 0.2 and intersect_point_num > 1) or (labels_gpu[i].sum() / masks[i][index].sum()) < 10/(500*500):
                keep_logits.append(logits[i][index])
                continue

            # thr = labels_gpu[i].sum() / masks[i][index].sum()
            # if thr < 0.2:
            #     keep_logits.append(logits[i][index])
            #     continue
            keep[index1[i]] = 1
            all_mask[index1[i], ...] = masks[i][index]
            all_area[index1[i]] = area[index]


        index1 = np.where(keep == 0)[0]
        n += 1
    return all_mask, all_area

def iter_every_infer(predictor,
                   p_point_coords,
                   p_point_labels,
                   all_mask,
                   all_area,
                   labels_gpu,
                   thr):
    point_mask = torch.zeros((all_mask.shape[1], all_mask.shape[2])).to(all_mask.device)
    point_mask[p_point_coords[:, 0, 1], p_point_coords[:, 0, 0]] = 1

    masks, scores, logits = predictor.predict(
        point_coords=p_point_coords,
        point_labels=p_point_labels,
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
        if not (num_min_contours < 3 or logits_p[i][index][logits_p[i][index] > 0.5].mean() > thr):
            continue

        # if masks[i][index].sum() / (masks[i][index].shape[0]*masks[i][index].shape[1]) > 0.6:
        #     continue

        if ((torch.any(all_mask, dim=0) > 0) * masks[i][index]).sum() / (masks[i][index].sum()) > 0.9:
            continue
        all_mask[i, ...] = masks[i][index]

        keep.append([i, index, area[index]])
    return all_mask, all_area



def get_p_point(all_mask, labels_gpu):
    """
    有一部分没有提取到，再次获取新的正点
    """
    # 汇总所有预测到的mask
    sum_mask = torch.any(all_mask.type(torch.uint8), dim=0, keepdim=True)
    # 所有预测到的mask与给的标签求soft_iou
    soft_iou = torch.sum(labels_gpu * sum_mask, dim=(1, 2)) / torch.sum(labels_gpu, dim=(1, 2))
    filt = soft_iou < 0.8
    label_new = labels_gpu[filt]
    label_new = label_new * (1 - sum_mask)
    label_new = label_new * torch.arange(1, label_new.shape[0] + 1).view(-1, 1, 1).to(label_new.device)
    label_new = torch.sum(label_new, dim=0)
    return label_new, filt


import torch.nn.functional as F
from sobel import Sobel
def predict_fast_gpu_ssam_v1(predictor, label, device="cuda:0"):
    """
    适合解析调整边界的问题，对于输入的label要求单个对象尽可能完整
    """
    sobel = Sobel().to(divice)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label.astype(np.uint8), connectivity=8)
    p_point_coords, p_point_labels, n_point_coords, n_point_labels = random_sample(torch.from_numpy(labels).cuda(), num_labels)

    labels_gpu = torch.from_numpy(labels).to(device)
    labels_gpu = F.one_hot(labels_gpu.long(), num_classes=num_labels).permute(2, 0, 1)[1:, ...].type(torch.uint8)
    result = torch.zeros((predictor.original_size[0], predictor.original_size[1]), dtype=torch.int16).to(device)

    all_mask = torch.zeros((p_point_coords.shape[0], predictor.original_size[0], predictor.original_size[1]), dtype=torch.uint8).to(device)
    all_area = torch.zeros((p_point_coords.shape[0], )).to(device)

    all_mask, all_area = iter_interaction_infer(predictor,
                                               p_point_coords,
                                               p_point_labels,
                                               n_point_coords,
                                               n_point_labels,
                                               all_mask,
                                               all_area,
                                               labels_gpu)

    # 获取正点,进行查漏迭代
    label_new, filt = get_p_point(all_mask, labels_gpu)
    n=0
    while filt.sum() > 1 and n < 5:
        p_point_coords, p_point_labels, n_point_coords, n_point_labels = random_sample(label_new,
                                                                                       filt.sum() + 1)
        label_new = F.one_hot(label_new.long(), num_classes=filt.sum() + 1).permute(2, 0, 1)[1:, ...]
        add_mask = torch.zeros((p_point_coords.shape[0], predictor.original_size[0], predictor.original_size[1]), dtype=torch.uint8).to(device)
        add_area = torch.zeros((p_point_coords.shape[0],)).to(device)
        add_mask, add_area = iter_interaction_infer(predictor,
                                                     p_point_coords,
                                                     p_point_labels,
                                                     n_point_coords,
                                                     n_point_labels,
                                                     add_mask,
                                                     add_area,
                                                     label_new)
        all_mask = torch.cat([all_mask, add_mask])
        all_area = torch.cat([all_area, add_area])
        label_new, filt = get_p_point(all_mask, labels_gpu)
        n+=1



    keep_mask = all_mask
    label_all = torch.ones(keep_mask.shape[0]).to(device)
    _, _, _, keep_inds = mask_matrix_nms(keep_mask, label_all, label_all, filter_thr=0.3)
    # 排序
    _, max_to_min = torch.sort(all_area[keep_inds], descending=True)
    keep_end = keep_inds[max_to_min]
    for k in range(keep_end.shape[0]):
        i = keep_end[k]
        result[keep_mask[i.item()] > 0] = i.item() + 1
    g = sobel(result[None, None, ...].float())
    g_mask = torch.sum(g[0].abs(), dim=0) == 0
    result = (result > 0) * g_mask
    return result.cpu().data.numpy()


# def predict_fast_gpu_ssam(predictor, label, divice="cuda:0"):
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label.astype(np.uint8), connectivity=8)
#     p_point_list = []
#     p_label_list = []
#     n_point_list = []
#     n_label_list = []
#
#     # for i in range(1, num_labels):
#     #     all_point = np.array(np.where(labels==i)).T[:, [1, 0]]
#     #     random_point = np.random.randint(0, len(all_point), 4)
#     #     p_point = all_point[random_point] #正点
#     #     p_label = np.ones((len(p_point), ))
#     #
#     #     neg_point = np.array(np.where((labels != i)&(labels != 0))).T[:, [1, 0]]
#     #     random_neg_point = np.random.randint(0, len(neg_point), 100)
#     #     n_point = neg_point[random_neg_point]  # 点
#     #     n_label = np.zeros((len(n_point),))
#     #
#     #     # point_list.append(np.concatenate([p_point, n_point], axis=0))
#     #     # label_list.append(np.concatenate([p_label, n_label], axis=0))
#     #     p_point_list.append(p_point)
#     #     p_label_list.append(p_label)
#     #     n_point_list.append(n_point)
#     #     n_label_list.append(n_label)
#     labels_gpu = torch.from_numpy(labels).cuda()
#     for i in range(1, num_labels):
#         all_point = torch.stack(torch.where(labels_gpu==i), dim=1)[:, [1, 0]]
#         random_point = torch.randint(0, all_point.shape[0], (4, ))
#         p_point = all_point[random_point]
#         p_label = torch.ones((len(p_point),))
#
#         neg_point = torch.stack(torch.where((labels_gpu != i) & (labels_gpu != 0)), dim=1)[:, [1, 0]]
#         random_neg_point = torch.randint(0, neg_point.shape[0], (100,))
#         n_point = neg_point[random_neg_point]
#         n_label = torch.zeros((len(n_point),))
#
#         p_point_list.append(p_point)
#         p_label_list.append(p_label)
#         n_point_list.append(n_point)
#         n_label_list.append(n_label)
#
#
#     result = torch.zeros((predictor.original_size[0], predictor.original_size[1]), dtype=torch.int16).to(divice)
#     all_mask = torch.zeros((len(p_point_list), 1024, 1024)).cuda()
#     all_area = torch.zeros((len(p_point_list), )).cuda()
#
#     # p_point_coords = np.array(p_point_list)
#     # p_point_labels = np.array(p_label_list)
#     # n_point_coords = np.array(n_point_list)
#     # n_point_labels = np.array(n_label_list)
#
#     p_point_coords = torch.stack(p_point_list, dim=0).cpu().data.numpy()
#     p_point_labels = torch.stack(p_label_list, dim=0).cpu().data.numpy()
#     n_point_coords = torch.stack(n_point_list, dim=0).cpu().data.numpy()
#     n_point_labels = torch.stack(n_label_list, dim=0).cpu().data.numpy()
#
#     point_mask = torch.zeros((1024, 1024)).cuda()
#     point_mask[p_point_coords[:, 0, 1], p_point_coords[:, 0, 0]] = 1
#
#     masks, scores, logits = predictor.predict(
#         point_coords=p_point_coords[:, :1, :],
#         point_labels=p_point_labels[:, :1],
#         multimask_output=True,
#     )
#
#     low_masks = logits > 0
#     low_masks_cpu = low_masks.cpu().data.numpy()
#
#     keep = []
#     keep_logits = []
#     for i in range(masks.shape[0]):
#         num_contours, area = CalContoursNum(low_masks_cpu[i])
#
#         index = np.argmin(np.array(num_contours))
#
#         num_min_contours = num_contours[index]
#         if num_min_contours > 3:
#             keep.append(0)
#             keep_logits.append(logits[i][index])
#             continue
#
#         thr = (point_mask * masks[i][index]).sum() / point_mask.sum()
#         if thr > 0.2:
#             keep.append(0)
#             keep_logits.append(logits[i][index])
#             continue
#         keep.append(1)
#         all_mask[i, ...] = masks[i][index]
#         all_area[i] = area[index]
#
#     keep = np.array(keep)
#     index1 = np.where(keep == 0)[0]
#     n = 0
#     while np.sum(1-keep)>0:
#         point_coords = np.concatenate([p_point_coords[index1, :1, :], n_point_coords[index1, :n+1, :]], axis=1)
#         point_labels = np.concatenate([p_point_labels[index1, :1], n_point_labels[index1, :n + 1]], axis=1)
#         mask_input = torch.stack(keep_logits, dim=0)[:, None, ...].cpu().data.numpy()
#         masks, scores, logits = predictor.predict(
#             point_coords=point_coords,
#             point_labels=point_labels,
#             mask_input=mask_input,
#             multimask_output=True,
#         )
#         low_masks = logits > 0
#         low_masks_cpu = low_masks.cpu().data.numpy()
#
#         keep_logits = []
#         for i in range(masks.shape[0]):
#             num_contours, area = CalContoursNum(low_masks_cpu[i])
#
#             index = np.argmin(np.array(num_contours))
#
#             num_min_contours = num_contours[index]
#             if num_min_contours > 3:
#                 keep_logits.append(logits[i][index])
#                 continue
#
#             thr = (point_mask * masks[i][index]).sum() / point_mask.sum()
#             if thr > 0.2:
#                 keep_logits.append(logits[i][index])
#                 continue
#             keep[index1[i]] = 1
#             all_mask[index1[i], ...] = masks[i][index]
#             all_area[index1[i]] = area[index]
#
#         index1 = np.where(keep == 0)[0]
#         n += 1
#
#     keep_mask = all_mask
#     label_all = torch.ones(keep_mask.shape[0]).to(divice)
#     _, _, _, keep_inds = mask_matrix_nms(keep_mask, label_all, label_all, filter_thr=0.3)
#
#     _, max_to_min = torch.sort(all_area[keep_inds], descending=True)
#     keep_end = keep_inds[max_to_min]
#     for k in range(keep_end.shape[0]):
#         i = keep_end[k]
#         result[keep_mask[i.item()] > 0] = i.item() + 1
#
#     return result.cpu().data.numpy()


def predict_fast_gpu_ssam_every(predictor, label, device="cuda:0"):
    """
    适合精细调整边界的问题，对于输入的label要求单个对象尽可能完整
    """
    sobel = Sobel().to(divice)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label.astype(np.uint8), connectivity=8)
    p_point_coords, p_point_labels, n_point_coords, n_point_labels = random_sample(torch.from_numpy(labels).cuda(), num_labels)

    labels_gpu = torch.from_numpy(labels).to(device)
    labels_gpu = F.one_hot(labels_gpu.long(), num_classes=num_labels).permute(2, 0, 1)[1:, ...].type(torch.uint8)
    result = torch.zeros((predictor.original_size[0], predictor.original_size[1]), dtype=torch.int16).to(device)

    all_mask = torch.zeros((p_point_coords.shape[0], predictor.original_size[0], predictor.original_size[1]), dtype=torch.uint8).to(device)
    all_area = torch.zeros((p_point_coords.shape[0], )).to(device)

    all_mask, all_area = iter_interaction_infer(predictor,
                                               p_point_coords,
                                               p_point_labels,
                                               n_point_coords,
                                               n_point_labels,
                                               all_mask,
                                               all_area,
                                               labels_gpu)

    # 获取正点,进行查漏迭代
    label_new, filt = get_p_point(all_mask, labels_gpu)
    n=0
    while filt.sum() > 1 and n < 5:
        p_point_coords, p_point_labels, n_point_coords, n_point_labels = random_sample(label_new,
                                                                                       filt.sum() + 1)
        label_new = F.one_hot(label_new.long(), num_classes=filt.sum() + 1).permute(2, 0, 1)[1:, ...]
        add_mask = torch.zeros((p_point_coords.shape[0], predictor.original_size[0], predictor.original_size[1]), dtype=torch.uint8).to(device)
        add_area = torch.zeros((p_point_coords.shape[0],)).to(device)
        add_mask, add_area = iter_interaction_infer(predictor,
                                                     p_point_coords,
                                                     p_point_labels,
                                                     n_point_coords,
                                                     n_point_labels,
                                                     add_mask,
                                                     add_area,
                                                     label_new)
        all_mask = torch.cat([all_mask, add_mask])
        all_area = torch.cat([all_area, add_area])
        label_new, filt = get_p_point(all_mask, labels_gpu)
        n+=1



    keep_mask = all_mask
    label_all = torch.ones(keep_mask.shape[0]).to(device)
    _, _, _, keep_inds = mask_matrix_nms(keep_mask, label_all, label_all, filter_thr=0.3)
    # 排序
    _, max_to_min = torch.sort(all_area[keep_inds], descending=True)
    keep_end = keep_inds[max_to_min]
    for k in range(keep_end.shape[0]):
        i = keep_end[k]
        result[keep_mask[i.item()] > 0] = i.item() + 1
    g = sobel(result[None, None, ...].float())
    g_mask = torch.sum(g[0].abs(), dim=0) == 0
    result = (result > 0) * g_mask
    return result.cpu().data.numpy()


def predict_fast_gpu_ssam(predictor, label, thr=0.98, divice="cuda:0"):
    """
    适合大面积提取，对地块要求不细，对label要求不高， thr越大图斑越少，单个图斑质量越好
    """
    sobel = Sobel().to(divice)
    model_size = predictor.model.image_encoder.img_size
    x_arange = np.arange(0, label.shape[1], 100)
    y_arange = np.arange(0, label.shape[0], 100)
    xv, yv = np.meshgrid(x_arange, y_arange)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)

    p_point_mask = label[yv, xv] == 1

    point_list = np.concatenate([xv[..., None], yv[..., None]], axis=1)

    input_point = np.array(point_list)[p_point_mask]
    if input_point.shape[0] == 0:
        input_point = np.array(np.where(label==1)).T[:, [1, 0]][:1, :]

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
        if not (num_min_contours < 3 or logits_p[i][index][logits_p[i][index] > 0.5].mean() > thr):
            continue

        # if masks[i][index].sum() / (masks[i][index].shape[0]*masks[i][index].shape[1]) > 0.6:
        #     continue

        if ((result > 0) * masks[i][index]).sum() / (masks[i][index].sum()) > 0.9:
            continue

        keep.append([i, index, area[index]])
    if len(keep) == 0:
        return result.cpu().data.numpy()
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
    return result.cpu().data.numpy()


def predict_fast_gpu_everything(predictor, divice="cuda:0"):
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

    return result.cpu().data.numpy()


import time
from datetime import datetime
def predict(model_type, weight_path, input_path0, input_path1, output_path, clip_size, clip_crop, batch_size=1,
             model_input_size=512, img_ratios=[0.5, 1.0, 2.0], divice='cuda:0'):
    begin_process = time.time()
    if divice == 'cpu':
        divice = 'cpu'
    else:
        divice = 'cuda:0'

    # 当保存整个模型的时候
    predictor = model_load(weight_path=weight_path, model_type=model_type, device=divice)

    input_list1 = []
    input_list2 = []
    if os.path.isdir(input_path1):
        input_list1 = all_path(input_path0)
        input_list2 = all_path(input_path1)
    else:
        input_list1.append(input_path0)
        input_list2.append(input_path1)

    now_index = 1
    for input_path in zip(input_list1, input_list2):
        image_embeddings_dict = {}
        pre_generator = clip_img(input_path[0], input_path[1], clip_size=clip_size, clip_crop=clip_crop)

        img_raster1 = gdal.Open(input_path[0], gdal.GA_ReadOnly)
        im_width = img_raster1.RasterXSize  # 栅格矩阵的列数
        im_height = img_raster1.RasterYSize  # 栅格矩阵的行数
        im_chanel = img_raster1.RasterCount

        ref_transform = img_raster1.GetGeoTransform()
        target_raster = gdal.GetDriverByName('GTiff').Create(output_path, im_width, im_height, 1, gdal.GDT_UInt16, options = ["TILED=YES", "COMPRESS=LZW"])
        target_raster.SetGeoTransform(ref_transform)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromWkt(img_raster1.GetProjectionRef())
        target_raster.SetProjection(outRasterSRS.ExportToWkt())
        read_time = 0
        write_time = 0
        pre_time = 0
        n=0
        while (pre_generator):
            batch_true_sum = 0
            time_b = time.time()
            try:
                img_buffer1, img_buffer2, begin_x, begin_y, offset_x, offset_y, write_width, write_height, index_xy = next(
                    pre_generator)
                batch_true_sum += 1
            except Exception as e:
                break
            read_info_list = [(begin_x, begin_y, offset_x, offset_y, write_width, write_height)]
            batch_x1 = np.zeros((batch_size, img_buffer1.shape[0], img_buffer1.shape[1], img_buffer1.shape[2]),
                                dtype=np.float)
            batch_x2 = np.zeros((batch_size, img_buffer2.shape[0], img_buffer2.shape[1], img_buffer2.shape[2]),
                                dtype=np.float)

            batch_x1[0, :] = img_buffer1[:]
            batch_x2[0, :] = img_buffer2[:]
            for batch_index in range(1, batch_size):
                try:
                    img_buffer1, img_buffer2, begin_x, begin_y, offset_x, offset_y, write_width, write_height, index_xy = next(pre_generator)
                    batch_true_sum += 1
                except Exception as e:
                    break
                read_info_list.append((begin_x, begin_y, offset_x, offset_y, write_width, write_height))
                batch_x1[batch_index, :] = img_buffer1[:]
                batch_x2[batch_index, :] = img_buffer2[:]

            read_time += (time.time() - time_b)
            time_b = time.time()


            # x1_m = np.array(
            #     [cv2.resize(i, (int(model_input_size*img_ratios[1]), int(model_input_size*img_ratios[1])), cv2.INTER_LINEAR) for i in batch_x1])
            x = np.array(batch_x1)
            label = np.array(batch_x2)
            mask = batch_x1[0].copy()
            mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            mask = np.where((mask == 0) + (mask==255), 0, 1)
            if mask.sum() == 0 or label[0, ..., 0].sum()==0:
                y = label[0, ..., 0]
            else:
                predictor.set_image(x[0].astype(np.uint8))

                y = predict_fast_gpu_ssam_every(predictor, label[0, ..., 0])
                # y = y * mask
                if img_buffer1.shape[0] != clip_size or img_buffer1.shape[1] != clip_size:
                    y_pad = np.zeros((clip_size, clip_size))
                    y_pad[:img_buffer1.shape[0], :img_buffer1.shape[1]] = y
                    y = y_pad

                if 'cuda' in divice:
                    torch.cuda.empty_cache()

            y = [y.astype('uint16')]
            y = np.array([cv2.resize(i, (clip_size, clip_size), cv2.INTER_NEAREST) for i in y])

            pre_time += (time.time() - time_b)

            time_b = time.time()

            for batch_index in range(batch_true_sum):
                print(batch_true_sum, batch_index)
                yy = y[batch_index, ::]

                if len(yy.shape) == 1:
                    yy = np.reshape(yy, [batch_x1.shape[1], batch_x1.shape[2]])

                # yy = yy.astype('uint8')

                begin_x = read_info_list[batch_index][0]
                begin_y = read_info_list[batch_index][1]
                offset_y = read_info_list[batch_index][3]
                offset_x = read_info_list[batch_index][2]
                write_height = read_info_list[batch_index][5]
                write_width = read_info_list[batch_index][4]


                yy = yy[offset_y:offset_y + write_height, offset_x:offset_x + write_width]

                target_raster.WriteRaster(begin_x, begin_y, write_width, write_height, yy.tostring())

            write_time += time.time() - time_b

    end_process = time.time()
    print('infer_time cost: ', pre_time)
    print('all cost: ', end_process - begin_process)



if __name__ == '__main__':
    # 单输入只有input_path0起作用，input_path1是变化那边的
    # model_type = sys.argv[1]
    # weight_path = sys.argv[2]
    # input_path0 = sys.argv[3]
    # output_dir = sys.argv[4]
    # divice = sys.argv[5]

    model_type = "vit_h"
    weight_path = "sam_vit_h_4b8939.pth"
    input_path0 = r'./14-2012-0420-6900-LA93-0M50-E080.tif'
    input_path1 = r'./14-2012-0420-6900-LA93-0M50-E080_result0313.tif'
    output_dir = r'./result/14-2012-0420-6900-LA93-0M50-E080_ssam_result.tif'
    divice = 'cuda'
    predict(model_type, weight_path, input_path0, input_path1, output_dir,
             clip_size=1024, clip_crop=128, model_input_size=1024, divice=divice)

