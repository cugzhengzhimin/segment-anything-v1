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


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        predictions = np.where(predictions > 0.8, 1, 0)
        gts = np.where(gts > 0.8, 1, 0)
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        print(iu)
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


# def eval(pred, target):



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



def get_infer_transform():
    transform = A.Compose([
        ToTensor(),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ToTensorV2(),
    ])
    return transform



def all_path(dir_path):
    file_list = []
    for maindir, subdir, file_name_list in os.walk(dir_path):
        for filename in file_name_list:
            if os.path.splitext(filename)[1] in ['.tif', '.img']:
                apath = os.path.join(maindir, filename)
                file_list.append(apath)
    return file_list


def parse_hdr_rpc(dir_hdr, dir_hdr_original):
    flag = 0
    with open(dir_hdr_original) as f:
        rpc = 'rpc info = {\n'
        for line in f:
            line.replace('\n', '')
            if 'fwhm' in line:
                flag = 0
            if flag:
                rpc += line
            if 'rpc info = {' in line:
                flag = 1
    rpc_list = rpc.strip('\n').strip('}')
    f = open(dir_hdr, 'a')
    f.writelines(rpc_list)


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


def clip_img2(input_path1, input_path2, clip_size, clip_crop):
    img_raster1 = gdal.Open(input_path1, gdal.GA_ReadOnly)
    im_chanel = img_raster1.RasterCount
    im_width = img_raster1.RasterXSize  # 栅格矩阵的列数
    im_height = img_raster1.RasterYSize  # 栅格矩阵的行数

    # 规则分块,这里clip_size是真正输出的大小，true_clip_size是裁切分块的大小
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
                if im_chanel == 1:
                    img_buffer1 = np.reshape(img_buffer1, [readheight, readwidth, 1]).repeat(3, axis=-1)
                    if readwidth != clip_size or readheight != clip_size:
                        img_buffer1_pad = np.zeros((clip_size, clip_size, img_buffer1.shape[-1]))
                        img_buffer1_pad[:img_buffer1.shape[0], :img_buffer1.shape[1], :] = img_buffer1
                        img_buffer1 = img_buffer1_pad

                else:
                    img_buffer1 = np.transpose(img_buffer1, [1, 2, 0])[..., [0, 1, 2]]
                    # if readwidth != clip_size or readheight != clip_size:
                    #     img_buffer1_pad = 0 * np.ones((clip_size, clip_size, 3))
                    #     img_buffer1_pad[:img_buffer1.shape[0], :img_buffer1.shape[1], :] = img_buffer1
                    #     img_buffer1 = img_buffer1_pad

            except Exception as e:
                print(e)
                # print(begin_x, begin_y, readwidth, readheight)

            print(true_begin_x, true_begin_y, true_begin_x - begin_x, true_begin_y - begin_y, write_width, write_height,
                  end_x, end_y)
            index_xy = [w, h]
            yield img_buffer1, true_begin_x, true_begin_y, true_begin_x - begin_x, true_begin_y - begin_y, write_width, write_height, index_xy


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
    _, _, _, c = Image.shape
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
        if np.max(correctImage) == np.min(correctImage):
            correctImage = correctImage / 255
        else:
            correctImage = 1 * ((correctImage - np.min(correctImage)) / (np.max(correctImage) - np.min(correctImage)))
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


# def predict_everything(predictor):
#     model_size = predictor.model.image_encoder.img_size
#     input_image = predictor.input_image[0].permute(1, 2, 0).cpu().data.numpy()
#     input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
#     x_arange = np.arange(0, model_size, 50)
#     y_arange = np.arange(0, model_size, 50)
#     xv, yv = np.meshgrid(x_arange, y_arange)
#     xv = xv.reshape(-1)
#     yv = yv.reshape(-1)
#     point_list = np.concatenate([xv[..., None], yv[..., None]], axis=1)
#
#     input_point = np.array(point_list)
#     result = np.zeros((predictor.original_size[0], predictor.original_size[1])).astype(np.uint16)
#     for i in range(0, len(point_list)):
#
#         if input_image[input_point[i][1], input_point[i][0]] == 0:
#             continue
#
#         # 初步预测
#         input_label = np.array([1])
#         masks, scores, logits = predictor.predict(
#             point_coords=input_point[i:i + 1],
#             point_labels=input_label,
#             multimask_output=True,
#         )
#         num_contours = CalContoursNum(masks)
#         index = np.argmin(np.array(num_contours))
#
#         num_min_contours = num_contours[index]
#         if num_min_contours > 3:
#             continue
#         # if masks[index].sum() / (1280*1280) > 0.1:
#         #     continue
#         if ((result > 0) * masks[index]).sum() / (masks[index].sum()) > 0.9:
#             continue
#         temp = (1 - (result > 0)) * masks[index]
#         result[temp > 0] = i + 1
#     return result


def predict_fast_everything(predictor):
    model_size = predictor.model.image_encoder.img_size
    x_arange = np.arange(0, model_size, 50)
    y_arange = np.arange(0, model_size, 50)
    xv, yv = np.meshgrid(x_arange, y_arange)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    point_list = np.concatenate([xv[..., None], yv[..., None]], axis=1)

    input_point = np.array(point_list)
    result = np.zeros((predictor.original_size[0], predictor.original_size[1])).astype(np.uint16)
    # result = torch.zeros((predictor.original_size[0], predictor.original_size[1]), dtype=torch.int16).cuda()
    # import time
    # s = time.time()
    masks, scores, logits = predictor.predict(
        point_coords=input_point[:, None, :],
        point_labels=np.ones((input_point.shape[0], 1)),
        multimask_output=True,
    )
    logits_p = logits.sigmoid()
    low_masks = logits > 0
    low_masks = low_masks.cpu().data.numpy()
    masks = masks.cpu().data.numpy()
    # masks_cuda = torch.from_numpy(masks.astype(np.uint8)).cuda()

    for i in range(masks.shape[0]):
        num_contours = CalContoursNum(low_masks[i])

        index = np.argmin(np.array(num_contours))

        num_min_contours = num_contours[index]
        if not (num_min_contours < 3 or logits_p[i][index][logits_p[i][index] > 0.5].mean() > 0.98):
            continue

        if masks[i][index].sum() / (masks[i][index].shape[0] * masks[i][index].shape[1]) > 0.6:
            continue

        if ((result > 0) * masks[i][index]).sum() / (masks[i][index].sum()) > 0.9:
            continue
        temp = (1 - (result > 0)) * masks[i][index]
        result[temp > 0] = i + 1

    # print(time.time()-s)
    return result

from sobel import Sobel
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


import time
from datetime import datetime
def predict3(model_type, weight_path, input_path0, output_path, clip_size, clip_crop, batch_size=1,
             model_input_size=512, img_ratios=[0.5, 1.0, 2.0], divice='cuda:0'):
    begin_process = time.time()
    if divice == 'cpu':
        divice = 'cpu'
    else:
        divice = 'cuda:0'

    # 当保存整个模型的时候
    predictor = model_load(weight_path=weight_path, model_type=model_type, device=divice)

    input_list1 = []
    if os.path.isdir(input_path0):
        input_list1 = all_path(input_path0)
    else:
        input_list1.append(input_path0)

    now_index = 1
    for input_path in input_list1:
        image_embeddings_dict = {}
        pre_generator = clip_img2(input_path, input_path, clip_size=clip_size, clip_crop=clip_crop)

        img_raster1 = gdal.Open(input_path, gdal.GA_ReadOnly)
        im_width = img_raster1.RasterXSize  # 栅格矩阵的列数
        im_height = img_raster1.RasterYSize  # 栅格矩阵的行数
        im_chanel = img_raster1.RasterCount

        ref_transform = img_raster1.GetGeoTransform()
        target_raster = gdal.GetDriverByName('GTiff').Create(output_path, im_width, im_height, 1, gdal.GDT_Byte, options = ["TILED=YES", "COMPRESS=LZW"])
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
                img_buffer1, begin_x, begin_y, offset_x, offset_y, write_width, write_height, index_xy = next(
                    pre_generator)
                batch_true_sum += 1
            except Exception as e:
                break
            read_info_list = [(begin_x, begin_y, offset_x, offset_y, write_width, write_height)]
            batch_x1 = np.zeros((batch_size, img_buffer1.shape[0], img_buffer1.shape[1], img_buffer1.shape[2]),
                                dtype=np.float)

            batch_x1[0, :] = img_buffer1[:]
            for batch_index in range(1, batch_size):
                try:
                    img_buffer1, begin_x, begin_y, offset_x, offset_y, write_width, write_height, index_xy = next(pre_generator)
                    batch_true_sum += 1
                except Exception as e:
                    break
                read_info_list.append((begin_x, begin_y, offset_x, offset_y, write_width, write_height))
                batch_x1[batch_index, :] = img_buffer1[:]

            read_time += (time.time() - time_b)
            time_b = time.time()


            # x1_m = np.array(
            #     [cv2.resize(i, (int(model_input_size*img_ratios[1]), int(model_input_size*img_ratios[1])), cv2.INTER_LINEAR) for i in batch_x1])
            x = np.array(batch_x1)
            mask = batch_x1[0].copy()
            mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            mask = np.where((mask == 0) + (mask==255), 0, 1)
            if mask.sum() == 0:
                y = mask
            else:
                predictor.set_image(x[0].astype(np.uint8))
                key = str(begin_x - offset_x) + ',' + str(begin_y - offset_y) + ',' + '1024,1024'
                image_embeddings_dict[key] = predictor.features.cpu().data.numpy().tolist()
                y = predict_fast_gpu_everything(predictor, divice=divice)
                # y = y * mask
                if img_buffer1.shape[0] != clip_size or img_buffer1.shape[1] != clip_size:
                    y_pad = np.zeros((clip_size, clip_size))
                    y_pad[:img_buffer1.shape[0], :img_buffer1.shape[1]] = y
                    y = y_pad

                if 'cuda' in divice:
                    torch.cuda.empty_cache()

            y = [255 * y.astype('uint8')]
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

                # if index_xy == [4, 1]:
                #     print(0)
                # if index_xy[0] == 0 and index_xy[1] == 0:
                #     yy = yy
                # elif index_xy[1] == 0 and index_xy[0] != 0:
                #     # 左边块的右边的边
                #     e_r = target_raster.ReadAsArray(xoff=begin_x-1, yoff=0, xsize=1, ysize=write_height)
                #     class_all = np.unique(yy[:write_height, offset_x])
                #     for i in range(class_all.shape[0]):
                #         # if class_all[i] == 0:
                #         #     continue
                #         a, b = np.unique(e_r[np.where(yy[:write_height, offset_x] == class_all[i])[0], :],
                #                          return_counts=True)
                #         replace = a[np.argmax(b)]
                #         # if replace == 0:
                #         #     continue
                #         yy[yy==class_all[i]] = replace
                # elif index_xy[1] != 0 and index_xy[0] == 0:
                #     e_t = target_raster.ReadAsArray(xoff=0, yoff=begin_y-1, xsize=write_width, ysize=1)
                #     class_all = np.unique(yy[offset_y, :write_width])
                #     for i in range(class_all.shape[0]):
                #         # if class_all[i] == 0:
                #         #     continue
                #         a, b = np.unique(e_t[:, np.where(yy[offset_y, :write_width] == class_all[i])[0]],
                #                          return_counts=True)
                #         replace = a[np.argmax(b)]
                #         # if replace == 0:
                #         #     continue
                #         yy[yy == class_all[i]] = replace
                # elif index_xy[1] != 0 and index_xy[0] != 0:
                #     e_r = target_raster.ReadAsArray(xoff=begin_x - 1, yoff=begin_y, xsize=1, ysize=write_height)
                #     class_all = np.unique(yy[offset_y:write_height+offset_y, offset_x])
                #     for i in range(class_all.shape[0]):
                #         # if class_all[i] == 0:
                #         #     continue
                #         a, b = np.unique(e_r[np.where(yy[offset_y:write_height+offset_x, offset_x] == class_all[i])[0], :],
                #                          return_counts=True)
                #         replace = a[np.argmax(b)]
                #         # if replace == 0:
                #         #     continue
                #         yy[yy == class_all[i]] = replace
                #     e_t = target_raster.ReadAsArray(xoff=begin_x, yoff=begin_y - 1, xsize=write_width, ysize=1)
                #     class_all = np.unique(yy[offset_y, offset_x:offset_x+write_width])
                #     for i in range(class_all.shape[0]):
                #         # if class_all[i] == 0:
                #         #     continue
                #         a, b = np.unique(e_t[:, np.where(yy[offset_y, offset_x:offset_x+write_width] == class_all[i])[0]],
                #                          return_counts=True)
                #         replace = a[np.argmax(b)]
                #         # if replace == 0:
                #         #     continue
                #         yy[yy == class_all[i]] = replace

                yy = yy[offset_y:offset_y + write_height, offset_x:offset_x + write_width]

                target_raster.WriteRaster(begin_x, begin_y, write_width, write_height, yy.tostring())
                # if index_xy[1] == 1 and index_xy[0] == 1:
                #     target_raster = None
            write_time += time.time() - time_b
        # json_out_path = os.path.splitext(output_path)[0] + '_image_embedding.json'
        # with open(json_out_path, 'w') as f:
        #     json.dump(image_embeddings_dict, f, ensure_ascii=False)

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
    weight_path = "./sam_vit_h_4b8939.pth"
    input_path0 = r'./bright-0_0_005_007.bmp'
    output_dir = r'./bright-0_0_005_007.tif'
    divice = 'cuda'
    predict3(model_type, weight_path, input_path0, output_dir,
             clip_size=1024, clip_crop=128, model_input_size=1024, divice=divice)

