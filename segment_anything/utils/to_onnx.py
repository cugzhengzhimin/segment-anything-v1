#!/usr/bin/env python
# -*_ coding: utf-8 -*-
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import onnx
import onnxruntime
import numpy as np
import sys
from mmcv.onnx import register_extra_symbolics
import cv2


def pytorch2onnx(model, input_names, inputs, output_names, dynamic_axes, output_file):
    # model.cpu().eval() # 若存在batchnorm、dropout层则一定要eval()!!!!再export,这样使得这些层的参数不再更新
    register_extra_symbolics(11)
    with torch.no_grad():
        torch.onnx.export(
            model, inputs,
            output_file,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=False,
            opset_version=11,
            dynamic_axes=dynamic_axes)
        print(f'Successfully exported ONNX model: {output_file}')



def test_onnx(onnx_path, torch_path):
    #检验onnx模型是否有错误
    model = onnx.load(onnx_path)  # 加载onnx
    onnx.checker.check_model(model)  # 检查生成模型是否错误

    #真正的推理测试
    device = 'gpu'
    if device == 'cpu':
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    else:
        try:
            ort_session = onnxruntime.InferenceSession(onnx_path,
                                                   providers=[('CUDAExecutionProvider', {"device_id": 0})])
        except Exception as e:
            ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])


    TestImgLoader = [(
                     '/geoway/sample_tiles/private_dataset/01_optical/01_segmentation/singleClass/01_building/roof_zhujianbu/roof_GF7_3606_51264/x1_left/gf7_3606_15.tif',
                     '/geoway/sample_tiles/private_dataset/01_optical/01_segmentation/singleClass/01_building/roof_zhujianbu/roof_GF7_3606_51264/x2_right/gf7_3606_15.tif')]
    dis_path = '/geoway/sample_tiles/private_dataset/01_optical/01_segmentation/singleClass/01_building/roof_zhujianbu/roof_GF7_3606_51264/x3_disp/gf7_3606_15.tif'

    ort_inputs = {ort_session.get_inputs()[0].name: left, ort_session.get_inputs()[1].name: right}
    for i in range(10000):
        ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0].shape)

    # #使用np.testing测试onnx和torch模型输出是否一致
    # device = 'cpu'
    # state_dict = torch.load(weight_path)
    # min_disp = -128
    # max_disp = 64
    # net = vitea_uper(in_channels=2)
    # net.load_state_dict(state_dict)
    # with torch.no_grad():
    #     leftX = torch.from_numpy(x_left)
    #     rightX = torch.from_numpy(x_right)
    #     torch_out = net(leftX, rightX)
    # np.testing.assert_allclose(torch_out[0].cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    bConvert = True
    weight_path = r'/home/zhengzhimin/ACVNet-main/log/vitea_build_dis1214.pth'
    out_onnx_path = r'/home/zhengzhimin/ACVNet-main/log/vitea_build_dis1214.onnx'
    if bConvert:
        # 单输入
        input_names = ["input"]
        output_names = ["output_0"]

        # # 多输入
        # input_names = ["input", "input1"]
        # output_names = ["output_0"]

        # input_left = torch.rand(1, 1, 768, 768)
        # input_right = torch.rand(1, 1, 768, 768)
        # inputs = (input_left, input_right)

        # mul, pan, ref = torch.rand(1, 3, 256, 256), torch.rand(1, 1, 256, 256), torch.rand(1, 3, 256, 256)
        # input_left = torch.cat([mul, pan, ref], 1)

        input_left = torch.rand(1, 2, 1024, 1024).cuda()

        # min_disp = -128
        # max_disp = 64

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = 'cpu'
        state_dict = torch.load(weight_path, map_location=torch.device('cpu')).module.state_dict()
        # net = res_swin_class(in_channels=6)
        # net.cpu().eval()
        # net.load_state_dict(state_dict)

        net = vitea_uper(in_channels=2)
        # net = oop_uper(in_channels=2)
        net.cuda().eval()
        net.load_state_dict(state_dict)

        dynamic_axes = {
            'input_left': {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'input_right': {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'output_0': {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'output_1': {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'output_2': {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
        }

        pytorch2onnx(net, input_names, input_left, output_names, None, out_onnx_path)
    else:
        test_onnx(out_onnx_path, weight_path)





