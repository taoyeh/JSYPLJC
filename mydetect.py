# 检测驾驶员是否分心，抽烟，睡觉

import numpy as np
import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized


# ====== 模型参数配置 ======
weights = r'weights/best.pt'  # 模型权重文件路径
opt_device = ''  # 设备选择: ''为自动选择, 'cpu'为CPU, '0'为第一个GPU
imgsz = 640  # 输入图像尺寸
opt_conf_thres = 0.6  # 置信度阈值(0-1之间)
opt_iou_thres = 0.45  # IOU阈值(用于非极大值抑制)

# ====== 初始化设置 ======
set_logging()  # 设置日志
device = select_device(opt_device)  # 选择设备(CPU或GPU)
half = device.type != 'cpu'  # 是否使用半精度(FP16) - GPU支持半精度

# ====== 加载模型 ======
model = attempt_load(weights, map_location=device)  # 加载FP32模型
imgsz = check_img_size(imgsz, s=model.stride.max())  # 检查图像尺寸是否符合模型要求
if half:
    model.half()  # 转换为FP16半精度
# ====== 获取类别名称和颜色 ======
names = model.module.names if hasattr(model, 'module') else model.names  # 获取类别名称
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]  # 为每个类别生成随机颜色

# 调用模型检测
def predict(im0s):
    # 对模型的预热
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # 初始化图像
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # 运行一次推理
    # ====== 图像预处理 ======
    # 使用letterbox调整图像尺寸
    img = letterbox(im0s, new_shape=imgsz)[0]
    # 转换颜色通道和维度顺序: BGR->RGB, HWC->CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)  # 确保内存连续

    # ====== 转换为PyTorch张量 ======
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # 转换为半精度或单精度
    img /= 255.0  # 归一化到[0,1]范围
    if img.ndimension() == 3:  # 如果是3维张量(没有批次维度)
        img = img.unsqueeze(0)  # 添加批次维度1*3x416x416

    # ====== 推理 ======
    pred = model(img)[0]  # 模型推理

    # ====== 非极大值抑制(NMS) ======
    pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres)

    # ====== 处理检测结果 ======
    ret = []  # 存储最终结果
    for i, det in enumerate(pred):  # 遍历每个检测结果(通常只有一个)
        if len(det):  # 如果有检测到目标
            # 将边界框坐标从缩放后的图像尺寸转换回原始图像尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            # 遍历每个检测到的目标
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]}'  # 获取类别标签
                prob = round(float(conf) * 100, 2)  # 计算置信度百分比(保留2位小数)
                ret_i = [label, prob, xyxy]  # 存储结果: [标签, 置信度, 边界框坐标]
                ret.append(ret_i)
        # 返回检测结果
        # 每个结果包含:
        #   label: 检测到的类别名称 ('face', 'smoke', 'drink', 'phone'等)
        #   prob: 置信度百分比 (0-100)
        #   xyxy: 边界框坐标 (左上角x, 左上角y, 右下角x, 右下角y)
    return ret


# 根据检测的结果，在frame上标注classs和绘制方框
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # 获取原始图像的宽高
    shape = img.shape[:2]  # [高度, 宽度]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例 (新尺寸/原尺寸)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 如果禁止放大，确保比例≤1
        r = min(r, 1.0)
    # 计算新尺寸和填充
    ratio = r, r  # 宽高比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 缩放后尺寸
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 需要填充的像素

    if auto:  # 自动填充，确保尺寸是32的倍数(YOLO要求)
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # 取模32
    elif scaleFill:  # 拉伸填充
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2  # 将填充分为两侧
    dh /= 2

    # 调整图像大小
    if shape[::-1] != new_unpad:  # 如果需要缩放
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # 双线性插值

    # 添加边框
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, (dw, dh)