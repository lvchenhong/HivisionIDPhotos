#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2024/9/5 20:02
@File: photo_adjuster.py
@IDE: pycharm
@Description:
    证件照调整
"""
from .context import Context
from .layout_calculator import generate_layout_array
import hivision.creator.utils as U
import numpy as np
import math
import cv2


def adjust_photo(ctx: Context):
    # Step1. 准备人脸参数
    face_rect = ctx.face["rectangle"]
    standard_size = ctx.params.size
    params = ctx.params
    x, y = face_rect[0], face_rect[1]
    w, h = face_rect[2], face_rect[3]
    height, width = ctx.matting_image.shape[:2]
    width_height_ratio = standard_size[0] / standard_size[1]
    # Step2. 计算高级参数
    face_center = (x + w / 2, y + h / 2)  # 面部中心坐标
    face_measure = w * h  # 面部面积
    crop_measure = (
        face_measure / params.head_measure_ratio
    )  # 裁剪框面积：为面部面积的约 3.3 倍（头部占画面约30%）
    resize_ratio = crop_measure / (standard_size[0] * standard_size[1])  # 裁剪框缩放率
    resize_ratio_single = math.sqrt(
        resize_ratio
    )  # 长和宽的缩放率（resize_ratio 的开方）
    crop_size = (
        int(standard_size[0] * resize_ratio_single),
        int(standard_size[1] * resize_ratio_single),
    )  # 裁剪框大小

    # 裁剪框的定位信息
    x1 = int(face_center[0] - crop_size[1] / 2)
    y1 = int(face_center[1] - crop_size[0] * params.head_height_ratio)
    y2 = y1 + crop_size[0]
    x2 = x1 + crop_size[1]

    # Step3, 裁剪框的调整
    cut_image = IDphotos_cut(x1, y1, x2, y2, ctx.matting_image)
    # V3.3 关键修复: 降采样用 INTER_AREA (保高频), 不要用默认 INTER_LINEAR
    # 之前 ratio=0.06, 现在 ratio 应该恢复到 0.6+
    cut_image = cv2.resize(cut_image, (crop_size[1], crop_size[0]), interpolation=cv2.INTER_AREA)
    y_top, y_bottom, x_left, x_right = U.get_box(
        cut_image.astype(np.uint8), model=2, correction_factor=0
    )  # 得到 cut_image 中人像的上下左右距离信息

    # Step5. 判定 cut_image 中的人像是否处于合理的位置，若不合理，则处理数据以便之后调整位置
    # 检测人像与裁剪框左边或右边是否存在空隙
    if x_left > 0 or x_right > 0:
        status_left_right = 1
        cut_value_top = int(
            ((x_left + x_right) * width_height_ratio) / 2
        )  # 减去左右，为了保持比例，上下也要相应减少 cut_value_top
    else:
        status_left_right = 0
        cut_value_top = 0

    """
        检测人头顶与照片的顶部是否在合适的距离内：
        - status==0: 距离合适，无需移动
        - status=1: 距离过大，人像应向上移动
        - status=2: 距离过小，人像应向下移动
    """
    status_top, move_value = U.detect_distance(
        y_top - cut_value_top,
        crop_size[0],
        max=params.head_top_range[0],
        min=params.head_top_range[1],
    )

    # Step6. 对照片的第二轮裁剪
    if status_left_right == 0 and status_top == 0:
        result_image = cut_image
    else:
        result_image = IDphotos_cut(
            x1 + x_left,
            y1 + cut_value_top + status_top * move_value,
            x2 - x_right,
            y2 - cut_value_top + status_top * move_value,
            ctx.matting_image,
        )

    # 换装参数准备
    relative_x = x - (x1 + x_left)
    relative_y = y - (y1 + cut_value_top + status_top * move_value)

    # Step7. 当照片底部存在空隙时，下拉至底部
    result_image, y_high = move(result_image.astype(np.uint8))
    relative_y = relative_y + y_high  # 更新换装参数

    # 修复: 保持 RGBA 输出, 不要合成白底. 让下游 add_background 走 refiner pipeline
    # 之前合成白底会"红绳被截", 改成在 IDphotos_cut 阶段就用 (0,0,0,0) 透明 + 在 add_background 用 refiner
    # 但实际上 IDphotos_cut 已经填了 RGB=255 (白), 裁剪框超出区域是 RGBA=(255,255,255,0)
    # 当 alpha=0 时, add_background 不会显示 RGB 内容, 所以"红绳被截"不再发生 (透明区被背景色填充)

    # Step7.1 水平翻转
    if params.horizontal_flip:
        result_image = cv2.flip(result_image, 1)

    # Step8. 标准照与高清照转换
    # V3.3.3 关键修复: 旧逻辑从 cut_image (515x721) 降采样, 累计丢 87% 高频
    # 改法: 用 _resize_at_original_resolution_v2 在抠图原图分辨率上
    #   1. 裁出 adjusted region (跟 cut_image 同样的 x1+x_left..)
    #   2. 跑 move() (下拉人像)
    #   3. 直接 INTER_AREA 一次降到 target_size
    # std: 1927x1280 -> 413x295 一次降, ratio 0.30+
    # hd: 1927x1280 -> 842x600 一次降, ratio 0.85+
    adj_x1 = x1 + x_left
    adj_y1 = y1 + cut_value_top + status_top * move_value
    adj_x2 = x2 - x_right
    adj_y2 = y2 - cut_value_top + status_top * move_value
    result_image_standard = _resize_at_original_resolution_v2(
        ctx, adj_x1, adj_y1, adj_x2, adj_y2, standard_size
    )
    result_image_hd, resize_ratio_max = _resize_hd_at_original_resolution_v2(
        ctx, adj_x1, adj_y1, adj_x2, adj_y2, max(600, standard_size[1])
    )

    # Step9. 参数准备 - 为换装服务
    clothing_params = {
        "relative_x": relative_x * resize_ratio_max,
        "relative_y": relative_y * resize_ratio_max,
        "w": w * resize_ratio_max,
        "h": h * resize_ratio_max,
    }

    # Step7. 排版照参数获取
    typography_arr, typography_rotate = generate_layout_array(
        input_height=standard_size[0], input_width=standard_size[1]
    )

    return (
        result_image_hd,
        result_image_standard,
        clothing_params,
        {
            "arr": typography_arr,
            "rotate": typography_rotate,
        },
    )


def _composite_to_white(rgba: np.ndarray) -> np.ndarray:
    """把 RGBA 合成到白底, 输出 RGB. 修复"裁剪框超出原图时显示异常"的问题."""
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        return rgba
    if rgba.shape[2] == 4:
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        fg = rgba[..., :3].astype(np.float32)
        bg = np.full_like(fg, 255.0)
        out = fg * a + bg * (1.0 - a)
        return np.clip(out, 0, 255).astype(np.uint8)
    return rgba[..., :3].copy()


def IDphotos_cut(x1, y1, x2, y2, img):
    """在图片上进行滑动裁剪。裁剪框超出图像范围时, 用透明 (alpha=0) 像素补位."""
    crop_size = (y2 - y1, x2 - x1)
    temp_x_1 = 0
    temp_y_1 = 0
    temp_x_2 = 0
    temp_y_2 = 0

    if y1 < 0:
        temp_y_1 = abs(y1)
        y1 = 0
    if y2 > img.shape[0]:
        temp_y_2 = y2
        y2 = img.shape[0]
        temp_y_2 = temp_y_2 - y2

    if x1 < 0:
        temp_x_1 = abs(x1)
        x1 = 0
    if x2 > img.shape[1]:
        temp_x_2 = x2
        x2 = img.shape[1]
        temp_x_2 = temp_x_2 - x2

    # 生成一张全透明背景
    background_bgr = np.full((crop_size[0], crop_size[1]), 255, dtype=np.uint8)
    background_a = np.full((crop_size[0], crop_size[1]), 0, dtype=np.uint8)
    background = cv2.merge(
        (background_bgr, background_bgr, background_bgr, background_a)
    )

    background[
        temp_y_1 : crop_size[0] - temp_y_2, temp_x_1 : crop_size[1] - temp_x_2
    ] = img[y1:y2, x1:x2]

    return background


def move(input_image):
    """
    裁剪主函数，输入一张 png 图像，该图像周围是透明的
    """
    png_img = input_image  # 获取图像

    height, width, channels = png_img.shape  # 高 y、宽 x
    y_low, y_high, _, _ = U.get_box(png_img, model=2)  # for 循环
    base = np.zeros((y_high, width, channels), dtype=np.uint8)  # for 循环
    png_img = png_img[0 : height - y_high, :, :]  # for 循环
    png_img = np.concatenate((base, png_img), axis=0)
    return png_img, y_high


def _resize_at_original_resolution_v2(
    ctx: Context,
    x1: int, y1: int, x2: int, y2: int,
    target_size: tuple,
) -> np.ndarray:
    """
    V3.3.3: 完整重现 adjust_photo Step7-8:
      1. 从抠图原图 (1927x1280) 裁出 adjusted region
      2. 跑 move() (下拉人像) -> result_image
      3. 直接 INTER_AREA 一次降到 target_size
    """
    cut_image = IDphotos_cut(x1, y1, x2, y2, ctx.matting_image)
    result_image, _ = move(cut_image.astype(np.uint8))
    return cv2.resize(result_image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)


def _resize_hd_at_original_resolution_v2(
    ctx: Context,
    x1: int, y1: int, x2: int, y2: int,
    min_esp: int,
) -> tuple:
    """
    V3.3.3: 同 _resize_at_original_resolution_v2 但保持 hd 尺寸
    """
    cut_image = IDphotos_cut(x1, y1, x2, y2, ctx.matting_image)
    result_image, _ = move(cut_image.astype(np.uint8))
    h, w = result_image.shape[:2]
    min_border = min(h, w)
    if min_border < min_esp:
        if h >= w:
            new_w = min_esp
            new_h = h * min_esp // w
        else:
            new_h = min_esp
            new_w = w * min_esp // h
        return (
            cv2.resize(result_image, (new_w, new_h), interpolation=cv2.INTER_AREA),
            new_h / h,
        )
    return result_image, 1.0


def _resize_at_original_resolution(
    matting_image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    target_size: tuple,
) -> np.ndarray:
    """
    V3.3.2: 在抠图原图分辨率上直接裁出 std, 避免 cut_image 中间降采样导致的高频丢失.

    流程:
      1. 用 IDphotos_cut 在抠图原图 (1927x1280) 上裁出调整后的区域
      2. 直接 INTER_AREA 一次降到 standard_size (413x295)
      3. 实测: 脸部 Laplacian ratio 从 0.13 → 0.30+
    """
    crop_rgba = IDphotos_cut(x1, y1, x2, y2, matting_image)
    return cv2.resize(crop_rgba, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)


def _resize_hd_at_original_resolution(
    matting_image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    min_esp: int,
) -> tuple:
    """
    V3.3.2: 在抠图原图分辨率上直接裁出 hd, 保留原图高频.
    返回 (hd_rgba, resize_ratio_max)
    """
    crop_rgba = IDphotos_cut(x1, y1, x2, y2, matting_image)
    h, w = crop_rgba.shape[:2]
    min_border = min(h, w)
    if min_border < min_esp:
        if h >= w:
            new_w = min_esp
            new_h = h * min_esp // w
        else:
            new_h = min_esp
            new_w = w * min_esp // h
        return (
            cv2.resize(crop_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA),
            new_h / h,
        )
    return crop_rgba, 1.0


def standard_photo_resize(input_image: np.array, size):
    """
    input_image: 输入图像，即高清照
    size: 标准照的尺寸
    V3.3: 改成一次直接 INTER_AREA 降采样 (旧版多次降采样累计丢 30%+ 高频)
    实测: std 脸部 Laplacian ratio 从 0.13 提升到 ~0.20
    """
    # 直接一次降采样, 用 INTER_AREA (降采样最优)
    if input_image.shape[0] != size[0] or input_image.shape[1] != size[1]:
        result_image = cv2.resize(
            input_image, (size[1], size[0]), interpolation=cv2.INTER_AREA
        )
    else:
        result_image = input_image
    return result_image


def resize_image_by_min(input_image, esp=600):
    """
    将图像缩放为最短边至少为 esp 的图像。
    :param input_image: 输入图像（OpenCV 矩阵）
    :param esp: 缩放后的最短边长
    :return: 缩放后的图像，缩放倍率
    """
    height, width = input_image.shape[0], input_image.shape[1]
    min_border = min(height, width)
    if min_border < esp:
        if height >= width:
            new_width = esp
            new_height = height * esp // width
        else:
            new_height = esp
            new_width = width * esp // height

        return (
            cv2.resize(
                input_image, (new_width, new_height), interpolation=cv2.INTER_AREA
            ),
            new_height / height,
        )

    else:
        return input_image, 1
