#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from PIL import Image
import io
import numpy as np
import cv2
import base64
from hivision.plugin.watermark import Watermarker, WatermarkerStyles


def save_image_dpi_to_bytes(image: np.ndarray, output_image_path: str = None, dpi: int = 300):
    """
    设置图像的DPI（每英寸点数）并返回字节流
    :param image: numpy.ndarray, 输入的图像数组
    :param output_image_path: Path to save the resized image. 保存调整大小后的文件路径。
    :param dpi: int, 要设置的DPI值，默认为300
    """
    # 按输出路径后缀决定格式；不识别则默认 PNG
    ext = ""
    if output_image_path:
        ext = os.path.splitext(output_image_path)[1].lower().lstrip(".")
    fmt = ext if ext in ("png", "jpg", "jpeg", "webp", "bmp") else "png"
    if fmt == "jpeg":
        fmt = "jpg"

    pil_image = Image.fromarray(image)
    byte_stream = io.BytesIO()

    if fmt == "jpg":
        # JPEG 不支持 alpha, 先合成白底
        if pil_image.mode == "RGBA":
            bg = Image.new("RGB", pil_image.size, (255, 255, 255))
            bg.paste(pil_image, mask=pil_image.split()[3])
            pil_image = bg
        elif pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.save(byte_stream, format="JPEG", dpi=(dpi, dpi), quality=95)
    else:
        pil_image.save(byte_stream, format=fmt.upper(), dpi=(dpi, dpi))

    image_bytes = byte_stream.getvalue()
    if output_image_path:
        with open(output_image_path, "wb") as f:
            f.write(image_bytes)
    return image_bytes


def resize_image_to_kb(input_image: np.ndarray, output_image_path: str = None, target_size_kb: int = 100, dpi: int = 300):
    """
    Resize an image to a target size in KB.
    将图像调整大小至目标文件大小（KB）。

    :param input_image_path: Path to the input image. 输入图像的路径。
    :param output_image_path: Path to save the resized image. 保存调整大小后的图像的路径。
    :param target_size_kb: Target size in KB. 目标文件大小（KB）。

    Example:
    resize_image_to_kb('input_image.jpg', 'output_image.jpg', 50)
    """

    if isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image)
    elif isinstance(input_image, Image.Image):
        img = input_image
    else:
        raise ValueError("input_image must be a NumPy array or PIL Image.")

    # Convert image to RGB mode if it's not
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Initial quality value
    quality = 95

    while True:
        # Create a BytesIO object to hold the image data in memory
        img_byte_arr = io.BytesIO()

        # Save the image to the BytesIO object with the current quality
        img.save(img_byte_arr, format="JPEG", quality=quality, dpi=(dpi, dpi))

        # Get the size of the image in KB
        img_size_kb = len(img_byte_arr.getvalue()) / 1024

        # Check if the image size is within the target size
        if img_size_kb <= target_size_kb or quality == 1:
            # If the image is smaller than the target size, add padding
            if img_size_kb < target_size_kb:
                padding_size = int(
                    (target_size_kb * 1024) - len(img_byte_arr.getvalue())
                )
                padding = b"\x00" * padding_size
                img_byte_arr.write(padding)

            # Save the image to the output path
            if output_image_path:
                with open(output_image_path, "wb") as f:
                    f.write(img_byte_arr.getvalue())
            
            return img_byte_arr.getvalue()

        # Reduce the quality if the image is still too large
        quality -= 5

        # Ensure quality does not go below 1
        if quality < 1:
            quality = 1


def resize_image_to_kb_base64(input_image, target_size_kb, mode="exact"):
    """
    Resize an image to a target size in KB and return it as a base64 encoded string.
    将图像调整大小至目标文件大小（KB）并返回base64编码的字符串。

    :param input_image: Input image as a NumPy array or PIL Image. 输入图像，可以是NumPy数组或PIL图像。
    :param target_size_kb: Target size in KB. 目标文件大小（KB）。
    :param mode: Mode of resizing ('exact', 'max', 'min'). 模式：'exact'（精确大小）、'max'（不大于）、'min'（不小于）。

    :return: Base64 encoded string of the resized image. 调整大小后的图像的base64编码字符串。
    """

    if isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image)
    elif isinstance(input_image, Image.Image):
        img = input_image
    else:
        raise ValueError("input_image must be a NumPy array or PIL Image.")

    # Convert image to RGB mode if it's not
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Initial quality value
    quality = 95

    while True:
        # Create a BytesIO object to hold the image data in memory
        img_byte_arr = io.BytesIO()

        # Save the image to the BytesIO object with the current quality
        img.save(img_byte_arr, format="JPEG", quality=quality)

        # Get the size of the image in KB
        img_size_kb = len(img_byte_arr.getvalue()) / 1024

        # Check based on the mode
        if mode == "exact":
            # If the image size is equal to the target size, we can return it
            if img_size_kb == target_size_kb:
                break

            # If the image is smaller than the target size, add padding
            elif img_size_kb < target_size_kb:
                padding_size = int(
                    (target_size_kb * 1024) - len(img_byte_arr.getvalue())
                )
                padding = b"\x00" * padding_size
                img_byte_arr.write(padding)
                break

        elif mode == "max":
            # If the image size is within the target size, we can return it
            if img_size_kb <= target_size_kb or quality == 1:
                break

        elif mode == "min":
            # If the image size is greater than or equal to the target size, we can return it
            if img_size_kb >= target_size_kb:
                break

        # Reduce the quality if the image is still too large
        quality -= 5

        # Ensure quality does not go below 1
        if quality < 1:
            quality = 1

    # Encode the image data to base64
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    return "data:image/png;base64," + img_base64


def numpy_2_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", img)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    return "data:image/png;base64," + base64_image


def base64_2_numpy(base64_image: str) -> np.ndarray:
    # Remove the data URL prefix if present
    if base64_image.startswith('data:image'):
        base64_image = base64_image.split(',')[1]
    
    # Decode base64 string to bytes
    img_bytes = base64.b64decode(base64_image)
    
    # Convert bytes to numpy array
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    
    # Decode the image array
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    
    return img

# 字节流转base64
def bytes_2_base64(img_byte_arr: bytes) -> str:
    base64_image = base64.b64encode(img_byte_arr).decode("utf-8")
    return "data:image/png;base64," + base64_image


def save_numpy_image(numpy_img, file_path):
    # 检查数组的形状
    if numpy_img.shape[2] == 4:
        # 将 BGR 转换为 RGB，并保留透明通道
        rgb_img = np.concatenate(
            (np.flip(numpy_img[:, :, :3], axis=-1), numpy_img[:, :, 3:]), axis=-1
        ).astype(np.uint8)
        img = Image.fromarray(rgb_img, mode="RGBA")
    else:
        # 将 BGR 转换为 RGB
        rgb_img = np.flip(numpy_img, axis=-1).astype(np.uint8)
        img = Image.fromarray(rgb_img, mode="RGB")

    img.save(file_path)


def numpy_to_bytes(numpy_img):
    img = Image.fromarray(numpy_img)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


def hex_to_rgb(value):
    value = value.lstrip("#")
    length = len(value)
    return tuple(
        int(value[i : i + length // 3], 16) for i in range(0, length, length // 3)
    )


def generate_gradient(start_color, width, height, mode="updown"):
    # 定义背景颜色
    end_color = (255, 255, 255)  # 白色

    # 创建一个空白图像
    r_out = np.zeros((height, width), dtype=int)
    g_out = np.zeros((height, width), dtype=int)
    b_out = np.zeros((height, width), dtype=int)

    if mode == "updown":
        # 生成上下渐变色
        for y in range(height):
            r = int(
                (y / height) * end_color[0] + ((height - y) / height) * start_color[0]
            )
            g = int(
                (y / height) * end_color[1] + ((height - y) / height) * start_color[1]
            )
            b = int(
                (y / height) * end_color[2] + ((height - y) / height) * start_color[2]
            )
            r_out[y, :] = r
            g_out[y, :] = g
            b_out[y, :] = b

    else:
        # 生成中心渐变色
        img = np.zeros((height, width, 3))
        # 定义椭圆中心和半径
        center = (width // 2, height // 2)
        end_axies = max(height, width)
        # 定义渐变色
        end_color = (255, 255, 255)
        # 绘制椭圆
        for y in range(end_axies):
            axes = (end_axies - y, end_axies - y)
            r = int(
                (y / end_axies) * end_color[0]
                + ((end_axies - y) / end_axies) * start_color[0]
            )
            g = int(
                (y / end_axies) * end_color[1]
                + ((end_axies - y) / end_axies) * start_color[1]
            )
            b = int(
                (y / end_axies) * end_color[2]
                + ((end_axies - y) / end_axies) * start_color[2]
            )

            cv2.ellipse(img, center, axes, 0, 0, 360, (b, g, r), -1)
        b_out, g_out, r_out = cv2.split(np.uint64(img))

    return r_out, g_out, b_out


def add_background(input_image, bgr=(0, 0, 0), mode="pure_color"):
    """
    本函数的功能为为透明图像加上背景 (走商业级 pipeline).
    规格:
      - rmbg-2.0 + retinaface
      - low refine (一次双边滤波)
      - hard alpha composite
      - feather 0.4px
      - halo suppression ON
      - 纯色背景, no blur fusion
      - light correction only: brightness +0.02, contrast +0.07, sharpen 0.6
      - 禁美颜, 禁 AI 增强
    """
    if input_image.ndim != 3 or input_image.shape[2] != 4:
        raise ValueError("The input image must have 4 channels.")

    height, width = input_image.shape[:2]

    # 1) 生成背景图 (BGR)
    if mode == "pure_color":
        bg = np.zeros((height, width, 3), dtype=np.float32)
        bg[..., 0] = bgr[0]
        bg[..., 1] = bgr[1]
        bg[..., 2] = bgr[2]
    elif mode == "updown_gradient":
        bg_b, bg_g, bg_r = generate_gradient(bgr, width, height, mode="updown")
        bg = cv2.merge([bg_b, bg_g, bg_r]).astype(np.float32)
    else:
        bg_b, bg_g, bg_r = generate_gradient(bgr, width, height, mode="center")
        bg = cv2.merge([bg_b, bg_g, bg_r]).astype(np.float32)

    # 2) 关键修复: 在送 refiner 之前, 先把 α=0 区域的 RGB 强制设为背景色
    # (Trae 旧 IDphotos_cut 阶段用 RGB=255 白底补位, 不处理会导致软边发白)
    rgba_clean = input_image.copy()
    alpha0_mask = rgba_clean[..., 3] == 0
    if alpha0_mask.any():
        rgba_clean[alpha0_mask, 0] = bgr[0]
        rgba_clean[alpha0_mask, 1] = bgr[1]
        rgba_clean[alpha0_mask, 2] = bgr[2]

    # 3) 走 idphoto_v4_photographic 摄影级合成
    from hivision.creator.matting_refiner import idphoto_v4_photographic
    bg_bgr_for_pipeline = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    output_bgr = idphoto_v4_photographic(rgba_clean, bg_bgr=bg_bgr_for_pipeline)

    # 4) 兜底: 合成后再硬覆盖 α=0 区域 (因为 refiner 内部可能因浮点误差回归)
    alpha = rgba_clean[..., 3].astype(np.float32) / 255.0
    if alpha0_mask.any():
        a3 = alpha[..., None]
        out = output_bgr.astype(np.float32) * a3 + bg * (1.0 - a3)
        output_bgr = np.clip(out, 0, 255).astype(np.uint8)

    # 5) 如果是渐变模式, 二次合成 (商业级只对纯色做校色)
    if mode != "pure_color":
        a3 = alpha[..., None]
        out = output_bgr.astype(np.float32) * a3 + bg * (1.0 - a3)
        output_bgr = np.clip(out, 0, 255).astype(np.uint8)

    # 6) HD-Commercial Balance Stage（淘宝/拼多多最优成交版本）
    from hivision.creator.hd_enhance import hd_commercial_balance
    output_bgr = hd_commercial_balance(output_bgr, face_mask=None, upscale=1.7)

    return output_bgr



def add_background_with_image(input_image: np.ndarray, background_image: np.ndarray) -> np.ndarray:
    """
    本函数的功能为为透明图像加上背景。
    :param input_image: numpy.array(4 channels), 透明图像
    :param background_image: numpy.array(3 channels), 背景图像
    :return: output: 合成好的输出图像
    """
    height, width = input_image.shape[:2]
    try:
        b, g, r, a = cv2.split(input_image)
    except ValueError:
        raise ValueError(
            "The input image must have 4 channels. 输入图像必须有4个通道，即透明图像。"
        )

    # 确保背景图像与输入图像大小一致
    background_image = cv2.resize(background_image, (width, height), cv2.INTER_AREA)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    b2, g2, r2 = cv2.split(background_image)

    a_cal = a / 255.0

    # 修正混合公式
    output = cv2.merge(
        (b * a_cal + b2 * (1 - a_cal),
         g * a_cal + g2 * (1 - a_cal),
         r * a_cal + r2 * (1 - a_cal))
    )

    return output.astype(np.uint8)

def add_watermark(
    image, text, size=50, opacity=0.5, angle=45, color="#8B8B1B", space=75
):
    image = Image.fromarray(image)
    watermarker = Watermarker(
        input_image=image,
        text=text,
        style=WatermarkerStyles.STRIPED,
        angle=angle,
        color=color,
        opacity=opacity,
        size=size,
        space=space,
    )
    return np.array(watermarker.image.convert("RGB"))
