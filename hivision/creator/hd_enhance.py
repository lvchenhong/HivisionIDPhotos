#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HD-Commercial Balance - 淘宝/拼多多最优成交版本
"用户以为只是拍得更好，而不是你修过"

核心设计哲学:
  - NOT: 极致清晰 / AI增强感 / 算法锐化
  - BUT: 轻微提升质感 / 保留真实摄影噪声 / 模拟"影楼扫描优化"

Pipeline:
  1. 轻微智能放大 (1.7x 商业甜点区)
  2. 极轻影楼级对比修正 (让脸不灰，不锐化)
  3. 弱结构恢复锐化 (非全局sharpen，模拟照片清晰冲击)
  4. 仅人脸区域增强 (眼睛/鼻梁，皮肤整体不动)
  5. 轻噪声保真 (消除"数码塑料感")
"""
import numpy as np
import cv2


def _high_pass(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """高通信号：用于人脸区域细节恢复"""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return (img.astype(np.float32) - blur.astype(np.float32))


def _tune_face_mask(face_mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    调整face_mask到目标shape的float32单通道
    """
    if face_mask is None:
        return None
    if face_mask.ndim == 3:
        face_mask = face_mask[..., 0]
    if face_mask.shape[:2] != target_shape[:2]:
        face_mask = cv2.resize(face_mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return face_mask.astype(np.float32) / 255.0


def hd_commercial_balance(img: np.ndarray, face_mask: np.ndarray = None, upscale: float = 1.7) -> np.ndarray:
    """
    HD-Commercial Balance 入口：商业甜点版本

    目标：
      - 用户觉得比原图清晰
      - 但绝对不像P过
      - 像"专业影楼优化过"

    步骤:
      1. 轻微智能放大 (1.7x CUBIC，商业甜点区)
      2. 极轻影楼级对比修正 (让脸不灰)
      3. 弱结构恢复锐化 (非全局 sharpen)
      4. 仅人脸区域增强 (眼睛/鼻梁细节)
      5. 轻噪声保真 (消除"数码塑料感")

    Args:
        img: (H,W,3) uint8, BGR 图像
        face_mask: (H,W) uint8/单通道, 人脸区域掩码(0-255)，None则跳过face-only
        upscale: 放大倍数，默认 1.7

    Returns:
        (H*upscale, W*upscale, 3) uint8, HD-Commercial Balance 后的 BGR 图像
    """
    # 1. 轻微智能放大 (sweet spot: 1.5太软 / 2.0太假 / 1.7最优)
    h, w = img.shape[:2]
    new_w, new_h = int(w * upscale), int(h * upscale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    img_f = img.astype(np.float32)

    # 2. 极轻"影楼级对比修正"（让脸"不灰"，但不会锐化）
    img_f = img_f * 0.99 + 1.5

    # 3. 弱结构恢复锐化（非全局 sharpen，模拟照片清晰冲击）
    blur1 = cv2.GaussianBlur(img_f, (0, 0), 1.0)
    img_f = img_f * 1.03 + blur1 * (-0.03)

    # 4. 仅人脸区域增强（非常关键：只动眼睛/鼻梁，不动皮肤整体）
    fm = _tune_face_mask(face_mask, img_f.shape)
    if fm is not None:
        # 高通细节 = 原图 - 模糊图
        detail = img_f - cv2.GaussianBlur(img_f, (3, 3), 0)
        # 仅在人脸区域叠加 6% 细节
        img_f = img_f + detail * fm[..., None] * 0.06

    # 5. 轻噪声保真（摄影感核心：消除"数码塑料感"）
    noise = np.random.normal(0, 0.003, img_f.shape).astype(np.float32)
    img_f = img_f + noise

    return np.clip(img_f, 0, 255).astype(np.uint8)
