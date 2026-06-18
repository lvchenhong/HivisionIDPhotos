#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alpha matting refinement & foreground decontamination pipeline.
"""
import numpy as np
import cv2
from typing import Tuple


def alpha_refine(rgb: np.ndarray, alpha: np.ndarray, radius: int = 2, eps: float = 1e-3) -> np.ndarray:
    a = alpha.astype(np.float32) / 255.0
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.shape[2] == 3 else rgb[..., 0]
    else:
        gray = rgb
    gray_f = gray.astype(np.float32) / 255.0
    a_uint8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    refined = cv2.bilateralFilter(a_uint8, d=radius * 2 + 1, sigmaColor=max(eps * 255, 10), sigmaSpace=radius)
    refined = cv2.bilateralFilter(refined, d=3, sigmaColor=10, sigmaSpace=1)
    return refined


def decontaminate(rgb: np.ndarray, alpha: np.ndarray, bg_bgr: Tuple[int, int, int],
                  threshold_lo: float = 0.02, threshold_hi: float = 0.98) -> np.ndarray:
    a = alpha.astype(np.float32) / 255.0
    b, g, r = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    bg_b, bg_g, bg_r = float(bg_bgr[0]), float(bg_bgr[1]), float(bg_bgr[2])
    trans = (a > threshold_lo) & (a < threshold_hi)
    safe_a = np.where(a > 0.01, a, 0.01)
    
    fg_b = np.where(trans, (b - bg_b * (1.0 - a)) / safe_a, b)
    fg_g = np.where(trans, (g - bg_g * (1.0 - a)) / safe_a, g)
    fg_r = np.where(trans, (r - bg_r * (1.0 - a)) / safe_a, r)
    
    fg_b = np.clip(fg_b, 0, 240)
    fg_g = np.clip(fg_g, 0, 240)
    fg_r = np.clip(fg_r, 0, 240)
    
    out = rgb.copy()
    out[..., 0] = fg_b.astype(np.uint8)
    out[..., 1] = fg_g.astype(np.uint8)
    out[..., 2] = fg_r.astype(np.uint8)
    return out


def feather(alpha: np.ndarray, radius: int = 1) -> np.ndarray:
    if alpha.dtype != np.float32:
        a = alpha.astype(np.float32) / 255.0
    else:
        a = alpha / 255.0
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(a, (ksize, ksize), sigmaX=max(radius * 0.6, 0.1))
    blurred_uint8 = (np.clip(blurred, 0, 1) * 255).astype(np.uint8)
    return blurred_uint8


def soft_sharpen(rgb: np.ndarray, alpha: np.ndarray, strength: float = 0.5,
                 threshold: float = 0.95) -> np.ndarray:
    if strength <= 0:
        return rgb
    kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5.0, -0.5], [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(rgb, -1, kernel * strength + np.eye(3) * (1 - strength))
    mask = (alpha.astype(np.float32) / 255.0) > threshold
    mask3 = np.repeat(mask[..., None], 3, axis=2)
    out = np.where(mask3, sharpened, rgb)
    return out.astype(np.uint8)


def suppress_bg_spill(rgb: np.ndarray, alpha: np.ndarray, bg_bgr: Tuple[int, int, int],
                      threshold: float = 0.02) -> np.ndarray:
    out = rgb.copy()
    bg_mask = (alpha.astype(np.float32) / 255.0) <= threshold
    bg_mask3 = np.repeat(bg_mask[..., None], 3, axis=2)
    bg_color = np.array(bg_bgr, dtype=np.uint8)
    out = np.where(bg_mask3, bg_color, out)
    return out


def compose_with_bg(rgb: np.ndarray, alpha: np.ndarray, bg_bgr: Tuple[int, int, int]) -> np.ndarray:
    a = alpha.astype(np.float32) / 255.0
    a3 = a[..., None]
    fg = rgb.astype(np.float32)
    bg = np.full_like(fg, 0.0)
    bg[..., 0] = bg_bgr[0]
    bg[..., 1] = bg_bgr[1]
    bg[..., 2] = bg_bgr[2]
    out = fg * a3 + bg * (1.0 - a3)
    return np.clip(out, 0, 255).astype(np.uint8)


def desaturate_blue_bg(rgb: np.ndarray, alpha: np.ndarray, bg_bgr: Tuple[int, int, int],
                       threshold: float = 0.5) -> np.ndarray:
    is_blue_bg = (bg_bgr[0] > bg_bgr[1]) and (bg_bgr[0] > bg_bgr[2])
    if not is_blue_bg:
        return rgb
    a = alpha.astype(np.float32) / 255.0
    b, g, r = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    edge = (a > 0.2) & (a < 0.85)
    blue_heavy = (b - r) > 30
    mask = edge & blue_heavy
    reduce = 0.3
    new_b = np.where(mask, b * (1 - reduce) + g * reduce, b)
    new_b = np.clip(new_b, 0, 255)
    out = rgb.copy()
    out[..., 0] = new_b.astype(np.uint8)
    return out


def refine_and_compose(rgba: np.ndarray, bg_bgr: Tuple[int, int, int],
                       feather_radius: int = 0, sharpen_strength: float = 0.0) -> np.ndarray:
    assert rgba.ndim == 3 and rgba.shape[2] == 4, "需要 RGBA 4 通道输入"
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    out = compose_with_bg(rgb, alpha, bg_bgr)
    return out
