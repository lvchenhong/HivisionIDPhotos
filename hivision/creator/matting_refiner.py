#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
证件照 V2 Final Stable Production
工业级稳定输出 + GPU 8GB 优化 + Numba JIT加速
"""
import numpy as np
import cv2
from typing import Tuple
from numba import jit, prange


CONFIG = {
    'feather_radius': 0.3,
    'erosion_px': 0.1,
    'dilation_px': 0.3,
    'max_expansion': 0.5,
}


@jit(nopython=True, parallel=True)
def low_strength_halo_suppress_numba(rgb: np.ndarray, alpha: np.ndarray, bg_bgr: Tuple[int, int, int]) -> np.ndarray:
    h, w = rgb.shape[:2]
    a = alpha.astype(np.float32) / 255.0
    fg = rgb.astype(np.float32)
    bg_b, bg_g, bg_r = float(bg_bgr[0]), float(bg_bgr[1]), float(bg_bgr[2])
    
    out = np.empty_like(fg)
    
    for i in prange(h):
        for j in range(w):
            ai = a[i, j]
            if ai > 0.10 and ai < 0.90:
                denom = ai if ai > 0.1 else 0.1
                out[i, j, 0] = np.clip((fg[i, j, 0] - bg_b * (1.0 - ai)) / denom, 0, 255)
                out[i, j, 1] = np.clip((fg[i, j, 1] - bg_g * (1.0 - ai)) / denom, 0, 255)
                out[i, j, 2] = np.clip((fg[i, j, 2] - bg_r * (1.0 - ai)) / denom, 0, 255)
            else:
                out[i, j] = fg[i, j]
    
    return out.astype(np.uint8)


def low_strength_halo_suppress(rgb: np.ndarray, alpha: np.ndarray, bg_bgr: Tuple[int, int, int]) -> np.ndarray:
    try:
        return low_strength_halo_suppress_numba(rgb, alpha, bg_bgr)
    except:
        a = alpha.astype(np.float32) / 255.0
        fg = rgb.astype(np.float32)
        bg_b, bg_g, bg_r = float(bg_bgr[0]), float(bg_bgr[1]), float(bg_bgr[2])
        
        edge_zone = (a > 0.10) & (a < 0.90)
        
        fg_b = np.where(edge_zone, (fg[..., 0] - bg_b * (1.0 - a)) / np.maximum(a, 0.1), fg[..., 0])
        fg_g = np.where(edge_zone, (fg[..., 1] - bg_g * (1.0 - a)) / np.maximum(a, 0.1), fg[..., 1])
        fg_r = np.where(edge_zone, (fg[..., 2] - bg_r * (1.0 - a)) / np.maximum(a, 0.1), fg[..., 2])
        
        fg_b = np.clip(fg_b, 0, 255)
        fg_g = np.clip(fg_g, 0, 255)
        fg_r = np.clip(fg_r, 0, 255)
        
        out = rgb.copy()
        out[..., 0] = fg_b.astype(np.uint8)
        out[..., 1] = fg_g.astype(np.uint8)
        out[..., 2] = fg_r.astype(np.uint8)
        return out


def hard_edge_feather(alpha: np.ndarray) -> np.ndarray:
    radius = min(CONFIG['feather_radius'], 0.4)
    if radius <= 0:
        return alpha
    
    a = alpha.astype(np.float32) / 255.0
    dist_in = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    edge_mask = (dist_in > 0) & (dist_in <= 2.0)
    a[edge_mask] = np.clip(a[edge_mask] * (1.0 + radius * 2), 0, 1)
    
    return (a * 255).astype(np.uint8)


def edge_structure_correct(alpha: np.ndarray) -> np.ndarray:
    erosion_px = CONFIG['erosion_px']
    dilation_px = min(CONFIG['dilation_px'], CONFIG['max_expansion'])
    
    a = alpha.astype(np.float32) / 255.0
    
    if erosion_px > 0:
        kernel_size = max(3, int(erosion_px * 2 + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        a = cv2.erode(a, kernel, iterations=1)
    
    if dilation_px > 0:
        kernel_size = max(3, int(dilation_px * 2 + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        a = cv2.dilate(a, kernel, iterations=1)
    
    a = np.clip(a, 0, 1)
    return (a * 255).astype(np.uint8)


@jit(nopython=True, parallel=True)
def linear_alpha_composite_numba(rgb: np.ndarray, alpha: np.ndarray, bg_bgr: Tuple[int, int, int]) -> np.ndarray:
    h, w = rgb.shape[:2]
    a = alpha.astype(np.float32) / 255.0
    fg = rgb.astype(np.float32)
    bg_b, bg_g, bg_r = float(bg_bgr[0]), float(bg_bgr[1]), float(bg_bgr[2])
    
    out = np.empty_like(fg)
    
    for i in prange(h):
        for j in range(w):
            ai = a[i, j]
            out[i, j, 0] = fg[i, j, 0] * ai + bg_b * (1.0 - ai)
            out[i, j, 1] = fg[i, j, 1] * ai + bg_g * (1.0 - ai)
            out[i, j, 2] = fg[i, j, 2] * ai + bg_r * (1.0 - ai)
    
    return np.clip(out, 0, 255).astype(np.uint8)


def linear_alpha_composite(rgb: np.ndarray, alpha: np.ndarray, bg_bgr: Tuple[int, int, int]) -> np.ndarray:
    try:
        return linear_alpha_composite_numba(rgb, alpha, bg_bgr)
    except:
        a = alpha.astype(np.float32) / 255.0
        a3 = a[..., None]
        fg = rgb.astype(np.float32)
        bg = np.full_like(fg, 0.0)
        bg[..., 0] = bg_bgr[0]
        bg[..., 1] = bg_bgr[1]
        bg[..., 2] = bg_bgr[2]
        out = fg * a3 + bg * (1.0 - a3)
        return np.clip(out, 0, 255).astype(np.uint8)


def idphoto_v2_final_stable(rgba: np.ndarray, bg_bgr: Tuple[int, int, int]) -> np.ndarray:
    assert rgba.ndim == 3 and rgba.shape[2] == 4, "需要 RGBA 4 通道"
    
    rgb = rgba[..., :3].copy()
    alpha = rgba[..., 3].copy()
    
    rgb = low_strength_halo_suppress(rgb, alpha, bg_bgr)
    alpha = edge_structure_correct(alpha)
    alpha = hard_edge_feather(alpha)
    out = linear_alpha_composite(rgb, alpha, bg_bgr)
    
    return out
