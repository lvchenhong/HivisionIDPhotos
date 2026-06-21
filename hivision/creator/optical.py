import numpy as np
import cv2


def apply_optical_layer(bg: np.ndarray, noise_scale: float = 0.008) -> np.ndarray:
    """
    添加光学层效果，增加真实感

    原理：
      1. 添加微小高斯噪声（模拟胶片颗粒）
      2. 添加柔和的边缘渐变（模拟镜头暗角）

    Args:
        bg: (H,W,3) uint8, 背景图像
        noise_scale: 噪声强度

    Returns:
        (H,W,3) uint8, 添加光学效果后的背景
    """
    bg_f = bg.astype(np.float32) / 255.0

    noise = np.random.normal(0, noise_scale, bg.shape).astype(np.float32)
    bg_f = bg_f + noise

    gradient = _soft_gradient(bg.shape)
    bg_f = bg_f * gradient

    return np.clip(bg_f * 255.0, 0, 255).astype(np.uint8)


def _soft_gradient(shape):
    """生成柔和的边缘渐变（暗角效果）"""
    h, w = shape[:2]
    cx, cy = w // 2, h // 2

    x = np.linspace(-1, 1, w).astype(np.float32)
    y = np.linspace(-1, 1, h).astype(np.float32)
    xx, yy = np.meshgrid(x, y)

    dist = np.sqrt(xx**2 + yy**2)
    dist = np.clip(dist, 0, 1.5)

    gradient = 1.0 - dist * 0.15
    gradient = np.maximum(gradient, 0.85)

    return gradient[..., None]
