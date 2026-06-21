import numpy as np
import cv2


def apply_hair_stochastic(alpha: np.ndarray) -> np.ndarray:
    """
    对头发边缘添加随机性，打破AI切割的整齐感

    原理：
      1. 提取边缘区域（alpha 在 0.1-0.9 之间的软边）
      2. 在边缘区域叠加 Perlin 噪声和高斯噪声
      3. 仅对头发区域（高频率边缘）应用更强的随机性

    Args:
        alpha: (H,W) uint8, 0-255

    Returns:
        (H,W) uint8, 0-255
    """
    alpha_f = alpha.astype(np.float32) / 255.0

    edge_mask = ((alpha_f > 0.1) & (alpha_f < 0.9)).astype(np.float32)
    if not edge_mask.any():
        return alpha

    kernel = np.ones((5, 5), np.float32) / 25.0
    edge_blur = cv2.filter2D(edge_mask, -1, kernel)
    edge_region = edge_blur > 0.05

    noise = _perlin_noise(alpha.shape, scale=8)
    jitter = np.random.normal(0, 0.8, alpha.shape).astype(np.float32)

    laplacian = cv2.Laplacian(alpha_f, cv2.CV_32F, ksize=3)
    hair_region = (np.abs(laplacian) > 0.1) & edge_region

    alpha_f[hair_region] += noise[hair_region] * 0.05
    alpha_f[hair_region] += jitter[hair_region] * 0.02

    alpha_f[edge_region] += noise[edge_region] * 0.02

    return np.clip(alpha_f * 255.0, 0, 255).astype(np.uint8)


def _perlin_noise(shape, scale=8):
    """生成简单的 Perlin 噪声近似"""
    h, w = shape[:2]
    octaves = 3
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0

    for _ in range(octaves):
        nh, nw = int(h / (scale * frequency)), int(w / (scale * frequency))
        if nh < 2 or nw < 2:
            break

        base_noise = np.random.rand(nh + 1, nw + 1).astype(np.float32)
        base_noise = cv2.resize(base_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise += base_noise * amplitude
        amplitude *= 0.5
        frequency *= 2.0

    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise) + 1e-6)
    return noise * 2.0 - 1.0
