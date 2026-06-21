import numpy as np
import cv2


def preserve_face_texture(fg: np.ndarray, alpha: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """
    保留人脸微纹理，防止过度平滑

    原理：
      1. 提取高频细节（边缘、纹理）
      2. 将高频细节叠加回原图，增强真实感

    Args:
        fg: (H,W,3) uint8, 前景图像
        alpha: (H,W) uint8, alpha 通道
        strength: 纹理增强强度

    Returns:
        (H,W,3) uint8, 纹理增强后的前景
    """
    fg_f = fg.astype(np.float32)

    blurred = cv2.GaussianBlur(fg_f, (0, 0), 2.0)
    detail = fg_f - blurred

    fg_f = fg_f + detail * strength

    mask = (alpha > 200).astype(np.float32)[..., None]
    result = fg_f * mask + fg.astype(np.float32) * (1.0 - mask)

    return np.clip(result, 0, 255).astype(np.uint8)
