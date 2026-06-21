import numpy as np
import cv2
from .context import Context
from .human_matting import get_rmbg_matting_2, WEIGHTS

def get_single_edge_roi(face_bbox, image_shape):
    x, y, w, h = face_bbox
    img_h, img_w = image_shape[:2]

    padding = 0.85
    top_pad = 0.45
    bottom_pad = 1.85

    roi_x = max(0, int(x - w * padding))
    roi_y = max(0, int(y - h * top_pad))
    roi_w = min(img_w - roi_x, int(w * (1 + padding * 2)))
    roi_h = min(img_h - roi_y, int(h * (1 + top_pad + bottom_pad)))

    return roi_x, roi_y, roi_w, roi_h

def refine_edge_alpha(matting_image, face_bbox, upscale_factor=1.8):
    bgr = matting_image[..., :3]
    alpha = matting_image[..., 3].astype(np.float32) / 255.0

    rx, ry, rw, rh = get_single_edge_roi(face_bbox, matting_image.shape)
    if rw < 64 or rh < 64:
        return matting_image

    roi_bgr = bgr[ry:ry+rh, rx:rx+rw]
    roi_alpha = alpha[ry:ry+rh, rx:rx+rw]

    edge_mask = (roi_alpha > 0.02) & (roi_alpha < 0.98)

    upscale_w = min(1024, int(rw * upscale_factor))
    upscale_h = min(1024, int(rh * upscale_factor))
    upscale_image = cv2.resize(
        roi_bgr, (upscale_w, upscale_h), interpolation=cv2.INTER_CUBIC
    )

    result = get_rmbg_matting_2(upscale_image, WEIGHTS["rmbg-2.0"])
    if result is None:
        return matting_image

    refined_roi_alpha = result[..., 3].astype(np.float32) / 255.0
    refined_roi_alpha = cv2.resize(
        refined_roi_alpha, (rw, rh), interpolation=cv2.INTER_CUBIC
    )
    refined_roi_alpha = np.clip(refined_roi_alpha, 0, 1)

    refined_edge_mask = (refined_roi_alpha > 0.02) & (refined_roi_alpha < 0.98)
    blend_region = edge_mask | refined_edge_mask
    if not blend_region.any():
        return matting_image
    blend_region = cv2.dilate(
        blend_region.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
    ).astype(bool)

    y_coords, x_coords = np.meshgrid(np.arange(rh), np.arange(rw), indexing='ij')
    dist_x = np.minimum(x_coords, rw - 1 - x_coords)
    dist_y = np.minimum(y_coords, rh - 1 - y_coords)
    dist_map = np.minimum(dist_x, dist_y) / min(rw, rh) * 2.0
    dist_map = np.clip(dist_map, 0, 1)

    blend_mask = dist_map * 0.7 + 0.1
    blend_mask = np.clip(blend_mask, 0, 0.75)
    blend_mask = blend_mask * blend_region.astype(np.float32)

    refined_alpha = alpha.copy()
    refined_alpha[ry:ry+rh, rx:rx+rw] = (
        refined_alpha[ry:ry+rh, rx:rx+rw] * (1 - blend_mask) +
        refined_roi_alpha * blend_mask
    )
    refined_alpha = np.clip(refined_alpha, 0, 1)

    return cv2.merge((bgr, (refined_alpha * 255).astype(np.uint8)))

def extract_human_multi_scale(ctx: Context):
    from .human_matting import extract_human_rmbg_2
    extract_human_rmbg_2(ctx)

    if ctx.face and ctx.face.get("rectangle") is not None:
        face_bbox = ctx.face["rectangle"]
        ctx.matting_image = refine_edge_alpha(ctx.matting_image, face_bbox)

    ctx.processing_image = ctx.matting_image.copy()
