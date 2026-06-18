import cv2
from hivision.creator.context import Context
from hivision.plugin.beauty.whitening import make_whitening
from hivision.plugin.beauty.base_adjust import (
    adjust_brightness_contrast_sharpen_saturation,
)


def beauty_face(ctx: Context):
    """
    对人脸进行美颜处理
    1. 美白
    2. 亮度

    :param ctx: Context对象，包含处理参数和图像
    """
    # 修复: 用 matting_image 的 RGB (已抠图) 而不是 origin_image
    # 这样美颜处理后的 RGB 与最终输出的边缘像素一致, 避免头发边缘出现色差
    # matting_image 是 RGBA, 取前 3 通道
    if ctx.matting_image.shape[2] == 4:
        middle_image = ctx.matting_image[..., :3].copy()
    else:
        middle_image = ctx.matting_image.copy()
    processed = False

    # 如果美白强度大于0，进行美白处理
    if ctx.params.whitening_strength > 0:
        middle_image = make_whitening(middle_image, ctx.params.whitening_strength)
        processed = True

    # 如果亮度、对比度、锐化强度不为0，进行亮度、对比度、锐化处理
    if (
        ctx.params.brightness_strength != 0
        or ctx.params.contrast_strength != 0
        or ctx.params.sharpen_strength != 0
        or ctx.params.saturation_strength != 0
    ):
        middle_image = adjust_brightness_contrast_sharpen_saturation(
            middle_image,
            ctx.params.brightness_strength,
            ctx.params.contrast_strength,
            ctx.params.sharpen_strength,
            ctx.params.saturation_strength,
        )
        processed = True

    # 如果进行了美颜处理，更新matting_image
    if processed:
        # 修复: 保持原有的 alpha 通道不变, 只替换 RGB
        b, g, r = cv2.split(middle_image)
        _, _, _, alpha = cv2.split(ctx.matting_image)
        ctx.matting_image = cv2.merge((b, g, r, alpha))
