import cv2
from hivision.creator.context import Context


def beauty_face(ctx: Context):
    """
    商业级证件照: 禁美颜, 禁 AI 增强.
    校色统一由 add_background() 里的 light_correction() 接管.
    这里什么都不做, 保留抠图原图.
    """
    return
