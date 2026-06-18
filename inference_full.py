"""
HivisionIDPhotos 完整 CLI 推理脚本 (扩展 inference.py)
- 支持 rmbg-2.0
- 支持全部 6 项美颜参数
- 支持 3 项构图参数
- 支持 face_alignment、纯色背景、DPI

使用示例:
    python inference_full.py \
        -i demo/images/test0.jpg \
        -o out.jpg \
        --matting_model rmbg-2.0 \
        --face_detect_model retinaface-resnet50 \
        --width 295 --height 413 \
        -c ffffff \
        --face_align true \
        --whitening_strength 1.1 \
        --brightness_strength 1.4 \
        --contrast_strength 1.5 \
        --saturation_strength -0.4 \
        --sharpen_strength 1.3 \
        --head_measure_ratio 0.23 \
        --top_distance_max 0.09 \
        --dpi 300
"""
import os
import sys
import cv2
import argparse
import numpy as np

# 抑制 onnxruntime / cv2 的一些无伤大雅 warning
os.environ.setdefault("PYTHONWARNINGS", "ignore")

from hivision.error import FaceError
from hivision.utils import hex_to_rgb, save_image_dpi_to_bytes
from hivision import IDCreator
from hivision.creator.choose_handler import choose_handler


MATTING_MODEL = [
    "hivision_modnet",
    "modnet_photographic_portrait_matting",
    "mnn_hivision_modnet",
    "rmbg-1.4",
    "rmbg-2.0",
    "birefnet-v1-lite",
]
FACE_DETECT_MODEL = ["mtcnn", "face_plusplus", "retinaface-resnet50"]


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def main():
    p = argparse.ArgumentParser(
        description="HivisionIDPhotos 完整参数版 (含美颜 + 构图 + rmbg-2.0)"
    )
    p.add_argument("-i", "--input_image_dir", required=True, help="输入图像路径")
    p.add_argument("-o", "--output_image_dir", required=True, help="输出图像路径")
    # 尺寸
    p.add_argument("--height", type=int, default=413, help="证件照高 (一寸 413)")
    p.add_argument("--width", type=int, default=295, help="证件照宽 (一寸 295)")
    # 背景
    p.add_argument("-c", "--color", default="ffffff", help="背景色 HEX, 默认 ffffff 白底")
    p.add_argument(
        "-r", "--render", type=int, default=0, choices=[0, 1, 2],
        help="背景渲染: 0 纯色 1 上下渐变 2 中心渐变",
    )
    p.add_argument("--dpi", type=int, default=300, help="输出 DPI")
    # 模型
    p.add_argument(
        "--matting_model", default="rmbg-2.0", choices=MATTING_MODEL,
    )
    p.add_argument(
        "--face_detect_model", default="retinaface-resnet50", choices=FACE_DETECT_MODEL,
    )
    # 构图
    p.add_argument("--face_align", type=str2bool, default=True)
    p.add_argument(
        "--head_measure_ratio", type=float, default=0.23,
        help="面部占画面比例 (默认 0.2, 用户设 0.23)",
    )
    p.add_argument(
        "--top_distance_max", type=float, default=0.09,
        help="头顶到画布上沿的最大比例 (默认 0.12, 用户设 0.09)",
    )
    p.add_argument(
        "--head_height_ratio", type=float, default=0.45,
        help="人脸中心在画面的垂直比例 (默认 0.45)",
    )
    # 美颜
    p.add_argument("--whitening_strength", type=float, default=1.1)
    p.add_argument("--brightness_strength", type=float, default=1.4)
    p.add_argument("--contrast_strength", type=float, default=1.5)
    p.add_argument("--saturation_strength", type=float, default=-0.4)
    p.add_argument("--sharpen_strength", type=float, default=1.3)
    p.add_argument(
        "--no_hd", action="store_true",
        help="只输出标准照, 不输出 _hd 大图",
    )

    args = p.parse_args()

    # 准备模型
    creator = IDCreator()
    choose_handler(creator, args.matting_model, args.face_detect_model)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    img = cv2.imread(args.input_image_dir, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[ERROR] 无法读取图像: {args.input_image_dir}")
        sys.exit(1)
    print(f"[INFO] 输入图像: {args.input_image_dir}, shape={img.shape}")

    size = (int(args.height), int(args.width))
    # head_top_range: (max, min), min = max - 0.02  (对齐 processor.py 的做法)
    head_top_range = (args.top_distance_max, max(0.0, args.top_distance_max - 0.02))

    # 背景色 RGB -> BGR
    rgb = hex_to_rgb(args.color)
    bgr = (rgb[2], rgb[1], rgb[0])

    try:
        result = creator(
            img,
            size=size,
            change_bg_only=False,
            head_measure_ratio=args.head_measure_ratio,
            head_height_ratio=args.head_height_ratio,
            head_top_range=head_top_range,
            whitening_strength=args.whitening_strength,
            brightness_strength=args.brightness_strength,
            contrast_strength=args.contrast_strength,
            saturation_strength=args.saturation_strength,
            sharpen_strength=args.sharpen_strength,
            face_alignment=args.face_align,
        )
    except FaceError:
        print("[ERROR] 人脸数量不等于 1, 请上传单张人脸的图像。")
        sys.exit(2)
    except Exception as e:
        print(f"[ERROR] 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)

    # result.standard 是 RGBA 抠好的图, 背景由 IDCreator 内部上好
    # result.hd 是高分辨率版本
    out_dir = os.path.dirname(os.path.abspath(args.output_image_dir)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 走完整 matting_refiner + add_background pipeline
    from hivision.utils import add_background as add_bg_func
    bgr_tuple = bgr  # 复用上面算好的 bgr

    # 走 refiner 合成
    result_std_bgr = add_bg_func(result.standard, bgr=bgr_tuple, mode="pure_color")
    cv2.imwrite(args.output_image_dir, result_std_bgr)
    print(f"[OK] 标准照已保存: {args.output_image_dir}")

    if not args.no_hd:
        base, ext = os.path.splitext(args.output_image_dir)
        hd_path = f"{base}_hd{ext}"
        result_hd_bgr = add_bg_func(result.hd, bgr=bgr_tuple, mode="pure_color")
        cv2.imwrite(hd_path, result_hd_bgr)
        print(f"[OK] 高清照已保存: {hd_path}")


if __name__ == "__main__":
    main()
