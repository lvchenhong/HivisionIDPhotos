import os
import cv2
import numpy as np
from PIL import Image
from hivision.creator.human_matting import (
    get_rmbg_matting,
    get_rmbg_matting_2,
    get_modnet_matting,
    get_modnet_matting_photographic_portrait_matting,
    get_birefnet_portrait_matting,
    WEIGHTS,
)


def load_test_image():
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    for f in os.listdir("."):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(f)
            if img is not None:
                print(f"Found test image: {f}")
                return img
    
    print("No test image found in current directory, using sample image")
    img = np.zeros((600, 400, 3), dtype=np.uint8)
    cv2.putText(img, "NO IMAGE", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return img


def save_mask_comparison(masks, model_names, output_dir="matting_comparison"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for mask, name in zip(masks, model_names):
        if mask is None:
            print(f"  {name}: FAILED")
            continue
        
        alpha = mask[..., 3] if mask.ndim == 3 and mask.shape[2] == 4 else mask
        
        mask_visual = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        mask_visual = cv2.resize(mask_visual, (400, 500))
        
        cv2.putText(mask_visual, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        output_path = os.path.join(output_dir, f"{name}_mask.png")
        cv2.imwrite(output_path, mask_visual)
        print(f"  {name}: OK -> {output_path}")


def main():
    print("=" * 60)
    print("A/B测试：不同Matting模型对比")
    print("=" * 60)
    
    img = load_test_image()
    print(f"Image shape: {img.shape}")
    
    models = [
        ("rmbg-1.4", get_rmbg_matting, WEIGHTS["rmbg-1.4"]),
        ("rmbg-2.0", get_rmbg_matting_2, WEIGHTS["rmbg-2.0"]),
        ("modnet_photographic", get_modnet_matting_photographic_portrait_matting, WEIGHTS["modnet_photographic_portrait_matting"]),
        ("birefnet-v1-lite", get_birefnet_portrait_matting, WEIGHTS["birefnet-v1-lite"]),
    ]
    
    masks = []
    model_names = []
    
    for name, func, weight_path in models:
        print(f"\nTesting: {name}")
        try:
            if os.path.exists(weight_path):
                mask = func(img, weight_path)
                masks.append(mask)
                model_names.append(name)
            else:
                print(f"  Weight file not found: {weight_path}")
                masks.append(None)
                model_names.append(name)
        except Exception as e:
            print(f"  Error: {e}")
            masks.append(None)
            model_names.append(name)
    
    print("\n" + "=" * 60)
    print("保存对比结果...")
    save_mask_comparison(masks, model_names)
    
    print("\n对比完成！请查看 matting_comparison/ 目录下的mask图片")
    print("建议重点关注：")
    print("1. 头发边缘是否有白边/光晕")
    print("2. 肩膀/衣领边缘是否干净")
    print("3. 小发丝是否能保留")


if __name__ == "__main__":
    main()