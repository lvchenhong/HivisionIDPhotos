"""
客观分析 HivisionIDPhotos 生成图, 输出可视化报告
- 构图比例 (顶/底/左/右留白, 脸宽占比, 面部垂直位置)
- alpha 边缘残留可视化
- 原图 vs 生成图 亮度/对比度/饱和度
- 肤色偏色检测
- 头顶上方毛糙残留
"""
import os
os.environ['ORT_DISABLE_TELEMETRY'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = r"D:\HivisionIDPhotos\HivisionIDPhotos-master"
SRC = r"C:\Users\Administrator\Desktop\e2efefae1791738e829b5c8cd4dbdf26.jpg"
OUT_HD = f"{BASE}\\test_output_user_1inch_hd.jpg"
OUT_STD = f"{BASE}\\test_output_user_1inch.jpg"
REPORT = f"{BASE}\\analysis_report.png"


def is_background(rgb, thresh=250):
    # 严格: 三通道都 >= 250 才是纯白底
    return (rgb[..., 0] >= thresh) & (rgb[..., 1] >= thresh) & (rgb[..., 2] >= thresh)


def is_skin(arr):
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (r > g) & (g > b) & (r > 60) & (r < 245) & (g > 40) & (b > 30) & (b < 220)


def luminance(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def analyze(path, label):
    im = Image.open(path)
    is_rgba = im.mode == 'RGBA'
    arr = np.array(im.convert('RGBA') if not is_rgba else im)
    H, W = arr.shape[:2]
    rgb = arr[..., :3].astype(np.float32)
    a = arr[..., 3].astype(np.float32) if arr.shape[2] == 4 else np.full((H, W), 255, np.float32)

    bg = is_background(rgb)
    fg = ~bg
    row_cov = fg.mean(axis=1)
    col_cov = fg.mean(axis=0)

    top = int(np.argmax(row_cov > 0.005))
    bot = H - 1 - int(np.argmax(row_cov[::-1] > 0.005))
    left = int(np.argmax(col_cov > 0.005))
    right = W - 1 - int(np.argmax(col_cov[::-1] > 0.005))

    info = {
        'label': label, 'path': path, 'size': (W, H), 'mode': im.mode,
        'top': top, 'bot': bot, 'left': left, 'right': right,
        'top_pct': top / H * 100, 'bot_pct': (H - 1 - bot) / H * 100,
        'left_pct': left / W * 100, 'right_pct': (W - 1 - right) / W * 100,
        'mean_rgb': rgb.reshape(-1, 3).mean(axis=0).round(1).tolist(),
        'std_rgb': rgb.reshape(-1, 3).std(axis=0).round(1).tolist(),
        'mean_lum': luminance(rgb).mean(),
        'alpha': a,
        'rgb': rgb,
        'fg': fg,
    }

    # 主体区域
    subj = rgb[top:bot + 1, left:right + 1]
    info['subj_mean_rgb'] = subj.reshape(-1, 3).mean(axis=0).round(1).tolist()
    info['subj_lum'] = luminance(subj)
    info['subj_lum_mean'] = info['subj_lum'].mean()
    info['subj_lum_std'] = info['subj_lum'].std()
    info['subj_saturation'] = (subj.max(axis=-1) - subj.min(axis=-1)).mean()

    # 面部
    subh = subj.shape[0]
    face_band = subj[int(subh * 0.20):int(subh * 0.55)]
    fb_fg = ~is_background(face_band)
    if fb_fg.any():
        fcols = fb_fg.mean(axis=0)
        info['face_w_pct'] = float(fcols.max() * 100)
        info['face_center_x_pct'] = float((np.argmax(fcols) / W + left / W) * 100)
    else:
        info['face_w_pct'] = 0
        info['face_center_x_pct'] = 50

    # 肤色
    skin_mask = is_skin(subj)
    if skin_mask.sum() > 100:
        skin_px = subj[skin_mask]
        info['skin_count'] = int(skin_mask.sum())
        info['skin_mean'] = skin_px.mean(axis=0).round(1).tolist()
        info['skin_RG'] = float(skin_px[:, 0].mean() / max(1, skin_px[:, 1].mean()))
        info['skin_RB'] = float(skin_px[:, 0].mean() - skin_px[:, 2].mean())
    else:
        info['skin_count'] = 0

    # alpha 边缘残留
    edge = (a > 10) & (a < 245)
    info['edge_pct'] = float(edge.mean() * 100)
    info['outside_residue'] = {
        'top': int((a[:5] > 0).sum()),
        'bot': int((a[-5:] > 0).sum()),
        'left': int((a[:, :5] > 0).sum()),
        'right': int((a[:, -5:] > 0).sum()),
    }
    if top > 0:
        info['hairline_residue'] = int((a[max(0, top - 12):top] > 5).sum())
    else:
        info['hairline_residue'] = 0

    return info


def make_report(src_info, out_info, out_path):
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # 行 0: 原图 / 输出图 / alpha 通道 / 边缘热图
    ax1 = fig.add_subplot(gs[0, 0])
    src_rgb = np.array(Image.open(SRC).convert('RGB'))
    ax1.imshow(src_rgb); ax1.set_title(f"SRC {src_info['size']}", fontsize=11); ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(out_info['rgb'].astype(np.uint8))
    ax2.set_title(f"OUT {out_info['size']}", fontsize=11); ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(out_info['alpha'], cmap='gray', vmin=0, vmax=255)
    ax3.set_title("Alpha (white=fg)", fontsize=11); ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    edge_mask = (out_info['alpha'] > 10) & (out_info['alpha'] < 245)
    overlay = out_info['rgb'].astype(np.uint8).copy()
    overlay[edge_mask] = [255, 0, 0]  # 红色标出过渡区
    ax4.imshow(overlay); ax4.set_title(f"Edge residues (red): {out_info['edge_pct']:.2f}%", fontsize=11); ax4.axis('off')

    # 行 1: 主体裁切 / 肤色 mask / 亮度图 / 对比柱状
    ax5 = fig.add_subplot(gs[1, 0])
    sub_rgb = out_info['rgb'][out_info['top']:out_info['bot']+1,
                              out_info['left']:out_info['right']+1].astype(np.uint8)
    ax5.imshow(sub_rgb); ax5.set_title("Subject crop", fontsize=11); ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 1])
    skin_mask = is_skin(sub_rgb)
    skin_vis = sub_rgb.copy()
    skin_vis[~skin_mask] = sub_rgb[~skin_mask] * 0.3
    ax6.imshow(skin_vis); ax6.set_title(f"Skin pixels: {out_info['skin_count']}", fontsize=11); ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    lum = luminance(out_info['rgb'])
    ax7.imshow(lum, cmap='gray'); ax7.set_title(f"Lum mean={out_info['mean_lum']:.1f}", fontsize=11); ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    # 原图 vs 输出 vs 主体 三个亮度分布
    src_lum = luminance(np.array(Image.open(SRC).convert('RGB')).astype(np.float32))
    ax8.hist(src_lum.ravel(), bins=60, alpha=0.4, label=f'SRC mean={src_lum.mean():.0f}', color='blue')
    ax8.hist(out_info['mean_lum']*np.ones(1), color='red', label=f'OUT mean={out_info["mean_lum"]:.0f}')
    ax8.hist(lum.ravel(), bins=60, alpha=0.4, label='OUT all', color='orange')
    ax8.set_title("Luminance histogram", fontsize=11)
    ax8.legend(fontsize=8); ax8.set_xlim(0, 255)

    # 行 2: 文本指标
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')
    txt_lines = [
        f"== 构图 (OUT) ==",
        f"  头顶留白: {out_info['top_pct']:.2f}%   期望 ~9.0%   (差异 {out_info['top_pct']-9.0:+.2f}%)",
        f"  底到下边: {out_info['bot_pct']:.2f}%   (理想 30~40%)",
        f"  左留白:   {out_info['left_pct']:.2f}%   右留白: {out_info['right_pct']:.2f}%   居中差 {(out_info['left_pct']-out_info['right_pct']):+.2f}%",
        f"  脸宽占比: {out_info['face_w_pct']:.1f}%   脸水平中心: {out_info['face_center_x_pct']:.1f}%   (理想 50%)",
        f"",
        f"== 抠图质量 ==",
        f"  alpha 边缘 (10<α<245) 占比: {out_info['edge_pct']:.2f}%   (理想 <0.5%, 越低越干净)",
        f"  头顶上方 12px 半透明残留: {out_info['hairline_residue']} 像素   (理想 0)",
        f"  主体外 4 边残留: 上{out_info['outside_residue']['top']} 下{out_info['outside_residue']['bot']} "
        f"左{out_info['outside_residue']['left']} 右{out_info['outside_residue']['right']}",
        f"",
        f"== 颜色 / 美颜 ==",
        f"  整图亮度 mean: SRC={src_lum.mean():.1f}  OUT={out_info['mean_lum']:.1f}   (变亮/暗 Δ={out_info['mean_lum']-src_lum.mean():+.1f})",
        f"  主体亮度 mean: {out_info['subj_lum_mean']:.1f}  std: {out_info['subj_lum_std']:.1f}  (std 高=对比度大)",
        f"  主体饱和度: {out_info['subj_saturation']:.1f}   (≈0 偏黑白, >30 偏鲜艳)",
        f"  肤色均值 RGB: {out_info.get('skin_mean', 'N/A')}   R/G={out_info.get('skin_RG', 0):.2f}  R-B={out_info.get('skin_RB', 0):.1f}",
        f"    (亚洲肤色参考 R/G≈1.10~1.30, R-B≈20~50)",
    ]
    ax9.text(0.02, 0.98, "\n".join(txt_lines), transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', edgecolor='gray'))

    fig.suptitle("HivisionIDPhotos 生成质量客观分析报告", fontsize=14, fontweight='bold')
    fig.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] report saved -> {out_path}")


if __name__ == '__main__':
    src_info = analyze(SRC, 'SRC')
    out_info = analyze(OUT_HD, 'OUT_HD')
    make_report(src_info, out_info, REPORT)
    # 打印关键指标
    print("\n=== KEY METRICS ===")
    print(f"OUT size: {out_info['size']}")
    print(f"  头顶留白 {out_info['top_pct']:.2f}% (期望 9%)")
    print(f"  脸宽占比 {out_info['face_w_pct']:.1f}%")
    print(f"  alpha 边缘 {out_info['edge_pct']:.2f}%")
    print(f"  头顶上方残留 {out_info['hairline_residue']} px")
    print(f"  整图亮度 SRC={luminance(np.array(Image.open(SRC).convert('RGB')).astype(np.float32)).mean():.1f}  OUT={out_info['mean_lum']:.1f}")
    print(f"  主体饱和度 {out_info['subj_saturation']:.1f}")
    print(f"  肤色 R/G={out_info.get('skin_RG', 0):.2f}  R-B={out_info.get('skin_RB', 0):.1f}")
