"""
YOLOv8s 轻量化改进模型 (GEA + KD) - 完整推理测试 Demo (不抽样版)
特点:
  - 处理指定文件夹下所有图片 (不随机抽样)
  - 任意尺寸/比例/格式的图片都能直接处理
  - 自动处理: 中文路径 / 灰度图 / RGBA透明通道 / 损坏图片
  - 输出: 可视化标注图 + YOLO格式txt + 汇总json + FPS统计
"""

import os
import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ---------------------------- 参数配置 ---------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8s 轻量化改进模型 - 全量测试 Demo')
    parser.add_argument('--weights', type=str,
                        default=r'D:\Capstone\runs\detect\yolov8s_GEA(KD)\weights\best.pt',
                        help='模型权重路径 (.pt)')
# ---------------------------- 文件路径调整 ---------------------------- #
    parser.add_argument('--source', type=str,
                        default=r'D:\Capstone\Code\test\images',
                        help='待测试图片所在文件夹 (或单张图片路径)')
    parser.add_argument('--save-dir', type=str,
                        default=r'D:\Capstone\Code\test\runs\demo_full',
                        help='结果保存根目录')
    parser.add_argument('--imgsz', type=int, default=1024, help='推理图像尺寸')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU 阈值')
    parser.add_argument('--device', type=str, default='0',
                        help='推理设备: cpu / 0 / 0,1')
    parser.add_argument('--warmup', type=int, default=3, help='预热次数')
    parser.add_argument('--recursive', action='store_true',
                        help='是否递归搜索子文件夹 (默认只搜当前文件夹)')
    return parser.parse_args()


# ---------------------------- 工具函数 ---------------------------- #
IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']


def safe_imread(path):
    """支持中文路径, 自动处理灰度/RGBA图, 失败返回 None"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    except Exception as e:
        print(f'    ⚠ 读取失败: {path.name}  ({e})')
        return None


def safe_imwrite(path, img):
    """支持中文路径的写入"""
    ext = Path(path).suffix
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(str(path))
    return success


def collect_all_images(source, recursive=False):
    """收集 source 下所有图片 (不抽样)"""
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f'路径不存在: {source}')

    # 单张图片
    if p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise ValueError(f'不支持的文件类型: {p.suffix}')
        return [p]

    # 文件夹
    if p.is_dir():
        if recursive:
            files = sorted([f for f in p.rglob('*') if f.is_file() and f.suffix.lower() in IMG_EXTS])
        else:
            files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS])
        if not files:
            raise FileNotFoundError(f'文件夹中没有找到图片: {source}')
        return files

    raise ValueError(f'不是有效的文件或文件夹: {source}')


def make_dirs(save_root):
    save_root = Path(save_root)
    (save_root / 'images').mkdir(parents=True, exist_ok=True)
    (save_root / 'labels').mkdir(parents=True, exist_ok=True)
    return save_root


def format_detection(result, img_name):
    info = {'image': img_name, 'detections': []}
    yolo_lines = []

    if result.boxes is None or len(result.boxes) == 0:
        return info, yolo_lines

    names = result.names
    h, w = result.orig_shape
    boxes = result.boxes

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        info['detections'].append({
            'id': i,
            'class_id': cls_id,
            'class_name': names[cls_id],
            'confidence': round(conf, 4),
            'bbox_xyxy': [round(v, 1) for v in (x1, y1, x2, y2)],
        })
        yolo_lines.append(
            f'{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}'
        )
    return info, yolo_lines


def print_detection_info(info, idx, total, time_ms, img_shape):
    h, w = img_shape[:2]
    print(f'\n[{idx}/{total}] {info["image"]}  尺寸: {w}x{h}  推理: {time_ms:.1f} ms')
    if not info['detections']:
        print('  └─ 未检测到目标')
        return
    print(f'  └─ 检测到 {len(info["detections"])} 个目标:')
    for det in info['detections'][:10]:
        x1, y1, x2, y2 = det['bbox_xyxy']
        print(f'     #{det["id"]:<3} {det["class_name"]:<15} '
              f'conf={det["confidence"]:.3f}  '
              f'bbox=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]')
    if len(info['detections']) > 10:
        print(f'     ... 还有 {len(info["detections"]) - 10} 个未显示')


# ---------------------------- 主流程 ---------------------------- #
def main():
    args = parse_args()

    # 1. 收集所有图片 (不抽样)
    img_paths = collect_all_images(args.source, recursive=args.recursive)
    print(f'\n[全量测试模式] 共找到 {len(img_paths)} 张图片')
    print(f'图片来源: {Path(args.source).resolve()}')
    print(f'递归子文件夹: {"是" if args.recursive else "否"}')

    save_root = make_dirs(args.save_dir)

    # 2. 加载模型
    print('\n' + '=' * 70)
    print(f'加载模型: {args.weights}')
    print(f'推理输入尺寸: {args.imgsz}')
    print(f'置信度: {args.conf}  IoU: {args.iou}  设备: {args.device}')
    model = YOLO(args.weights)
    print(f'类别数: {len(model.names)}')
    print(f'类别: {model.names}')
    print('=' * 70)

    # 3. 预热
    warmup_img = None
    for p in img_paths:
        img = safe_imread(p)
        if img is not None:
            warmup_img = img
            break
    if warmup_img is None:
        print('错误: 所有图片都无法读取')
        return

    if args.warmup > 0:
        print(f'\n预热推理 {args.warmup} 次...')
        for _ in range(args.warmup):
            model.predict(warmup_img, imgsz=args.imgsz,
                          conf=args.conf, iou=args.iou,
                          device=args.device, verbose=False)
        print('预热完成')
        print('-' * 70)

    # 4. 推理所有图片
    all_results = []
    timings = {'preprocess': [], 'inference': [], 'postprocess': [], 'total': []}
    skipped = []
    total_detections = 0

    for idx, img_path in enumerate(img_paths, 1):
        img = safe_imread(img_path)
        if img is None:
            print(f'\n[{idx}/{len(img_paths)}] {img_path.name}  ⚠ 跳过 (读取失败)')
            skipped.append(img_path.name)
            continue

        t_start = time.time()
        results = model.predict(
            source=img,
            imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=args.device, verbose=False,
        )
        t_total = (time.time() - t_start) * 1000
        result = results[0]

        spd = result.speed
        timings['preprocess'].append(spd['preprocess'])
        timings['inference'].append(spd['inference'])
        timings['postprocess'].append(spd['postprocess'])
        timings['total'].append(t_total)

        info, yolo_lines = format_detection(result, img_path.name)
        info['time_ms'] = round(t_total, 2)
        info['orig_size'] = [img.shape[1], img.shape[0]]
        info['speed_breakdown'] = {k: round(v, 2) for k, v in spd.items()}
        all_results.append(info)
        total_detections += len(info['detections'])

        print_detection_info(info, idx, len(img_paths), t_total, img.shape)

        # 保存可视化图
        annotated = result.plot()
        safe_imwrite(save_root / 'images' / img_path.name, annotated)

        # 保存 YOLO txt
        txt_path = save_root / 'labels' / (img_path.stem + '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

    # 5. 汇总 json
    summary_path = save_root / 'detections.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 6. FPS / 速度统计
    n = len(timings['total'])
    if n == 0:
        print('\n没有成功处理的图片')
        return

    avg_pre = sum(timings['preprocess']) / n
    avg_inf = sum(timings['inference']) / n
    avg_post = sum(timings['postprocess']) / n
    avg_total = sum(timings['total']) / n
    fps_inference = 1000 / avg_inf if avg_inf > 0 else 0
    fps_pipeline = 1000 / avg_total if avg_total > 0 else 0

    print('\n' + '=' * 70)
    print(f'全量测试完成 - 共处理 {n} 张图片, 总检测数 {total_detections} 个')
    print(f'推理速度统计 (平均值, imgsz={args.imgsz})')
    print('-' * 70)
    print(f'  预处理      : {avg_pre:>8.2f} ms')
    print(f'  推理        : {avg_inf:>8.2f} ms   <-- 纯模型推理')
    print(f'  后处理(NMS) : {avg_post:>8.2f} ms')
    print(f'  总耗时      : {avg_total:>8.2f} ms')
    print('-' * 70)
    print(f'  纯推理 FPS  : {fps_inference:>8.2f}')
    print(f'  端到端 FPS  : {fps_pipeline:>8.2f}')
    print(f'  平均每图检测: {total_detections / n:.1f} 个目标')
    print('=' * 70)

    if skipped:
        print(f'\n⚠ 跳过的图片 ({len(skipped)} 张): {skipped}')

    print('\n结果文件保存位置:')
    print(f'  可视化图片  : {(save_root / "images").resolve()}')
    print(f'  YOLO标签txt : {(save_root / "labels").resolve()}')
    print(f'  汇总 JSON   : {summary_path.resolve()}')


if __name__ == '__main__':
    main()