"""
YOLOv8s 轻量化改进模型 (GEA + KD) - 视频推理 Demo
功能:
  - 处理单个视频文件 或 文件夹下所有视频
  - 输出: 标注后的视频 + 每帧检测json + 实时/平均FPS
  - 可选: 实时窗口预览 (按 q 退出)
  - 可选: 抽帧处理 (每 N 帧处理一次, 加快测试)
"""

import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ---------------------------- 参数配置 ---------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8s 视频推理 Demo')
    parser.add_argument('--weights', type=str,
                        default=r'D:\Capstone\runs\detect\yolov8s_GEA(KD)\weights\best.pt',
                        help='模型权重路径 (.pt)')
    parser.add_argument('--source', type=str,
                        default=r'D:\Capstone\Code\test\videos',
                        help='视频文件路径 或 包含视频的文件夹')
    parser.add_argument('--save-dir', type=str,
                        default=r'D:\Capstone\Code\test\runs\demo_video',
                        help='结果保存根目录')
    parser.add_argument('--imgsz', type=int, default=1024, help='推理图像尺寸')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU 阈值')
    parser.add_argument('--device', type=str, default='0', help='设备: cpu / 0')
    parser.add_argument('--show', action='store_true',
                        help='实时显示窗口 (按 q 退出当前视频)')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='抽帧间隔 (1=每帧都处理, 2=隔1帧, 3=隔2帧...)')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='最多处理多少帧 (0=不限制, 用于快速测试)')
    parser.add_argument('--no-save-video', action='store_true',
                        help='不保存输出视频 (只看实时效果)')
    return parser.parse_args()


# ---------------------------- 工具函数 ---------------------------- #
VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']


def collect_videos(source):
    """收集所有视频"""
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f'路径不存在: {source}')
    if p.is_file():
        if p.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f'不支持的视频格式: {p.suffix}')
        return [p]
    if p.is_dir():
        files = sorted([f for f in p.iterdir()
                        if f.is_file() and f.suffix.lower() in VIDEO_EXTS])
        if not files:
            raise FileNotFoundError(f'文件夹中没有找到视频: {source}')
        return files
    raise ValueError(f'无效路径: {source}')


def format_detection(result):
    """提取一帧的检测信息"""
    dets = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    names = result.names
    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
        dets.append({
            'class_id': cls_id,
            'class_name': names[cls_id],
            'confidence': round(conf, 4),
            'bbox_xyxy': [round(v, 1) for v in (x1, y1, x2, y2)],
        })
    return dets


def process_video(video_path, model, args, save_root):
    """处理单个视频"""
    print('\n' + '=' * 70)
    print(f'处理视频: {video_path.name}')

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'  ⚠ 无法打开视频, 跳过')
        return None

    # 视频参数
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / src_fps if src_fps > 0 else 0

    print(f'  分辨率: {width}x{height}  原始FPS: {src_fps:.1f}  '
          f'总帧数: {total_frames}  时长: {duration:.1f}s')
    if args.frame_skip > 1:
        print(f'  抽帧间隔: 每 {args.frame_skip} 帧处理一次')
    if args.max_frames > 0:
        print(f'  最多处理: {args.max_frames} 帧')

    # 输出视频写入器
    writer = None
    if not args.no_save_video:
        out_path = save_root / 'videos' / f'{video_path.stem}_detected.mp4'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 输出 FPS 按抽帧调整, 这样播放速度看起来正常
        out_fps = src_fps / args.frame_skip
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height))

    # 推理循环
    frame_idx = 0       # 视频原始帧序号
    processed = 0       # 实际处理的帧数
    inference_times = []
    all_frame_dets = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 抽帧
        if frame_idx % args.frame_skip != 0:
            frame_idx += 1
            continue

        # 推理
        t0 = time.time()
        results = model.predict(
            frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=args.device, verbose=False,
        )
        dt = (time.time() - t0) * 1000
        inference_times.append(dt)
        result = results[0]

        # 解析 + 记录
        dets = format_detection(result)
        all_frame_dets.append({
            'frame': frame_idx,
            'time_ms': round(dt, 2),
            'num_detections': len(dets),
            'detections': dets,
        })

        # 绘制
        annotated = result.plot()

        # 叠加实时信息
        fps_now = 1000 / dt if dt > 0 else 0
        info_text = f'Frame {frame_idx}/{total_frames}  FPS:{fps_now:.1f}  Det:{len(dets)}'
        cv2.putText(annotated, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 写入输出视频
        if writer is not None:
            writer.write(annotated)

        # 实时显示
        if args.show:
            cv2.imshow(f'Detection - {video_path.name}', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('  用户中断 (按了 q)')
                break

        # 进度条 (每 30 帧打印一次)
        processed += 1
        if processed % 30 == 0 or processed == 1:
            progress = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f'  [{progress:5.1f}%] 帧 {frame_idx}/{total_frames}  '
                  f'最近 FPS: {fps_now:.1f}  检测目标: {len(dets)}')

        frame_idx += 1
        if args.max_frames > 0 and processed >= args.max_frames:
            print(f'  达到最大帧数 {args.max_frames}, 停止')
            break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # 单视频统计
    if not inference_times:
        print('  ⚠ 没有处理任何帧')
        return None

    avg_inf = sum(inference_times) / len(inference_times)
    avg_fps = 1000 / avg_inf
    total_dets = sum(d['num_detections'] for d in all_frame_dets)

    summary = {
        'video': video_path.name,
        'resolution': [width, height],
        'src_fps': round(src_fps, 2),
        'total_frames': total_frames,
        'processed_frames': len(inference_times),
        'frame_skip': args.frame_skip,
        'avg_inference_ms': round(avg_inf, 2),
        'avg_fps': round(avg_fps, 2),
        'total_detections': total_dets,
        'avg_detections_per_frame': round(total_dets / len(inference_times), 2),
        'frames': all_frame_dets,
    }

    # 保存每帧检测详情
    json_path = save_root / 'json' / f'{video_path.stem}.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\n  视频处理完成:')
    print(f'    处理帧数    : {len(inference_times)} / {total_frames}')
    print(f'    平均推理    : {avg_inf:.2f} ms/帧')
    print(f'    平均 FPS    : {avg_fps:.2f}')
    print(f'    总检测目标  : {total_dets}')
    if writer is not None:
        print(f'    输出视频    : {save_root / "videos" / (video_path.stem + "_detected.mp4")}')
    print(f'    检测 JSON   : {json_path}')

    return summary


# ---------------------------- 主流程 ---------------------------- #
def main():
    args = parse_args()

    videos = collect_videos(args.source)
    print(f'\n找到 {len(videos)} 个视频:')
    for v in videos:
        print(f'  - {v.name}')

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 70)
    print(f'加载模型: {args.weights}')
    print(f'推理输入: {args.imgsz}  conf: {args.conf}  iou: {args.iou}  设备: {args.device}')
    model = YOLO(args.weights)
    print(f'类别: {model.names}')

    # 处理每个视频
    all_summaries = []
    for v in videos:
        s = process_video(v, model, args, save_root)
        if s:
            all_summaries.append(s)

    # 总汇总
    if all_summaries:
        total_summary_path = save_root / 'all_videos_summary.json'
        with open(total_summary_path, 'w', encoding='utf-8') as f:
            # 总汇总不存每帧详情, 只存统计信息
            brief = [{k: v for k, v in s.items() if k != 'frames'}
                     for s in all_summaries]
            json.dump(brief, f, indent=2, ensure_ascii=False)
        print('\n' + '=' * 70)
        print(f'全部完成! 共处理 {len(all_summaries)} 个视频')
        print(f'总汇总: {total_summary_path}')


if __name__ == '__main__':
    main()