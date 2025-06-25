import cv2
import os
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/fast-reid'))

def point_side(line, point):
    (x1, y1), (x2, y2) = line
    x, y = point
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

def main(video_path, output_dir="output", yolo_weights="yolov8l.pt"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载YOLOv8模型
    model = YOLO(yolo_weights)

    # 初始化DeepSort
    # deepsort = DeepSort(model_path = "/home/gpu/project/aikeliu/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7" )  # 你可能需要根据实际情况传递参数
    deepsort = DeepSort(model_path= "/home/gpu/project/aikeliu/deep_sort_pytorch/thirdparty/fast-reid/weights/market_bot_R50.pth",
                        model_config= "/home/gpu/project/aikeliu/deep_sort_pytorch/thirdparty/fast-reid/configs/Market1501/bagtricks_R50.yml")
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_dir, "tracked_output.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # 1. 定义计数线的位置 (y坐标设为视频高度的一半)
    line_position = int(height / 2)
    crossing_line = [(254, 403), (449, 495)]

    # 2. 初始化计数器和已穿过ID的集合
    person_counter = 0
    crossed_ids = set()

    # 3. 用于存储每个人轨迹的字典
    tracks = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8推理
        results = model(frame)
        
        bboxes_xywh = []
        confs = []
        clss = []
        
        if results[0].boxes:
            # 取出所有框的类别
            all_bboxes_xywh = results[0].boxes.xywh.cpu().numpy()
            all_confs = results[0].boxes.conf.cpu().numpy()
            all_clss = results[0].boxes.cls.cpu().numpy()
            # 只保留类别为0（人）的目标
            mask = all_clss == 0
            bboxes_xywh = all_bboxes_xywh[mask]
            confs = all_confs[mask]
            clss = all_clss[mask]

        # DeepSORT跟踪
        if len(bboxes_xywh)>0:
            outputs, _ = deepsort.update(bboxes_xywh, confs, clss, frame)
        
        # outputs: [x1, y1, x2, y2, track_cls, track_id]
        active_ids = set()
        if len(outputs) > 0:
            for output in outputs:
                x1, y1, x2, y2, cls, track_id = output
                active_ids.add(track_id)

                # 计算中心点
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # 记录轨迹
                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append((cx, cy))

                # --- 核心：穿越检测逻辑 ---
                # 确保轨迹至少有两个点才能判断穿越
                if len(tracks[track_id]) >= 2:
                    # 获取上一个点和当前点
                    prev_point = tracks[track_id][-2]
                    curr_point = tracks[track_id][-1]
                    
                    # 如果点从线上方移动到下方，或从下方移动到上方
                    side_prev = point_side(crossing_line, prev_point)
                    side_curr = point_side(crossing_line, curr_point)
                    if side_prev * side_curr < 0:
                        # 并且这个ID还没有被计数过
                        if track_id not in crossed_ids:
                            person_counter += 1
                            crossed_ids.add(track_id)
                            # (可选) 在穿越时打印信息
                            print(f"ID {track_id} crossed the line! Total count: {person_counter}")

                # 绘制包围框和ID
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'ID:{int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # --- 可视化部分 ---
        # # 1. 绘制所有活跃ID的轨迹
        # for tid in active_ids:
        #     pts = tracks[tid]
        #     if len(pts) > 1:
        #         # 只保留最近的轨迹点，防止画面杂乱
        #         if len(pts) > 30:
        #             tracks[tid] = pts[-30:]
        #         # 绘制轨迹线
        #         for i in range(1, len(pts)):
        #             cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 2)

        # 2. 绘制计数线
        cv2.line(frame, crossing_line[0], crossing_line[1], (255, 0, 0), 3)

        # 3. 绘制统计人数
        cv2.putText(frame, f'Crossed Count: {person_counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        out.write(frame)

    cap.release()
    out.release()
    print(f"跟踪结果已保存到: {out_path}")

if __name__ == "__main__":
    import numpy as np
    main("/home/gpu/project/aikeliu/deep_sort_pytorch/MOT17-09-SDP-raw.webm", output_dir="output")