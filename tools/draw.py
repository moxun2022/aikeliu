import cv2
import argparse
import numpy as np

# 全局变量
points = []
frame_to_draw_on = None

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，用于记录和绘制点"""
    global points, frame_to_draw_on

    if event == cv2.EVENT_LBUTTONDOWN:
        # 添加新点
        points.append((x, y))
        print(f"Point added: ({x}, {y})")

        # 在图像上绘制
        # 画点
        cv2.circle(frame_to_draw_on, (x, y), 5, (0, 255, 0), -1)
        # 如果有多个点，就连成线
        if len(points) > 1:
            cv2.line(frame_to_draw_on, points[-2], points[-1], (0, 0, 255), 2)
        
        cv2.imshow("Draw Region/Line", frame_to_draw_on)

def main(video_path):
    """主函数"""
    global points, frame_to_draw_on

    # 1. 打开视频并读取第一帧
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame from the video.")
        cap.release()
        return
    
    # 复制第一帧用于绘制
    frame_to_draw_on = first_frame.copy()
    original_frame = first_frame.copy() # 保存原始帧用于重置

    # 2. 创建窗口并设置鼠标回调
    window_name = "Draw Region/Line"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("--- Interactive Region/Line Drawer ---")
    print("Instructions:")
    print(" - Left-click to add points.")
    print(" - Press 's' to save the coordinates and quit.")
    print(" - Press 'r' to reset all points.")
    print(" - Press 'q' to quit without saving.")
    print("------------------------------------")

    while True:
        # 在图像上显示提示信息
        display_frame = frame_to_draw_on.copy()
        cv2.putText(display_frame, "Press 's' to save, 'r' to reset, 'q' to quit", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF

        # 按 's' 保存
        if key == ord('s'):
            if len(points) < 2:
                print("Error: You need at least 2 points to define a line or region.")
            else:
                print("\nCoordinates saved successfully!")
                print("Copy the following line into your tracking script:")
                print("-------------------------------------------------")
                # 如果是两个点，定义为线
                if len(points) == 2:
                    print(f"counting_line = {points}")
                # 如果是多个点，定义为区域
                else:
                    print(f"counting_region = np.array({points}, np.int32)")
                print("-------------------------------------------------")
                break
        
        # 按 'r' 重置
        elif key == ord('r'):
            points = []
            frame_to_draw_on = original_frame.copy()
            print("Points have been reset. You can start drawing again.")
        
        # 按 'q' 退出
        elif key == ord('q'):
            print("Quitting without saving.")
            break
            
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively draw a line or region on a video frame.")
    parser.add_argument("--video_path", required=True, help="Path to the video file.")
    args = parser.parse_args()
    
    main(args.video_path)
