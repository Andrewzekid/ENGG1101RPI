from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO("last.pt")

# 打开摄像头（0 表示默认 USB 摄像头）
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 推理
    results = model.predict(frame, imgsz=640, conf=0.25)

    # 将检测结果画到 frame 上
    annotated_frame = results[0].plot()

    # 显示画面
    cv2.imshow("YOLO Live", annotated_frame)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()