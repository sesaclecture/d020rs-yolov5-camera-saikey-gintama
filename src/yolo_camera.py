import torch
import cv2

# Model​
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# Video capture
cap = cv2.VideoCapture(0)

# TODO: Loop for camera frames
while(cap.isOpened()):
    # Read frame (BGR to RGB)
    ret, frame = cap.read()
    # TODO: break the loop on error
    if not ret or frame is None:
        print("not ret or frame is None")
        break

    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    # TODO: Boudning box 그리기
    for i, obj in enumerate(results.xyxy[0]):
        # TODO: 인식결과를 표시하기 위한 좌표를 얻음
        print("obj : ", obj)
        x1, y1, x2, y2 = map(int, obj[:4]) 
        conf = float(obj[4])
        cls  = int(obj[5])

        # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
        label = f"{model.names[cls]} {conf:.2f}"

        # TODO: OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
        obj_info = list(map(int, obj))
        print(f"Object {i}: {model.names[obj_info[5]]}")
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


    # TODO: 화면 표시
    cv2.imshow("frame", frame)

    # TODO: 종료를 위한 key 처리
    key = cv2.waitKey(1) & 0xFF
    if key & 0xFF == ord('q'): break # 종료 

cap.release()
cv2.destroyAllWindows()
