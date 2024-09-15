import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("C:/Users/areba/OneDrive/Documentos/hackmty2024/yolo_env/videos/prueba2.mp4")
heatmap_obj = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, view_img=True, shape="circle", names=model.names)

classes_for_heatmap = [0]  # Classes to visualize
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    tracks = model.track(im0, persist=True, show=False, classes=classes_for_heatmap)
    im0 = heatmap_obj.generate_heatmap(im0, tracks)
    cv2.imshow("Heatmap", im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()