from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from torchvision.ops import nms

app = FastAPI()

# Load YOLOv8 model (update path if needed)
model = YOLO("models/best.pt")

# Define thresholds
MIN_CONFIDENCE = 0.4
IOU_THRESHOLD = 0.3

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    results = model(img, conf=MIN_CONFIDENCE, augment=True)
    print(f"Raw detections: {len(results[0].boxes)}")

    raw_predictions = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        if conf < MIN_CONFIDENCE:
            continue

        class_names = model.names if hasattr(model, 'names') else {}
        cls_name = class_names.get(cls_id, f"class_{cls_id}")

        raw_predictions.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": cls_id,
            "class_name": cls_name
        })

    if raw_predictions:
        boxes_tensor = torch.tensor([p["bbox"] for p in raw_predictions], dtype=torch.float32)
        scores_tensor = torch.tensor([p["confidence"] for p in raw_predictions])
        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=IOU_THRESHOLD)
        predictions = [raw_predictions[i] for i in keep_indices]
    else:
        predictions = []

    # Optional: draw boxes
    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox"]
        cls_name = pred["class_name"]
        conf = pred["confidence"]
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Optional: save annotated image
    cv2.imwrite("output.png", img)

    if not predictions:
        return {"results": "no results"}

    return {"results": predictions}
