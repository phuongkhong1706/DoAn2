import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import vision.utils.box_utils_numpy as box_utils


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []

    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]

        if probs.shape[0] == 0:
            continue

        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)

        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])

    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])

    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height

    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


# Đường dẫn tới file nhãn và mô hình ONNX
label_path = "models/voc-model-labels.txt"
onnx_path = "models/onnx/version-RFB-320.onnx"

# Đọc các nhãn
class_names = [name.strip() for name in open(label_path).readlines()]

# Tải mô hình ONNX
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

# Khởi tạo video capture
cap = cv2.VideoCapture(0)

threshold = 0.7
sum_detections = 0

while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        print("No image captured.")
        break

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0).astype(np.float32)

    # Thực hiện inference
    confidences, boxes = ort_session.run(None, {input_name: image})

    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        # Uncomment the line below to display labels
        # cv2.putText(orig_image, label, (box[0] + 20, box[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    sum_detections += boxes.shape[0]
    orig_image = cv2.resize(orig_image, (0, 0), fx=0.7, fy=0.7)
    cv2.imshow('Annotated', orig_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Total detections: {}".format(sum_detections))