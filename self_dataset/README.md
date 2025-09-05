# 🧠 YOLOv5 Object Detection – Custom Training

This project implements an **Object Detection** model using YOLOv5, trained on a **custom dataset** created with [MakeSense.ai](https://makesense.ai).
**Objective:** Train a model to detect custom classes using a limited dataset, enhanced with **data augmentation** techniques.

---

## 📦 Installation & Requirements

First, install the required Python packages:

```bash
# Install packages for data augmentation and image processing
pip install albumentations==1.3.0 opencv-python-headless

# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Install YOLOv5 dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

- **Number of initial images:** 62 (labeled using MakeSense.ai)
- **Train/Validation Split:** 80% train / 20% validation
- **Applied Augmentations using Albumentations:**

  - Horizontal Flip
  - Brightness & Contrast Adjustment
  - Gaussian Noise & Blur
  - Shift, Scale & Rotate

**Example `data.yaml` configuration:**

```yaml
train: /content/dataset_yolo5/images/train
val: /content/dataset_yolo5/images/val

nc: 3
names: ["marziye", "one", "two"]
```

- `nc` → number of classes
- `names` → list of class labels

---

## 🏋️ Training

The YOLOv5-Large model was trained for 100 epochs using the following command:

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data /content/dataset_yolo5/data.yaml \
  --weights yolov5l.pt \
  --name final_tunedv44 \
  --patience 50
```

**Key Training Metrics:**

- Precision ≈ 1.0
- Recall ≈ 1.0
- mAP\@0.5 ≈ 1.0
- mAP\@0.5:0.95 ≈ 0.6

> ⚠️ Note: `--patience 50` stops training early if the model does not improve after 50 epochs.

---

## 🎥 Inference / Testing

Run inference on images or video files using:

```bash
python detect.py \
  --weights runs/train/final_tunedv44/weights/best.pt \
  --img 640 \
  --conf 0.25 \
  --iou-thres 0.45 \
  --source /content/drive/MyDrive/testtt.mp4
```

**Parameters:**

- `--weights` → path to trained model weights
- `--img` → input image size
- `--conf` → confidence threshold for predictions
- `--iou-thres` → IoU threshold for non-max suppression
- `--source` → input image, folder, video, or webcam stream

---

## ✅ Results & Conclusion

- The small dataset was successfully augmented to improve training performance.
- YOLOv5-Large achieved **high precision and recall** metrics on the custom dataset.
- The trained model performs well on video inference.
- This setup can be scaled to larger datasets and additional classes with minimal adjustments.

---

## ⚙️ Optional: Exporting the Model

You can export the trained model to formats like **ONNX** or **TensorRT** for deployment:

```bash
python export.py --weights runs/train/final_tunedv44/weights/best.pt --include onnx
```

- This allows faster inference on production environments.

---
