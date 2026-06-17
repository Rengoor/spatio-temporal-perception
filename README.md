# Spatio-Temporal Robotic Perception Pipeline for Dynamic Environments

An end-to-end computer vision and tracking system built to handle real-world robotic challenges (motion blur, lighting drops, and occlusions). This system modifies a spatial backbone with temporal feature fusion layers, calculates a custom inter-frame consistency loss, and deploys via a hardware-agnostic C++ inference engine using ONNX Runtime.

## 🚀 Key Architectural Features
* **Spatial Core:** YOLOv10 (NMS-free end-to-end object detector).
* **Temporal Integration:** Custom multi-frame ConvLSTM neck implemented in PyTorch to persist spatial memory across sequential video frames.
* **Tracking Block:** ByteTrack association framework for robust identity maintenance.
* **Production Deployment:** Real-time processing engine written entirely in C++ using OpenCV and ONNX Runtime.
* **MLOps Core:** Self-hosted MLflow tracking for hyperparameter optimization and FiftyOne for visual dataset error slicing.

---

## 🛠️ Tech Stack & Dependencies
* **Frameworks:** PyTorch, TorchVision
* **Dataset Engine:** FiftyOne
* **Tracking Server:** MLflow
* **Runtime Optimization:** ONNX Runtime C++, OpenCV C++, CMake

---

## 📊 Dataset Evaluation: OpenLORIS-Object
We utilize the **OpenLORIS-Object** robotic dataset. Unlike sterile academic benchmarks, it features significant:
1. **Dynamic Illumination:** Day-to-night transitions.
2. **Heavy Occlusion:** Objects moving completely behind static industrial structures.
3. **Sensor Artifacts:** Severe robotic camera motion blur.

---

## 📈 Metric Dashboard (To Be Populated)
| Model Variant | Input Context | mAP@0.5 | mIoU | ID Switches (IDS) | Latency (C++ Engine) |
|---|---|---|---|---|---|
| Baseline YOLOv10 | Single Frame | - | - | - | - |
| **Spatio-Temporal YOLOv10** | **3-Frame Sequence** | **-** | **-** | **-** | **-** |

---

## 🛠️ Step-by-Step Build Instructions
*(To be completed during Phase 4)*