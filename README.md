# Smart Surveillance System Using OpenCV & YOLO

## Project Description
This project is a **Smart Surveillance System** developed using **OpenCV and YOLO**.  
It performs real-time human detection using a laptop webcam and monitors a **restricted area**.  
If a human enters the restricted area, the system raises an alert and captures an image as evidence.

---

## Objectives
- Detect humans accurately in real-time
- Monitor restricted areas
- Trigger alert on suspicious activity
- Capture image evidence automatically

---

## Technologies Used
- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Laptop Webcam

---

## Syllabus Mapping
| Topic | Implementation |
|-------|----------------|
| Image Filtering | Frame preprocessing |
| Image Classification | Human detection |
| Image Segmentation | Bounding boxes |
| Video Processing | Real-time webcam feed |
| Object Detection | YOLO person detection |
| Object Tracking | Continuous frame detection |

---

## How It Works
1. Webcam captures live video
2. YOLO detects humans in each frame
3. Restricted area is defined dynamically
4. If a human enters restricted area:
   - Red bounding box is shown
   - Alert sound is triggered
   - Snapshot is saved automatically

---

## How to Run
1. Install dependencies:
   ```bash
   pip install ultralytics opencv-python
