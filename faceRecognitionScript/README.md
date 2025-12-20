# 📘 Advanced AI Attendance & Security System – Jetson Nano

*A high-definition (1080p) real-time face recognition system featuring GPU acceleration, Google Sheets logging, and dual-mode security breach detection.*

---

## ⚠️ CRITICAL: PRE-REQUISITE STEP

**Before running any code**, you must download the required model weights, engines, and configuration files.

👉 **[Download Project Dependencies Here](https://drive.google.com/drive/folders/1Xfy9o1Z_QP_hlToJZ2dnD5TEpQB8zFUH?usp=drive_link)**

*Ensure all downloaded files (e.g., `.caffemodel`, `.prototxt`, `.onnx`, and `.engine`) are placed directly in the root directory as shown in the Project Structure below.*

---

## ⭐ Overview

This project is an industrial-grade attendance and security solution optimized for the **NVIDIA Jetson Nano**. It handles wide-angle surveillance to detect multiple faces simultaneously and intelligently distinguishes between authorized entry, piggybacking (tailgating), and unauthorized access.

### 🚀 Key Features

* **Full HD 1080p Wide-Angle Processing:** Optimized for high-clarity recognition in large environments.
* **Dual-Mode Security Logic:** Distinguishes between **Tailgating** (following a known person) and **Unauthorized Entry** (forced entry).
* **GPU Accelerated:** Leverages Jetson CUDA cores for real-time resizing and DNN inference.
* **Smart Cloud Logging:** Automatic sync to Google Sheets with cooldown timers.

---

## 📂 Project Structure

```text
project/
├── facdec_recognitionScript.py  # Main AI & Security Engine
├── enroll_faces.py              # Script to train/generate encodings.pickle
├── imageCaptureLoopScript.py    # Automated dataset collection tool
├── encodings.pickle             # Pre-computed face signatures (Database)
├── deploy.prototxt              # SSD Face Detector configuration
├── res10_300x300_ssd...         # SSD Detector model weights
├── face_recognition_sface...    # SFace recognition model (ONNX/Engine)
├── facedetectionattendece...json # Google Service Account Key
├── ProjectDependencies.url      # MUST DOWNLOAD BEFORE RUNNING
├── ProjectResource&Results.url  # Link to external results and datasets
├── faces/                       # Dataset storage
│   └── unknown/                 # Default folder for unidentified faces
└── results/                     # Performance benchmarks and demo clips
    ├── demo/
    └── Jtop_reaings/            # Hardware utilization screenshots

```

---

## 🧠 Security & Tailgating Logic

The system uses a virtual tripwire set at **75% of the frame height**.

| Event | Logic Condition | Action |
| --- | --- | --- |
| **Authorized** | Face matches database > Threshold | Logs Name, Opens "Gate" (5s Grace) |
| **Tailgating** | Unknown face crosses while Gate is **Open** | Logs `SECURITY: TAILGATE (PIGGYBACKING)` |
| **Unauthorized** | Unknown face crosses while Gate is **Closed** | Logs `SECURITY: UNAUTHORIZED ENTRY` |

---

## ⚙️ Installation & Setup

### 1️⃣ Install Libraries

The Jetson Nano requires JetPack 4.6+. Run:

```bash
sudo apt-get update
pip3 install gspread oauth2client numpy opencv-python

```

### 2️⃣ Google Sheets Integration

* Ensure your `.json` key is in the root folder.
* Share your target Google Sheet with the Service Account email found inside the JSON.

---

## 📸 Adding New Faces (Instructions)

### Step A: Collect Images

1. Create a new folder inside `faces/` named after the person (e.g., `faces/Hamza`).
2. Run the capture script: `python3 imageCaptureLoopScript.py`
3. Move and pose; the script saves images every 300ms.

### Step B: Update the Database

After adding new photos, you must re-generate the facial signatures:

1. Run: `python3 enroll_faces.py`
2. This updates `encodings.pickle` so the system recognizes the new person.

Note: imageCaptureLoopScript.py can be used to automatically creating folders and taking pictures

---

## ▶️ Running the System

```bash
python3 facdec_recognitionScript.py

```

**Prompts:**

* `Enable tailgating detection? (y/n)`: Activates tripwire logic.
* `Enter recognition threshold %`: Recommended **60** for standard lighting.

---

## 🧪 Testing Benchmarks

Refer to `results/Jtop_reaings/` to view hardware performance:

* **jtop_wihtoutface.png**: Idle load.
* **jtop_withface.png**: Load during active 1080p recognition.

---

## 👨‍💻 Author

**Hamza Khan** 🔗 LinkedIn: [https://pk.linkedin.com/in/hamza-khan-rajput](https://pk.linkedin.com/in/hamza-khan-rajput)

---

## 📜 License

MIT License — Free for personal and commercial use.

---
