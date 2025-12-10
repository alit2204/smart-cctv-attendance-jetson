# 📘 Face Recognition Attendance System – Jetson Nano
*A real-time face recognition system using OpenCV, LBPH, Jetson Nano, and Google Sheets for attendance logging.*

---

## ⭐ Overview
This project performs **real-time face detection**, **face recognition**, and automatically logs attendance to **Google Sheets** using a secure Google Service Account.

The system uses:

- **Jetson Nano**
- **USB Camera**
- **OpenCV + LBPH Recognizer**
- **GStreamer**
- **Google Sheets API**

It also includes a **camera capture loop script** that can be used to **automatically collect training images** for each person.

---

# 🚀 Jetson Nano Setup (Must Read First)

Before installing or running this project:

## 1️⃣ Flash the Jetson Nano Using NVIDIA SDK Manager
Flash JetPack & OS using NVIDIA’s SDK Manager tool.

## 2️⃣ After Flashing, Follow the Startup Guide
📄 **Getting Started Guide (Google Docs)**
👉 https://docs.google.com/document/d/1pjvb5xHp7uPkiiE9mQ_X6I_RjF071LiR7rjuKqJBzgo/edit?usp=sharing

This guide covers:

- First boot
- System configuration
- Python & OpenCV setup
- Camera testing
- Jetson optimizations

**This guide MUST be completed before running this project.**

---

# 📂 Project Structure

```
project/
│── faces/                      # Face dataset folders
│── face_model.yml              # Trained LBPH model
│── facdec_recognitionScript.py # Main attendance system script
│── capture_loop.py             # Optional: image collection script for dataset
│── facedetectionattendece-e013eda423c6_googleSheetAccesskey.json # Service account key (DO NOT UPLOAD TO GITHUB)
│── README.md                   # Documentation
```

---

# 📸 Using the Camera Capture Loop Script

This project includes (or recommends using) a **camera capture loop script** that:

✔ Captures an image every **300 ms**  
✔ Saves it automatically in a folder  
✔ Helps build high-quality datasets for each person  
✔ Great for collecting **10–50 training images** per subject  
✔ Improves accuracy of the LBPH model  

Typical dataset structure:

```
faces/
   ├── Ali/
   ├── Ahmed/
   └── Sara/
```

Run the capture script → look at the camera → images are saved automatically.

---

# 🧠 How the Attendance System Works

## 1. Face Detection
Uses OpenCV Haarcascade (`haarcascade_frontalface_default.xml`) to locate faces in each frame.

## 2. Face Recognition
Uses **LBPH Recognizer** because it is:

- Fast
- Lightweight
- Works well on Jetson Nano
- Does not need GPU

## 3. Attendance Logging
When a face is recognized:

- **Column B → Name**
- **Column C → Timestamp**

are appended to the Google Sheet.

Column A is intentionally left empty for future indexing.

## 4. One Attendance per Person
Each person is logged **one time per session**, preventing duplicates.

## 5. GStreamer Camera Pipeline
Jetson Nano works best with this stable pipeline:

```
v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink
```

---

# ⚙️ Installation Guide (For Anyone Downloading the Project)

## 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-repo>/smart-cctv-attendance-jetson.git
cd smart-cctv-attendance-jetson
```

## 2️⃣ Install Dependencies
```bash
sudo apt-get update
sudo apt-get install -y python3-opencv python3-pip
pip3 install --upgrade pip
pip3 install gspread oauth2client numpy
```

_Note:_ On Jetson Nano, it may be preferable to install OpenCV via JetPack or use prebuilt packages for best performance.

## 3️⃣ Enable Required Google APIs

Enable in Google Cloud Console:

- **Google Sheets API**
- **Google Drive API**

## 4️⃣ Create a Google Service Account
Download the JSON key and save it as:

```
facedetectionattendece-e013eda423c6_googleSheetAccesskey.json
```

Place it in the project directory.

## 5️⃣ Share Your Google Sheet with the Service Account Email
Example:

```
facedetectionattendecejetsonna@facedetectionattendece.iam.gserviceaccount.com
```

Give **Editor** access.

## 6️⃣ Prepare Face Dataset
Create a folder per person:

```
faces/Ali
faces/Ahmed
faces/John
```

Use the **camera capture loop script** to automatically collect images.

## 7️⃣ Train LBPH Face Model
The training script will generate:

```
face_model.yml
```

Place it in the project folder.

A simple training script example can be provided on request.

## 8️⃣ Set Your Google Sheet ID
Extract from your Sheet URL:

```
https://docs.google.com/spreadsheets/d/<YOUR_SHEET_ID>/edit
```

Insert into the script:

```python
sheet = client.open_by_key("YOUR_SHEET_ID").sheet1
```

---

# ▶️ Running the System

```bash
python3 facdec_recognitionScript.py
```

Press **Q** to exit.

---

# 🧪 Testing the System

When the camera recognizes a face:

- A rectangle appears around the face
- The person's name is displayed
- Attendance is uploaded instantly to Google Sheets

Check the Google Sheet to confirm entries in:

- **Column B → Name**
- **Column C → Timestamp**

---

# 🛠 Troubleshooting

### ❌ Camera Not Detected
```bash
ls /dev/video*
```
Ensure webcam is connected and device node exists. If using a Jetson CSI camera, update the GStreamer pipeline accordingly.

### ❌ SpreadsheetNotFound
- Wrong sheet ID
- Sheet not shared with service account

### ❌ API Permission Error
Enable both:

- Google Sheets API
- Google Drive API

Also ensure service account JSON matches the project that has APIs enabled.

### ❌ Recognition Accuracy Low
- Add more training images
- Use the capture loop script
- Improve lighting and camera angle

---

# 👨‍💻 Author

**Hamza Khan**
🔗 LinkedIn: https://pk.linkedin.com/in/hamza-khan-rajput

---

# 📜 License
MIT License — free for personal and commercial use.

---

# 🤝 Contributing
Pull requests and suggestions are welcome!

---

# ⚙️ Optional Enhancements (Available on request)

- Auto-training script
- Auto-start on boot (systemd service)
- Google Drive image upload
- Multi-sheet logging
- Daily attendance limitation
- Snapshot images saved to Google Drive with links in the sheet

---
