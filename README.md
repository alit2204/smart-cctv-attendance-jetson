# SMART CCTV ATTENDANCE SYSTEM WITH TAILGATING DETECTION

[cite_start]The primary motivation of this project is to develop an integrated solution aimed at solving issues like proxy attendance and tailgating attacks by combining face recognition-based attendance tracking with smart surveillance[cite: 30, 31]. [cite_start]The system will use advanced computer vision and deep learning algorithms to automatically detect, identify, and track individuals, while simultaneously scanning entry points for unauthorized access[cite: 32].

[cite_start]The implementation will be done on a resource-constrained edge device platform, specifically the **NVIDIA Jetson Nano 4GB development kit**, to meet real-time performance and scalability requirements[cite: 33, 76]. [cite_start]The project is structured into two main phases: Smart Attendance and Tailgating Detection[cite: 85, 86].

---

## 🛠️ Setup Instructions

This system is built for deployment on an edge device.

1.  [cite_start]**Clone this repository:** `git clone https://github.com/alit2204/smart-cctv-attendance-jetson` [cite: 117]
2.  [cite_start]**Hardware Required:** Jetson Nano B01 development kit (4 GB LPDDR4 RAM) and an Integrated CSI camera module for live video input[cite: 90, 94, 103].
3.  [cite_start]**Install Dependencies:** Configure the Jetson Nano and install **JetPack**, **TensorRT**, and **OpenCV**[cite: 92].
4.  **Code Access:** Open `/src` for simulation and embedded source code files.

---

## 📂 Folder Structure

[cite_start]The repository maintains the required organized structure for project elements[cite: 115]:

* `/docs` – reports, meeting minutes, references, and diagrams
* `/src` – Simulation and embedded source code
* `/data` – datasets
* `/results` – graphs, figures, and logs

---

## 🧑‍💻 Team Roles

| Member | CMS | Role | Primary Responsibilities |
| :--- | :--- | :--- | :--- |
| **Umbreen Shah Nawaz** | [cite_start]00000537511 [cite: 7] | [cite_start]Documentation Life Cycle [cite: 106] | [cite_start]Manage Project Schedule and GitHub updates, preparation of reports/presentations, and overall coordination[cite: 106]. |
| **Muhammad Ali Tahir** | [cite_start]00000450271 [cite: 8] | [cite_start]Simulation and Algorithm [cite: 107] | [cite_start]Develop face recognition/tailgating detection algorithms, train and test models, and optimize/integrate for real-time processing[cite: 107]. |
| **Hamza Khan** | [cite_start]00000505212 [cite: 9] | [cite_start]Embedded System [cite: 107] | [cite_start]Deploy optimized models on NVIDIA Jetson Nano, configure hardware/camera, and ensure efficient on-device processing[cite: 107]. |
