# SMART CCTV ATTENDANCE SYSTEM WITH TAILGATING DETECTION

The primary motivation of this project is to develop an integrated solution aimed at solving issues like proxy attendance and tailgating attacks by combining face recognition-based attendance tracking with smart surveillance. The system will use advanced computer vision and deep learning algorithms to automatically detect, identify, and track individuals, while simultaneously scanning entry points for unauthorized access.

The implementation will be done on a resource-constrained edge device platform, specifically the **NVIDIA Jetson Nano 4GB development kit**, to meet real-time performance and scalability requirements. The project is structured into two main phases: Smart Attendance and Tailgating Detection.

---

## Setup Instructions

This system is built for deployment on an edge device.

1.  **Clone this repository:** `git clone https://github.com/alit2204/smart-cctv-attendance-jetson`
2.  **Hardware Required:** Jetson Nano B01 development kit (4 GB LPDDR4 RAM) and an Integrated CSI camera module for live video input.
3.  **Install Dependencies:** Configure the Jetson Nano and install **JetPack**, **TensorRT**, and **OpenCV**.
4.  **Code Access:** Open `/src` for simulation and embedded source code files.

---

## Folder Structure

The repository maintains the required organized structure for project elements:

* `/docs` – reports, meeting minutes, references, and diagrams
* `/src` – Simulation and embedded source code
* `/data` – datasets
* `/results` – graphs, figures, and logs

---

## Team Roles

| Member | CMS | Role | Primary Responsibilities |
| :--- | :--- | :--- | :--- |
| **Umbreen Shah Nawaz** | 00000537511 | Documentation Life Cycle | Manage Project Schedule and GitHub updates, preparation of reports/presentations, and overall coordination. |
| **Muhammad Ali Tahir** | 00000450271 | Simulation and Algorithm | Develop face recognition/tailgating detection algorithms, train and test models, and optimize/integrate for real-time processing. |
| **Hamza Khan** | 00000505212 | Embedded System | Deploy optimized models on NVIDIA Jetson Nano, configure hardware/camera, and ensure efficient on-device processing. |
