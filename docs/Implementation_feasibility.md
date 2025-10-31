# Step 5: Implementation Feasibility for Jetson Nano 4GB (10-12 Weeks)

## Project Sub-Topic Area
**Smart CCTV Attendance System with Tailgating Detection**

---

## Most Suitable Techniques from Literature Survey

### (a) Technique 1: CenterFace for Face Detection
**Paper:** CenterFace: Joint Face Detection and Alignment Using Face as Point [9]  
**Link:** https://onlinelibrary.wiley.com/doi/10.1155/2020/7845384  
**Alternative Link:** https://arxiv.org/abs/1911.03599

**Why Selected:**
- Anchor-free, one-stage design specifically optimized for edge devices with limited memory
- Achieves real-time performance even on single CPU core
- Simultaneously provides face detection AND facial landmark alignment (crucial for face recognition accuracy)
- Training time reduced from 5 days to 1 day compared to alternatives
- WIDER FACE benchmark scores: 93.5% (Easy), 92.4% (Medium), 87.5% (Hard)
- Proven successful deployment on embedded systems

---

### (b) Technique 2: MobileFaceNet with ArcFace Loss for Face Recognition
**Paper:** Face Recognition-Based Mass Attendance Using YOLOv5 and ArcFace [2]  
**Link:** https://doi.org/10.1007/978-3-031-34619-4_39

**Supporting Paper:** EdgeFace: Efficient Face Recognition Model for Edge Devices [10]  
**Link:** https://ieeexplore.ieee.org/document/10388036/

**Why Selected:**
- MobileFaceNet has <1M parameters, perfect for 4GB memory constraint
- 18-24ms inference time on mobile CPUs
- Paper [2] validates YOLOv5/MobileFaceNet + ArcFace combination for mass attendance
- ArcFace loss function improves discriminative power for face embeddings
- EdgeFace [10] shows even better results (99.73% LFW accuracy with 1.77M parameters) as potential upgrade path
- Explicitly designed for resource-constrained mobile/embedded deployment

---

### (c) Technique 3: Three-Region Field-of-View Counting for Tailgating Detection
**Paper:** Real-Time AI-Driven People Tracking and Counting Using Overhead Cameras [4]  
**Link:** https://arxiv.org/abs/2411.10072

**Supporting Paper:** Piggybacking Detection Based on Coupled Body-Feet Recognition at Entrance Control [3]  
**Link:** https://doi.org/10.1007/978-3-030-33904-3_74

**Why Selected:**
- Achieves 97% accuracy with 20-27 FPS on low-power edge devices
- Three-region algorithm (entry zone / middle zone / exit zone) provides robust counting validation
- Uses SSD MobileNet (lightweight) for head detection
- Custom feature-based tracking avoids complex deep learning tracking models
- Paper [3] provides validation that geometric/vision-based approach works for piggybacking detection
- Computational efficiency suitable for concurrent operation with face recognition

---

## Feasibility Assessment for NVIDIA Jetson Nano 4GB

### Hardware Specifications Review
- **Jetson Nano B01 (with Waveshare JetBot):** 
  - 4GB LPDDR4 RAM
  - 128-core Maxwell GPU
  - Quad-core ARM A57 @ 1.43 GHz
  - 5-10W power consumption
  - Integrated camera module

### Phase 1: Face Recognition Attendance System (Weeks 1-6)

#### **FEASIBILITY: HIGH (90%)**

**Implementation Plan:**
1. **Face Detection (CenterFace):**
   - Use pre-trained CenterFace model from official repository
   - Fine-tune on student dataset (50-100 faces, 20-30 images per person)
   - Convert to ONNX format for TensorRT optimization
   - Expected: 20-25 FPS at 640×480 resolution

2. **Face Recognition (MobileFaceNet + ArcFace):**
   - Train MobileFaceNet with ArcFace loss on student dataset
   - Use Google Colab with free GPU for training (3-4 days)
   - Export model and optimize with TensorRT (FP16 precision)
   - Expected: 95-98% recognition accuracy based on Papers [1][2]

3. **TensorRT Optimization (Based on Papers [6][7]):**
   - Framework conversion: PyTorch → ONNX → TensorRT
   - Apply FP16 precision (reduces memory by 50%, minimal accuracy loss)
   - Layer fusion and kernel tuning via jetson-inference library
   - Expected speedup: 100-200% based on Paper [6] benchmarks

**Timeline:**
- Weeks 1-2: Dataset collection, hardware setup, camera calibration
- Weeks 3-4: Model training and optimization
- Weeks 5-6: Integration, testing, and debugging

**Resources Required:**
- ✓ Jetson Nano B01 with JetBot (provided)
- ✓ Camera module (integrated with JetBot)
- **NEED:** SanDisk microSD card 64GB UHS-I (Class 10) - ~$15-20
- **NEED:** Power supply 5V 4A barrel connector - ~$10-15 (if not included)
- **NEED:** WiFi/Ethernet connectivity for remote development
- **OPTIONAL:** USB keyboard, mouse, HDMI monitor for initial setup

**Potential Challenges:**
- Memory constraints during concurrent model inference → Solution: Sequential processing with buffering (Paper [6] methodology)
- Varying lighting conditions in deployment environment → Solution: Data augmentation during training with brightness/contrast variations
- Real-time latency requirements → Solution: TensorRT FP16 optimization reduces inference time by 40-60%

**Risk Mitigation:**
- Backup plan: Use pre-trained RetinaFace + ArcFace models if custom training fails
- Simplified approach: Use only face recognition for Phase 1 before attempting Phase 2

---

### Phase 2: Tailgating Detection (Weeks 7-12)

#### **FEASIBILITY: MODERATE (75%)**

**Implementation Plan:**
1. **Head Detection & Counting:**
   - Deploy lightweight SSD MobileNet V1 for head/person detection
   - Implement three-region field-of-view algorithm (Paper [4]):
     * Entry zone: Person enters frame
     * Middle zone: Person transitions through doorway
     * Exit zone: Person leaves frame
   - Count transitions through all three zones

2. **Tracking & Validation:**
   - Implement simple centroid-based tracking (less complex than DeepSORT)
   - Compare: (Number of tracked heads) vs. (Number of recognized faces)
   - Trigger tailgating alert if: `head_count > face_count` within 3-second window

3. **Integration:**
   - Run face recognition and head counting in parallel using multi-threading
   - Use inference pipelining from Paper [6] to balance GPU/CPU workload
   - Expected: 18-22 FPS with both systems running concurrently

**Timeline:**
- Weeks 7-8: Head detection model integration and testing
- Weeks 9-10: Tracking algorithm development and calibration
- Weeks 11-12: Full system integration, optimization, and validation

**Resources Required:**
- ✓ All Phase 1 resources
- **NEED:** Calibration setup (measuring tape, markers) for camera positioning
- **OPTIONAL:** Secondary USB camera for dual-angle coverage (improves robustness)

**Potential Challenges:**
- Occlusion when multiple people enter simultaneously → Solution: Three-region algorithm handles overlapping entries
- False positives from people walking closely → Solution: Tunable threshold with configurable sensitivity
- Depth/distance estimation without stereo camera → Solution: Use head size variance as proxy for distance (Paper [5] approach)

**Risk Mitigation:**
- **Simplified approach (if needed):** Skip three-region algorithm, use simple person counter
  - If `person_count > recognized_face_count` → Flag as potential tailgating
  - Reduces complexity significantly, still achieves 85-90% detection
- **Fallback plan:** Implement only Phase 1 robustly, demonstrate Phase 2 as proof-of-concept

---

## Resource Summary (Beyond Jetson Nano Kit)

### Essential Items (Must Purchase):
1. **SanDisk Extreme 64GB microSD Card (UHS-I, U3, V30)** - $18-22
   - Required for JetPack OS, models, datasets, logs
   - Minimum 32GB, recommend 64GB for comfort

2. **5V 4A Power Supply with Barrel Connector** - $12-18
   - Required for Jetson Nano stable operation
   - Must provide sufficient current for GPU usage

### Recommended Items:
3. **USB Keyboard + Mouse + HDMI Monitor** - $30-50 (or use existing)
   - For initial setup and debugging
   - Can use headless SSH after initial configuration

4. **USB to Ethernet Adapter** (if no WiFi dongle) - $10-15
   - Stable network connection for model downloads and remote development

5. **Small Cooling Fan** - $5-10
   - Paper [7] recommends additional cooling for continuous operation
   - Prevents thermal throttling during model training/inference

### Optional Items (Nice to Have):
6. **Portable Power Bank (20,000mAh with 5V 3A output)** - $25-35
   - Enables mobile/demo deployment without wall power

7. **Camera Tripod/Mount** - $15-25
   - Stable positioning for consistent testing and demos

8. **Anti-static Wrist Strap** - $5-8
   - Safety measure mentioned in project guidebook

**Total Essential Budget:** ~$30-40  
**Total Recommended Budget:** ~$80-120

---

## Team Workload Distribution (3-4 Students, Part-time)

### Student 1: Embedded Systems (Ali Tahir)
- Hardware setup, JetPack installation, camera calibration
- TensorRT optimization and model deployment
- Performance benchmarking (FPS, latency, memory usage)
- Power consumption analysis
- **Weekly Commitment:** 10-12 hours

### Student 2: Simulation & Algorithms (Hamza)
- Face detection model (CenterFace) integration and testing
- Tailgating detection algorithm development (three-region counting)
- Head tracking implementation
- Testing on simulation/recorded videos before hardware deployment
- **Weekly Commitment:** 10-12 hours

### Student 3: Research & Documentation (Umbreen)
- Dataset collection and organization
- Face recognition model training (MobileFaceNet + ArcFace)
- Database management (student face embeddings)
- Literature review maintenance, report writing, GitHub documentation
- **Weekly Commitment:** 10-12 hours

**Total Weekly Commitment:** 30-36 hours (manageable for part-time students)

---

## Expected Performance Metrics

### Face Recognition Attendance (Phase 1):
- **Accuracy:** 95-98% (based on Papers [1][2])
- **Speed:** 20-25 FPS at 640×480 resolution
- **False Positive Rate:** <2%
- **False Negative Rate:** <3%
- **Power Consumption:** 5-8W average

### Tailgating Detection (Phase 2):
- **Detection Accuracy:** 90-93% (based on Paper [3] 92.9% benchmark)
- **False Alarm Rate:** 5-10% (acceptable for prototype)
- **Processing Latency:** <100ms per frame
- **Concurrent Operation:** 18-22 FPS with both systems active

### System Integration:
- **Startup Time:** <30 seconds
- **Memory Usage:** 3.2-3.6 GB (leaving 400-800 MB headroom)
- **Storage Requirements:** 8-12 GB (OS, models, datasets, logs)

---

## Critical Success Factors

1. **Early Validation (Week 1):**
   - Verify JetBot camera compatibility with JetPack
   - Test basic OpenCV capture pipeline before proceeding
   - Confirm GPIO/sensor access for potential future expansions

2. **Incremental Testing (Every 2 weeks):**
   - Don't wait until final integration to test components
   - Maintain separate test scripts for each module
   - Document issues in GitHub project board immediately

3. **Dataset Quality (Weeks 2-3):**
   - Minimum 20 images per student across varied conditions:
     * Different lighting (bright, dim, natural, artificial)
     * Multiple angles (straight-on, ±30° horizontal, ±15° vertical)
     * Different expressions (neutral, smiling)
     * With/without glasses, different hairstyles
   - Poor dataset = poor recognition, regardless of model quality

4. **TensorRT Optimization Priority (Week 5):**
   - This is NON-NEGOTIABLE for real-time performance
   - Paper [6] shows 101-680% improvement - essential for Jetson Nano
   - Allocate 1 full week for optimization, don't rush this step

5. **Fallback Plan (Week 8 Decision Point):**
   - Assess Phase 1 completion status
   - If behind schedule: Simplify Phase 2 to basic person counting
   - Better to have robust Phase 1 + simple Phase 2 than incomplete both

---

## Comparison to Literature Benchmarks

Your project scope aligns well with:

- **Paper [7]'s implementation:** Jetson Nano B01, dual tasks, TensorRT optimization, successfully completed by university research team
- **Papers [1][2]:** Similar attendance systems completed in academic settings with comparable resources
- **Paper [4]:** Demonstrates edge device feasibility for doorway monitoring (your tailgating use case)

**Key Difference:** Your project combines two separate systems (attendance + tailgating) which increases complexity by ~30%. However, the two-phase approach (attendance first, then tailgating) manages this risk effectively.

---

## Final Feasibility Assessment

### **Phase 1 (Face Recognition Attendance): 90% FEASIBLE**
✓ Well-documented approaches in literature  
✓ Proven Jetson Nano implementations exist  
✓ Pre-trained models available  
✓ Manageable scope for 6 weeks

### **Phase 2 (Tailgating Detection): 75% FEASIBLE**
✓ Papers [3][4] provide clear implementation roadmap  
✓ Simplified approach (person counting) increases to 90% feasibility  
⚠ Geometric depth estimation may require additional sensors/calibration  
⚠ Concurrent processing adds memory pressure

### **Overall Project: 85% FEASIBLE**
**Confidence Level:** High for delivering functional prototype within 10-12 week semester timeline

**Recommendation:** Proceed with implementation following the two-phase incremental approach. Allocate 60% effort to Phase 1 (weeks 1-6) and 40% to Phase 2 (weeks 7-12). Maintain weekly progress meetings and be prepared to simplify Phase 2 if time constraints emerge.

---

## Next Steps (Immediate Actions)

1. **Week 1 Tasks:**
   - Order essential hardware (microSD card, power supply)
   - Set up Jetson Nano with JetPack 4.6 or later
   - Install OpenCV, TensorRT, PyTorch
   - Test camera capture pipeline

2. **GitHub Setup:**
   - Create repository: `smart-cctv-attendance-jetson`
   - Add instructor and lab engineer as collaborators
   - Create project board with milestones
   - Upload this literature review to `/docs/literature_review.md`

3. **Team Coordination:**
   - Schedule weekly sync meeting (1 hour)
   - Set up group communication (WhatsApp/Slack)
   - Divide dataset collection responsibilities
   - Establish code review process

4. **Instructor Consultation:**
   - Share this feasibility assessment
   - Get approval for simplified Phase 2 approach if needed
   - Confirm hardware budget/procurement process
   - Clarify demo requirements for Week 19 presentation