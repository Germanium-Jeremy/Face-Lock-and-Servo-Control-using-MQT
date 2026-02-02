# Face Recognition & Tracking System - Complete Run Guide

This guide walks you through running the face recognition & tracking system step-by-step, following the methodology described in the ResearchGate paper: **"Face Recognition with ArcFace ONNX and 5-Point Alignment"**.

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.8 or higher installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] ArcFace ONNX model downloaded to `models/embedder_arcface.onnx`
- [ ] MediaPipe face landmarker file at `face_landmarker.task`
- [ ] A working webcam/camera connected
- [ ] Virtual environment activated (if using one)

---

## Step-by-Step Execution Guide

### **STEP 1: Verify System Setup**

First, verify that all components are working correctly by testing each stage of the pipeline.

#### 1.1 Test Camera Access

```bash
python -m src.camera
```

**What to expect:**
- Camera feed should appear in a window
- Press `q` to quit

**If camera doesn't open:**
- Try changing camera index in the code (0, 1, or 2)
- Close other applications using the camera
- Check camera permissions

---

#### 1.2 Test Face Detection (Haar Cascade)

```bash
python -m src.detect
```

**What to expect:**
- Green rectangles should appear around detected faces
- Press `q` to quit

**Verification:**
- ✅ Face detection working if you see green boxes around faces
- ❌ If no boxes appear, check lighting and face visibility

---

#### 1.3 Test 5-Point Landmark Detection (MediaPipe)

```bash
python -m src.landmarks
```

**What to expect:**
- 5 green dots should appear on facial landmarks:
  - Left eye
  - Right eye
  - Nose tip
  - Left mouth corner
  - Right mouth corner
- Press `q` to quit

**Verification:**
- ✅ Landmarks working if you see 5 green dots on your face
- ❌ If dots don't appear, ensure face is clearly visible and well-lit

---

#### 1.4 Test Face Alignment (5-Point to 112×112)

```bash
python -m src.align
```

**What to expect:**
- Main window: Camera feed with face detection and landmarks
- **NEW:** Bounding box smoothly tracks face movement (face tracking enabled)
- Second window: Aligned 112×112 face crop
- Press `q` to quit
- Press `s` to save an aligned face crop to `data/debug_aligned/`

**Verification:**
- ✅ Alignment working if you see a properly aligned face in the second window
- The aligned face should be centered with eyes, nose, and mouth in standard positions
- ✅ **Tracking working:** Bounding box should smoothly follow your face as you move
- ❌ If alignment looks wrong, check that landmarks are detected correctly

---

#### 1.5 Test Embedding Generation (ArcFace ONNX)

**⚠️ IMPORTANT:** This step requires the ArcFace ONNX model to be present.

```bash
python -m src.embed
```

**What to expect:**
- Main window: Camera feed with face detection
- Embedding heatmap visualization
- Embedding statistics displayed
- Press `q` to quit
- Press `p` to print embedding details to console

**Verification:**
- ✅ Embedding working if you see the heatmap and statistics
- ❌ If you get "Model file not found", download the ArcFace ONNX model (see `MODEL_DOWNLOAD.md`)

**Expected output in console (when pressing 'p'):**
```
[embedding]
 dim: 512
 min/max: -0.123 / 0.456
 first10: [0.023 -0.045 0.123 ...]
```

---

### **STEP 2: Enroll Identities**

Now that all components are verified, you can enroll people into the system.

#### 2.1 Start Enrollment

```bash
python -m src.enroll
```

**Process:**

1. **Enter person's name** when prompted (e.g., "Alice", "Bob", "Charlie")
   - Use clear, unique names
   - Avoid special characters

2. **Position the person** in front of the camera
   - Ensure good, consistent lighting
   - Face should be clearly visible
   - Person should look at the camera

3. **Capture samples:**
   - **SPACE**: Capture one sample manually (when face is detected)
   - **a**: Toggle auto-capture mode (captures every 0.25 seconds)
   - **s**: Save enrollment (after collecting enough samples)
   - **r**: Reset NEW samples (keeps existing crops on disk)
   - **q**: Quit without saving

4. **Collect multiple samples:**
   - **Minimum:** 15 samples per person (recommended)
   - **Best practice:** 20-30 samples with variations:
     - Slight head turns (left/right)
     - Different expressions (neutral, smile)
     - Slight angle variations
   - Each sample is automatically aligned to 112×112

5. **Save enrollment:**
   - Press `s` when you have enough samples
   - The system will:
     - Generate embeddings for all samples
     - Compute mean embedding (template)
     - L2-normalize the template
     - Save to database

**What gets saved:**
- Aligned face crops: `data/enroll/<name>/*.jpg`
- Face database: `data/db/face_db.npz` (embeddings)
- Metadata: `data/db/face_db.json` (information about the database)

**Tips for best results:**
- ✅ Use stable, consistent lighting
- ✅ Vary poses slightly during enrollment
- ✅ Ensure face is clearly visible in all samples
- ✅ Collect samples over 10-15 seconds
- ✅ **NEW:** Bounding box will smoothly track your face movement during enrollment
- ❌ Avoid extreme angles or poor lighting
- ❌ Don't rush - quality over quantity

---

#### 2.2 Enroll Multiple Identities

**Repeat Step 2.1 for each person you want to recognize.**

**Requirements:**
- Enroll at least **10 different identities** (as per project requirements)
- Each identity should have **15+ samples**
- Use unique names for each person

**Example workflow:**
```bash
# Person 1
python -m src.enroll
# Enter: Alice
# Capture 20 samples, press 's' to save

# Person 2
python -m src.enroll
# Enter: Bob
# Capture 20 samples, press 's' to save

# ... repeat for all 10+ people
```

**Re-enrollment:**
- If you run enrollment again with the same name, existing crops are loaded
- New samples are added to existing ones
- Template is recomputed from all samples

---

### **STEP 3: Evaluate Threshold**

Before running live recognition, you must determine the optimal threshold for matching.

#### 3.1 Run Threshold Evaluation

```bash
python -m src.evaluate
```

**What this does:**
- Loads all enrollment crops from `data/enroll/`
- Generates embeddings for each crop
- Computes pairwise distances:
  - **Genuine pairs:** Different samples of the same person
  - **Impostor pairs:** Samples from different people
- Analyzes distance distributions
- Suggests optimal threshold based on target FAR (False Acceptance Rate)

**Expected output:**
```
=== Distance Distributions (cosine distance = 1 - cosine similarity) ===
Genuine (same person): n=450 mean=0.234 std=0.045 p05=0.156 p50=0.231 p95=0.312
Impostor (diff persons): n=3200 mean=0.678 std=0.123 p05=0.512 p50=0.671 p95=0.845

=== Threshold Sweep ===
thr=0.10 FAR= 0.00% FRR=45.23%
thr=0.20 FAR= 0.15% FRR=12.34%
thr=0.30 FAR= 0.85% FRR= 3.45%
thr=0.34 FAR= 1.00% FRR= 2.30%
thr=0.40 FAR= 2.15% FRR= 1.12%
...

Suggested threshold (target FAR 1.0%): thr=0.34 FAR=1.00% FRR=2.30%

(Equivalent cosine similarity threshold ~ 0.660, since sim = 1 - dist)
```

**Understanding the output:**
- **Genuine distances:** Lower values = same person samples are similar (good!)
- **Impostor distances:** Higher values = different people are dissimilar (good!)
- **FAR (False Acceptance Rate):** Percentage of impostors incorrectly accepted
- **FRR (False Rejection Rate):** Percentage of genuine users incorrectly rejected
- **Suggested threshold:** Balance between FAR and FRR

**⚠️ IMPORTANT:** 
- **Note the suggested threshold value** (e.g., 0.34)
- You will use this value in live recognition
- **Do not guess the threshold** - always run evaluation first!

---

### **STEP 4: Live Face Recognition**

Now you're ready to run live recognition!

#### 4.1 Start Recognition

```bash
python -m src.recognise
```

**What to expect:**
- Camera feed with face detection
- For each detected face:
  - **Green box + name:** Recognized identity (distance below threshold)
  - **Red box + "Unknown":** Unrecognized face (distance above threshold)
  - Distance and similarity scores displayed
  - Aligned face thumbnails shown on the right side

**Controls:**
- **q**: Quit
- **r**: Reload database from disk (useful after enrolling new people)
- **+** or **=**: Increase threshold (more permissive, accepts more matches)
- **-**: Decrease threshold (more strict, rejects more matches)
- **d**: Toggle debug overlay (shows additional information)
- **t**: Toggle face tracking ON/OFF (enabled by default)

**Face Tracking Feature:**
- **Bounding boxes smoothly track faces** as they move across the frame
- Each face gets a **track ID** that persists across frames
- Tracking helps maintain identity associations even when detection briefly fails
- **Track IDs** are displayed on each bounding box
- **Velocity prediction** helps track faces through occlusions

**Understanding the display:**
- **Distance:** Cosine distance (lower = more similar)
  - Below threshold = match
  - Above threshold = no match
- **Similarity:** Cosine similarity (higher = more similar)
  - Range: -1.0 to 1.0
  - Typical matches: 0.65-0.95

**Adjusting threshold on the fly:**
- If too many false positives (wrong people recognized):
  - Press `-` to decrease threshold (more strict)
- If too many false negatives (known people not recognized):
  - Press `+` to increase threshold (more permissive)
- Start with the threshold suggested by `evaluate.py`

---

## Complete Workflow Summary

Here's the complete workflow from start to finish:

```bash
# 1. Setup and Verification (one-time)
python -m src.camera          # Test camera
python -m src.detect          # Test detection
python -m src.landmarks       # Test landmarks
python -m src.align           # Test alignment
python -m src.embed           # Test embedding

# 2. Enrollment (repeat for each person)
python -m src.enroll          # Enroll Person 1
python -m src.enroll          # Enroll Person 2
# ... enroll 10+ people

# 3. Threshold Evaluation (once, after all enrollments)
python -m src.evaluate        # Get optimal threshold

# 4. Live Recognition (whenever you want to recognize faces)
python -m src.recognise       # Start recognition
```

---

## Troubleshooting

### Issue: "Model file not found"

**Solution:**
1. Download the ArcFace ONNX model (see `MODEL_DOWNLOAD.md`)
2. Place it at: `models/embedder_arcface.onnx`
3. Verify the file exists and is not corrupted

---

### Issue: Camera not opening

**Solution:**
1. Try different camera indices:
   - Edit the code: change `VideoCapture(0)` to `VideoCapture(1)` or `VideoCapture(2)`
2. Close other applications using the camera
3. Check camera permissions (especially on Linux/Mac)
4. On Linux: `sudo usermod -a -G video $USER` (then logout/login)

---

### Issue: No face detected

**Solution:**
1. Ensure good lighting (face should be clearly visible)
2. Face should not be too far from camera
3. Remove obstructions (glasses, masks, etc.)
4. Try adjusting `min_size` parameter in detection code if needed

---

### Issue: Poor recognition accuracy

**Solution:**
1. **Enroll more samples:** 15+ samples per person (20-30 recommended)
2. **Vary enrollment conditions:**
   - Different poses (slight left/right turns)
   - Different expressions
   - Slight angle variations
3. **Run threshold evaluation:** Use the suggested threshold from `evaluate.py`
4. **Check alignment:** Use `align.py` to verify faces are properly aligned
5. **Improve lighting:** Use consistent, good lighting during enrollment and recognition

---

### Issue: "Unknown" for known people

**Solution:**
1. **Increase threshold:** Press `+` in recognition window
2. **Re-enroll with more samples:** Run `enroll.py` again with the same name
3. **Check lighting:** Ensure recognition lighting matches enrollment lighting
4. **Verify alignment:** Check that faces are being aligned correctly

---

### Issue: Wrong person recognized

**Solution:**
1. **Decrease threshold:** Press `-` in recognition window
2. **Re-run evaluation:** Run `evaluate.py` again to get a better threshold
3. **Improve enrollment:** Enroll more varied samples for each person
4. **Check database:** Press `r` to reload database if you just enrolled someone

---

### Issue: Import errors

**Solution:**
1. Ensure virtual environment is activated
2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```
3. Check Python version: `python --version` (should be 3.8+)
4. Verify all packages in `requirements.txt` are installed

---

## Verification Checklist

Before considering the system complete, verify:

- [ ] All pipeline stages work (camera → detect → landmarks → align → embed)
- [ ] At least 10 identities enrolled
- [ ] Each identity has 15+ samples
- [ ] Threshold evaluation completed
- [ ] Suggested threshold noted
- [ ] Live recognition works with reasonable accuracy
- [ ] Known faces are recognized correctly
- [ ] Unknown faces are properly rejected
- [ ] All files in correct directory structure

---

## Quick Reference: Keyboard Controls

### Enrollment (`src/enroll.py`)
- **SPACE**: Capture one sample
- **a**: Toggle auto-capture mode
- **s**: Save enrollment
- **r**: Reset NEW samples (keeps existing)
- **q**: Quit
- **Note:** Face tracking is enabled - bounding box smoothly follows face movement

### Recognition (`src/recognise.py`)
- **q**: Quit
- **r**: Reload database
- **+** or **=**: Increase threshold
- **-**: Decrease threshold
- **d**: Toggle debug overlay
- **t**: Toggle face tracking ON/OFF (enabled by default)

### Alignment (`src/align.py`)
- **q**: Quit
- **s**: Save aligned face crop

### Embedding (`src/embed.py`)
- **q**: Quit
- **p**: Print embedding statistics

---

## File Structure Reference

After running the system, you should have:

```
FaceRecognition/
├── models/
│   └── embedder_arcface.onnx    # ArcFace model (you download this)
├── face_landmarker.task          # MediaPipe model (usually included)
├── data/
│   ├── enroll/                   # Enrollment crops
│   │   ├── Alice/
│   │   │   ├── 1234567890.jpg
│   │   │   └── ...
│   │   ├── Bob/
│   │   └── ...
│   ├── db/                       # Face database
│   │   ├── face_db.npz          # Embeddings (binary)
│   │   └── face_db.json          # Metadata
│   └── debug_aligned/            # Debug aligned crops
└── src/                          # Source code
    ├── camera.py
    ├── detect.py
    ├── landmarks.py
    ├── align.py
    ├── embed.py
    ├── enroll.py
    ├── recognise.py
    ├── evaluate.py
    └── haar_5pt.py
```

---

## Next Steps

1. **Experiment with thresholds:** Try different threshold values to see the trade-off between FAR and FRR
2. **Test with different conditions:** Try recognition in different lighting, angles, and distances
3. **Improve enrollment:** Collect more varied samples for better accuracy
4. **Analyze performance:** Use `evaluate.py` to understand your system's performance characteristics

---

## Getting Help

If you encounter issues:

1. **Check error messages carefully** - they often contain helpful information
2. **Review code comments** in each module for implementation details
3. **Verify all dependencies** are installed correctly
4. **Ensure model files** are present and valid
5. **Check the troubleshooting section** above
6. **Review README.md** for detailed documentation

---

**Good luck with your face recognition system!** 🎯

For more details, see:
- `README.md` - Complete project documentation
- `QUICKSTART.md` - Quick start guide
- `MODEL_DOWNLOAD.md` - Model download instructions
