# Face Recognition with ArcFace ONNX and 5-Point Alignment

A complete face recognition system implementing the pipeline described in the ResearchGate publication: [Face Recognition with ArcFace ONNX and 5-Point Alignment](https://www.researchgate.net/publication/399996476_Face_Recognition_with_ArcFace_ONNX_and_5-Point_Alignment)

## Project Overview

This project implements a research-grade face recognition system using:
- **Haar Cascade** for face detection
- **MediaPipe FaceMesh** for 5-point facial landmark detection
- **ArcFace ONNX** for generating face embeddings
- **Cosine similarity** with threshold-based matching for recognition

## Features

- ✅ Real-time face detection and recognition
- ✅ **Face tracking** - Bounding boxes smoothly track faces across frames
- ✅ 5-point facial landmark alignment (112×112 crops)
- ✅ ArcFace embedding generation using ONNX Runtime (CPU-only)
- ✅ Multi-identity enrollment system
- ✅ Threshold evaluation and tuning
- ✅ Live recognition with adjustable thresholds
- ✅ L2-normalized embeddings for optimal matching
- ✅ Persistent track IDs for each face
- ✅ Velocity prediction for robust tracking

## Project Structure

```
FaceRecognition/
├── src/
│   ├── camera.py          # Camera test utility
│   ├── detect.py          # Face detection demo
│   ├── landmarks.py        # 5-point landmark detection demo
│   ├── align.py           # Face alignment demo
│   ├── embed.py           # Embedding generation demo
│   ├── enroll.py          # Identity enrollment tool
│   ├── recognise.py       # Live face recognition
│   ├── evaluate.py        # Threshold evaluation
│   └── haar_5pt.py        # Core detection and alignment module
├── data/
│   ├── enroll/            # Enrollment crops (per identity)
│   ├── db/                # Face database (embeddings)
│   └── debug_aligned/     # Debug aligned face crops
├── models/
│   └── embedder_arcface.onnx  # ArcFace ONNX model (download required)
├── face_landmarker.task   # MediaPipe face landmarker model
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Prerequisites

- Python 3.8+ (recommended: 3.10 or 3.11)
- Webcam/camera
- Windows/Linux/macOS

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd FaceRecognition
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Models

#### ArcFace ONNX Model

The ArcFace embedding model (`embedder_arcface.onnx`) is required but not included in the repository due to size. You need to download it:

**Option 1: Download from InsightFace**
- Visit: https://github.com/deepinsight/insightface
- Look for ArcFace ONNX models (typically named `w600k_r50.onnx` or similar)
- Place the model file in `models/embedder_arcface.onnx`

**Option 2: Use Pre-converted Model**
- Search for "ArcFace ONNX model" or "InsightFace ONNX"
- Common sources: Hugging Face, GitHub releases
- Ensure the model expects 112×112 input and outputs embeddings

**Option 3: Convert from PyTorch/TensorFlow**
- If you have an ArcFace model in another format, convert it to ONNX
- Input shape should be: `(1, 3, 112, 112)` (BGR normalized)
- Output should be a 512-dimensional embedding vector

#### MediaPipe Face Landmarker

The `face_landmarker.task` file should already be in the project root. If missing:
- Download from: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- Place it in the project root directory

## How to Run

### Stage 1: Test Camera

```bash
python -m src.camera
```
Press `q` to quit. Verify your camera works.

### Stage 2: Test Face Detection

```bash
python -m src.detect
```
Press `q` to quit. You should see green rectangles around detected faces.

### Stage 3: Test 5-Point Landmarks

```bash
python -m src.landmarks
```
Press `q` to quit. You should see 5 green dots on facial landmarks.

### Stage 4: Test Alignment

```bash
python -m src.align
```
Press `q` to quit, `s` to save an aligned face crop. The aligned 112×112 face should appear in a separate window.

### Stage 5: Test Embedding Generation

```bash
python -m src.embed
```
Press `q` to quit, `p` to print embedding statistics. You should see the embedding heatmap visualization.

### Stage 6: Enroll Identities

**Important:** You must enroll at least 10 different identities as per project requirements.

```bash
python -m src.enroll
```

**Controls:**
- Enter the person's name when prompted
- **SPACE**: Capture one sample (when face is detected)
- **a**: Toggle auto-capture mode (captures periodically)
- **s**: Save enrollment (after collecting enough samples)
- **r**: Reset NEW samples (keeps existing crops)
- **q**: Quit

**Tips:**
- Collect 15+ samples per person for best results
- Use stable lighting
- Vary poses slightly (left/right, different expressions)
- Each sample is automatically aligned to 112×112

**Output:**
- Aligned crops saved to: `data/enroll/<name>/*.jpg`
- Embeddings saved to: `data/db/face_db.npz`
- Metadata saved to: `data/db/face_db.json`

### Stage 7: Evaluate Threshold

Before running live recognition, evaluate the optimal threshold:

```bash
python -m src.evaluate
```

This script will:
- Load all enrollment crops
- Compute genuine (same person) and impostor (different person) distances
- Suggest an optimal threshold based on FAR (False Acceptance Rate)
- Display distance distribution statistics

**Output example:**
```
=== Distance Distributions ===
Genuine (same person): n=... mean=0.234 std=0.045 ...
Impostor (diff persons): n=... mean=0.678 std=0.123 ...

Suggested threshold (target FAR 1.0%): thr=0.34 FAR=0.85% FRR=2.3%
```

**Note:** Use the suggested threshold value in live recognition. Do not guess!

### Stage 8: Live Recognition

```bash
python -m src.recognise
```

**Controls:**
- **q**: Quit
- **r**: Reload database from disk
- **+** or **=**: Increase threshold (more permissive)
- **-**: Decrease threshold (more strict)
- **d**: Toggle debug overlay
- **t**: Toggle face tracking ON/OFF (enabled by default)

**Display:**
- Green box + label: Recognized identity
- Red box + "Unknown": Unrecognized face
- **Track ID** displayed on each bounding box (when tracking enabled)
- Bounding boxes smoothly track faces as they move
- Distance and similarity scores shown for each face
- Aligned face thumbnails on the right side

## How Enrollment Works

1. **Capture Phase:**
   - Camera captures frames
   - Haar cascade detects faces
   - MediaPipe extracts 5-point landmarks
   - Face is aligned to 112×112 using affine transformation
   - Aligned crop is saved to `data/enroll/<name>/`

2. **Embedding Phase:**
   - Each aligned crop is passed through ArcFace ONNX model
   - Generates a 512-dimensional embedding vector
   - Embedding is L2-normalized

3. **Template Creation:**
   - All embeddings for a person are averaged
   - Mean embedding is L2-normalized again
   - Saved as the person's template in the database

4. **Re-enrollment:**
   - If `data/enroll/<name>/` already exists, existing crops are loaded
   - New captures are added to the existing set
   - Template is recomputed from all samples

## Threshold Used

The threshold is determined by running `evaluate.py` on your enrollment data. The evaluation script:

1. Computes pairwise cosine distances between:
   - **Genuine pairs**: Different samples of the same person
   - **Impostor pairs**: Samples from different people

2. Finds the threshold that minimizes False Acceptance Rate (FAR) while keeping False Rejection Rate (FRR) reasonable

3. Typical threshold range: **0.30 - 0.40** (cosine distance)

**Important:** The threshold is data-dependent. Always run evaluation on your specific enrollment data before using live recognition.

## Pipeline Explanation

### Complete Pipeline Flow

```
Camera Input
    ↓
Haar Cascade Detection (face bounding box)
    ↓
MediaPipe FaceMesh (5-point landmarks)
    ↓
5-Point Alignment → 112×112 crop
    ↓
ArcFace ONNX Embedding → 512-dim vector (L2-normalized)
    ↓
Cosine Similarity with Database
    ↓
Threshold Decision → Identity or "Unknown"
```

### Why 5-Point Alignment?

- **Normalization**: Aligns faces to a standard pose, reducing pose variation
- **Consistency**: Ensures eyes, nose, and mouth are in consistent positions
- **Better Embeddings**: ArcFace models are trained on aligned faces
- **Robustness**: Handles slight rotations and scale variations

### Why Embeddings Instead of Memorization?

- **Generalization**: Embeddings capture facial features, not pixel values
- **Efficiency**: 512-dim vector vs. storing all images
- **Robustness**: Works with different lighting, expressions, angles
- **Scalability**: Can compare against thousands of identities quickly

### Cosine Similarity and Thresholds

- Embeddings are **L2-normalized**, so cosine similarity = dot product
- **Cosine distance** = 1 - cosine similarity
- **Lower distance** = more similar faces
- **Threshold** determines acceptance/rejection boundary

## Troubleshooting

### Camera Not Opening

- Try different camera indices (0, 1, 2) in the code
- Check if another application is using the camera
- On Linux, ensure proper permissions: `sudo usermod -a -G video $USER`

### Model File Missing

- Ensure `models/embedder_arcface.onnx` exists
- Check file path is correct
- Verify the model file is not corrupted

### Face Not Detected

- Ensure good lighting
- Face should be clearly visible and not too far from camera
- Try adjusting `min_size` parameter in detection code

### Poor Recognition Accuracy

- Enroll more samples per person (15+ recommended)
- Ensure varied poses and expressions during enrollment
- Run threshold evaluation to find optimal threshold
- Check that alignment is working correctly (use `align.py`)

### Import Errors

- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Check Python version compatibility

## System Requirements

- **CPU**: Any modern CPU (runs on CPU-only, no GPU required)
- **RAM**: 2GB+ recommended
- **Camera**: Any USB webcam or built-in camera
- **OS**: Windows 10+, Linux, macOS

## Project Requirements Checklist

- [x] 5-point facial landmark alignment
- [x] ArcFace embeddings using ONNX Runtime (CPU-only)
- [x] Enrollment system with multiple samples per identity
- [x] Threshold evaluation script
- [x] Live recognition with threshold-based decisions
- [x] L2-normalized embeddings
- [x] Proper project structure (src/, data/, models/)
- [x] Complete README documentation

## License

This project is for educational purposes. Please refer to the original ResearchGate publication for academic citations.

## References

- ResearchGate Publication: [Face Recognition with ArcFace ONNX and 5-Point Alignment](https://www.researchgate.net/publication/399996476_Face_Recognition_with_ArcFace_ONNX_and_5-Point_Alignment)
- InsightFace: https://github.com/deepinsight/insightface
- MediaPipe: https://developers.google.com/mediapipe
- OpenCV: https://opencv.org/

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments in each module
3. Ensure all dependencies are correctly installed
4. Verify model files are present and valid

---

**Good luck building your face recognition system!** 🎯
