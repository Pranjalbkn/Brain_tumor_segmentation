# Brain Tumor Segmentation & Classification

A deep learning-powered web application for automated brain tumor detection, segmentation, and classification using MRI images. Built with FastAPI and PyTorch.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121.1-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

##  Overview

NeuroScan provides two main functionalities:

1. **Brain Tumor Segmentation** - Precisely identifies and segments tumor regions in brain MRI scans using a custom U-Net architecture (BRISC-UNet)
2. **Tumor Classification** - Classifies brain tumors into different categories using machine learning techniques with GLCM feature extraction

##  Features

-  **Accurate Segmentation** - Deep learning-based tumor boundary detection
-  **Multi-class Classification** - Categorizes tumors into different types
-  **Web Interface** - User-friendly HTML interface for easy interaction
-  **Fast Processing** - Optimized inference pipeline
-  **Visualization** - Clear visual outputs with segmentation masks and overlays
-  **RESTful API** - Easy integration with other systems

##  Architecture

### Segmentation Module
- **Model**: BRISC-UNet (Custom U-Net architecture)
- **Input**: 256x256 MRI images
- **Output**: Segmentation mask with tumor boundaries
- **Post-processing**: Graph-cut refinement, morphological operations

### Classification Module
- **Features**: GLCM texture analysis
- **Model**: Machine learning classifier (scikit-learn based)
- **Input**: Preprocessed MRI images
- **Output**: Tumor class prediction

##  Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/gouravslnk/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify model files**
Ensure the following model files are in the `models/` directory:
- `BRISC_UNet.pth` (Segmentation model)
- `tumor_classifier_model.pkl` (Classification model)

### Running the Application

Start the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```

The application will be available at:
- **Home**: http://localhost:8000
- **Segmentation**: http://localhost:8000/segmentation
- **Classification**: http://localhost:8000/classification

## 📁 Project Structure

```
brain-tumor-segmentation/
├── main.py                      # FastAPI application entry point
├── segmentation.py              # Tumor segmentation module & API
├── classification.py            # Tumor classification module & API
├── requirements.txt             # Python dependencies
├── models/
│   ├── BRISC_UNet.pth          # Pre-trained segmentation model
│   └── tumor_classifier_model.pkl  # Pre-trained classification model
├── static/
│   ├── home.html               # Landing page
│   ├── segmentation.html       # Segmentation interface
│   └── classification.html     # Classification interface
└── README.md
```

##  API Endpoints

### Segmentation API
```http
POST /segment
Content-Type: multipart/form-data

Parameters:
- file: MRI image file (JPEG/PNG)
- threshold: (optional) Segmentation threshold (0.0-1.0)

Response:
{
  "segmented_image": "base64_encoded_image",
  "processing_time": 1.23,
  "tumor_detected": true
}
```

### Classification API
```http
POST /classify
Content-Type: multipart/form-data

Parameters:
- file: MRI image file (JPEG/PNG)

Response:
{
  "prediction": "tumor_class",
  "confidence": 0.95,
  "processing_time": 0.45
}
```

##  Technology Stack

- **Backend**: FastAPI
- **Deep Learning**: PyTorch, TorchVision
- **Image Processing**: OpenCV, Pillow, scikit-image
- **ML Tools**: NumPy, joblib, PyMaxFlow
- **Server**: Uvicorn (ASGI)

##  Model Information

### BRISC-UNet
- Custom U-Net architecture optimized for brain MRI segmentation
- Input size: 256×256 pixels
- Output: Binary segmentation mask
- Post-processing includes graph-cut refinement and morphological operations

### Classification Model
- Feature extraction using GLCM (Gray Level Co-occurrence Matrix)
- Multiple texture features analyzed
- Trained on curated brain tumor dataset

##  Use Cases

- Medical research and education
- Computer-aided diagnosis (CAD) systems
- Healthcare demonstrations
- Academic projects and presentations
- Radiology workflow assistance

##  Notes

- This application is intended for educational and research purposes
- Not a substitute for professional medical diagnosis
- Always consult qualified healthcare professionals for medical decisions

##  Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
