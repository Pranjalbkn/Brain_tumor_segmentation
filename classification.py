import os
import io
import base64
import cv2
import numpy as np
import joblib
from typing import List, Tuple
from PIL import Image

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from skimage.feature import graycomatrix, graycoprops
from skimage import measure

# Initialize Router
router = APIRouter()

# --- Configuration & Model Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjust this path if your model is in a different folder
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "tumor_classifier_model.pkl")

classification_model = None

def load_classification_model():
    """
    Singleton pattern to load the model once.
    """
    global classification_model
    if classification_model is None:
        try:
            # Check if file exists to avoid generic joblib errors
            if os.path.exists(CLASSIFICATION_MODEL_PATH):
                classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
                print(f"Classification model loaded from {CLASSIFICATION_MODEL_PATH}")
            else:
                # Fallback: try loading from current directory if 'models' folder doesn't exist
                local_path = "tumor_classifier_model.pkl"
                if os.path.exists(local_path):
                    classification_model = joblib.load(local_path)
                    print(f"Classification model loaded from {local_path}")
                else:
                    print(f"ERROR: Model not found at {CLASSIFICATION_MODEL_PATH} or {local_path}")
                    classification_model = None
        except Exception as e:
            print(f"ERROR loading classification model: {e}")
            classification_model = None
    return classification_model


# --- Core Logic Functions (Preserved from Streamlit) ---

def build_gc_mask_from_seed(image_bgr, clean_seed_mask):
    gc_mask = np.full(image_bgr.shape[:2], cv2.GC_PR_BGD, np.uint8)
    gc_mask[clean_seed_mask == 0] = cv2.GC_BGD
    gc_mask[clean_seed_mask == 1] = cv2.GC_PR_FGD
    core = cv2.erode(clean_seed_mask, np.ones((5,5), np.uint8), iterations=1)
    gc_mask[core == 1] = cv2.GC_FGD
    return gc_mask

def seed_from_otsu(image_gray):
    image_blur = cv2.medianBlur(image_gray, 5)
    _, otsu_mask = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_open = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    clean_seed_mask = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, np.ones((10,10), np.uint8))
    return clean_seed_mask

def fallback_seed(image_gray):
    image_blur = cv2.medianBlur(image_gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(image_blur)
    _, otsu_mask2 = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_open = cv2.morphologyEx(otsu_mask2, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    clean_seed_mask = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, np.ones((10,10), np.uint8))
    return clean_seed_mask

def grabcut_refine(image_bgr, init_mask):
    gc_mask = build_gc_mask_from_seed(image_bgr, init_mask)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(image_bgr, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
    return final_mask

def extract_features_and_mask(pil_img):
    try:
        image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        clean_seed_mask = seed_from_otsu(image_gray)
        contours, _ = cv2.findContours(clean_seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            clean_seed_mask = fallback_seed(image_gray)
            contours, _ = cv2.findContours(clean_seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, None, None

        final_mask = grabcut_refine(image_bgr, (clean_seed_mask > 0).astype(np.uint8))

        labels = measure.label(final_mask)
        props = measure.regionprops(labels, intensity_image=image_gray)
        if not props:
            return None, None, None
        
        tumor_blob = max(props, key=lambda p: p.area)

        area = tumor_blob.area
        perimeter = tumor_blob.perimeter
        eccentricity = tumor_blob.eccentricity
        solidity = tumor_blob.solidity

        minr, minc, maxr, maxc = tumor_blob.bbox
        crop_gray = image_gray[minr:maxr, minc:maxc]
        crop_mask = final_mask[minr:maxr, minc:maxc].astype(bool)
        
        if crop_gray.size < 4 or crop_mask.sum() < 10:
            return None, None, None

        crop = crop_gray.copy()
        bg_mean = int(crop_gray[crop_mask].mean()) if crop_mask.sum() > 0 else int(crop_gray.mean())
        crop[~crop_mask] = bg_mean
        crop_u8 = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        distances = [1, 2, 4]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(crop_u8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        def agg(prop):
            m = graycoprops(glcm, prop)
            return float(m.mean())

        contrast = agg('contrast')
        energy = agg('energy')
        homogeneity = agg('homogeneity')
        correlation = agg('correlation')

        features = [area, perimeter, eccentricity, solidity, contrast, energy, homogeneity, correlation]
        return features, final_mask, image_bgr
    
    except Exception as e:
        # Log the error but don't crash, return None to handle in the endpoint
        print(f"Extraction Error: {e}")
        return None, None, None


# --- API Endpoints ---

@router.post("/classify")
async def classify_tumor(file: UploadFile = File(...)):
    try:
        # 1. Read Image
        image_data = await file.read()
        pil_img = Image.open(io.BytesIO(image_data))

        # 2. Encode Original Image to Base64 (for frontend display)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        original_img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # 3. Extract Features and Mask (Core Logic)
        features, segmentation_mask, image_bgr = extract_features_and_mask(pil_img)

        if features is None:
            return JSONResponse(status_code=400, content={
                'success': False,
                'error': 'Could not find a reliable tumor seed or contours. Try another image.'
            })

        # 4. Generate Visualization (Green Tumor over Grayscale)
        # This duplicates the logic from your Streamlit logic:
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

        green_overlay = np.zeros_like(image_bgr)
        green_overlay[..., 1] = 255
        tumor_color = cv2.addWeighted(image_bgr, 0.6, green_overlay, 0.4, 0)

        out = gray3.copy()
        m = segmentation_mask.astype(bool)
        out[m] = tumor_color[m]

        # Convert the processed OpenCV image to Base64
        rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        pil_out = Image.fromarray(rgb_out)
        buffered_out = io.BytesIO()
        pil_out.save(buffered_out, format="PNG")
        segmented_img_base64 = base64.b64encode(buffered_out.getvalue()).decode()

        # 5. Load Model and Predict
        pipe = load_classification_model()
        if pipe is None:
            return JSONResponse(status_code=500, content={
                'success': False,
                'error': 'Classification model not loaded. Check server logs.'
            })

        prediction = pipe.predict([features])
        probability = pipe.predict_proba([features])
        confidence = float(np.max(probability)) * 100.0
        predicted_class = prediction[0]

        # 6. Construct Response
        response = {
            'success': True,
            'original_image': original_img_base64,
            'segmented_image': segmented_img_base64,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'features': {
                'Area': features[0],
                'Perimeter': features[1],
                'Eccentricity': features[2],
                'Solidity': features[3],
                'Contrast': features[4],
                'Energy': features[5],
                'Homogeneity': features[6],
                'Correlation': features[7],
            },
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})