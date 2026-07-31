import os
import base64
import io
import time
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import maxflow
except ImportError:
    maxflow = None


router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEGMENTATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "BRISC_UNet.pth")
IMAGE_SIZE = 256


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(1, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.center = CBR(512, 1024)

        self.dec4 = CBR(1024 + 512, 512)
        self.dec3 = CBR(512 + 256, 256)
        self.dec2 = CBR(256 + 128, 128)
        self.dec1 = CBR(128 + 64, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        center = self.center(self.pool(e4))

        d4 = F.interpolate(center, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.final(d1))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentation_model = None


def load_segmentation_model():
    global segmentation_model
    if segmentation_model is not None:
        return segmentation_model

    try:
        print(f"Loading UNet segmentation model on device: {DEVICE}")
        model = UNet()
        state_dict = torch.load(SEGMENTATION_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        segmentation_model = model
        print("UNet segmentation model loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Segmentation model file not found at {SEGMENTATION_MODEL_PATH}. Using mock function.")
        segmentation_model = "MOCK"
    except Exception as e:
        print(f"CRITICAL ERROR loading segmentation model weights: {e}")
        segmentation_model = "MOCK"

    return segmentation_model


def grabcut_refine_segmentation(img_gray, prob):
    mask = np.where(prob >= 0.5, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return final_mask


def maxflow_refine(img_gray, prob, lam=25.0):
    if not maxflow:
        raise RuntimeError("The 'pymaxflow' library is required but not installed.")
    H, W = img_gray.shape
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((H, W))
    eps = 1e-6
    g.add_grid_tedges(nodeids, -np.log(1 - prob + eps), -np.log(prob + eps))
    g.maxflow()
    seg = g.get_grid_segments(nodeids)
    return np.where(seg, 255, 0).astype(np.uint8)


def unet_prob_map(img_gray_np, model, device, in_size=IMAGE_SIZE):
    H, W = img_gray_np.shape[:2]
    img_resized = cv2.resize(img_gray_np, (in_size, in_size)).astype(np.float32) / 255.0
    tensor_input = torch.from_numpy(img_resized)[None, None].to(device)
    with torch.no_grad():
        prob_small = model(tensor_input).squeeze().cpu().numpy()
    return cv2.resize(prob_small, (W, H), interpolation=cv2.INTER_LINEAR)


def overlay_mask(image_np, mask_np, color=(0, 0, 255), alpha=0.4):
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    mask_bgr = np.zeros_like(img_bgr)
    mask_bgr[mask_np > 0] = color
    return cv2.addWeighted(img_bgr, 1, mask_bgr, alpha, 0)


def pytorch_segmentation(input_image: Image.Image, mode: str) -> Tuple[Image.Image, str]:
    img_gray_pil = input_image.convert("L")
    img_gray_np = np.array(img_gray_pil)
    prob_map = unet_prob_map(img_gray_np, segmentation_model, DEVICE)

    final_mask_np = None
    color = (0, 0, 255)

    if mode == "GrabCut":
        final_mask_np = grabcut_refine_segmentation(img_gray_np, prob_map)
        mode_text = "GrabCut Refinement"
        color = (0, 255, 0)
    elif mode == "MaxFlow":
        final_mask_np = maxflow_refine(img_gray_np, prob_map)
        mode_text = "MaxFlow Refinement"
        color = (255, 0, 0)
    else:
        final_mask_np = (prob_map > 0.5).astype(np.uint8) * 255
        mode_text = "UNet Only"

    unet_overlay_bgr = overlay_mask(img_gray_np, final_mask_np, color=color)
    unet_overlay_rgb = cv2.cvtColor(unet_overlay_bgr, cv2.COLOR_BGR2RGB)
    segmented_image = Image.fromarray(unet_overlay_rgb)

    return segmented_image, mode_text


def mock_segmentation(input_image: Image.Image) -> Tuple[Image.Image, str]:
    time.sleep(1.5)
    segmented_image = input_image.convert("RGBA")
    draw = ImageDraw.Draw(segmented_image)
    width, height = input_image.size

    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 5
    bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
    mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.ellipse(bbox, fill=(255, 0, 0, 100))

    segmented_image = Image.alpha_composite(segmented_image, mask)

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(segmented_image)
    text = "MOCK SEGMENTATION: MODEL FAILED TO LOAD"
    text_w, text_h = 350, 40
    draw.text((10, height - text_h - 10), text, fill=(255, 255, 0, 255), font=font)
    return segmented_image.convert("RGB"), "Mock Result"


@router.post("/predict")
async def predict_segmentation(
    file: UploadFile = File(...),
    mode: str = Form("UNet Only"),
):
    try:
        image_data = await file.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        if input_image.width < 100 or input_image.height < 100:
            raise HTTPException(status_code=400, detail="Image resolution is too low.")

        model = load_segmentation_model()

        if model and model != "MOCK":
            segmented_image, mode_text = pytorch_segmentation(input_image, mode)
            result_message = f"Segmentation process successful ({mode_text})."
        else:
            segmented_image, mode_text = mock_segmentation(input_image)
            result_message = f"Segmentation process failed ({mode_text})."

        buffer = io.BytesIO()
        segmented_image.save(buffer, format="JPEG", quality=90)
        img_str = (base64.b64encode(buffer.getvalue()).decode())

        return JSONResponse(content={"result_image": img_str, "message": result_message})

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    except Exception as e:
        if "pymaxflow" in str(e):
            detail_msg = "MaxFlow library is missing. Please run: pip install pymaxflow"
        else:
            detail_msg = f"Internal server error during image processing: {e.__class__.__name__}"
        raise HTTPException(status_code=500, detail=detail_msg)
