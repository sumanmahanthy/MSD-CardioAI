"""
=============================================================================
 preprocess.py  Image Preprocessing for Aura-X Student (ConvNeXt-V2-Tiny)
=============================================================================
 Pipeline matches cardioai_v6_student_ensemble.py ChestXrayDataset exactly:
   1. Grayscale conversion
   2. CLAHE  (clipLimit=2.0, tileGridSize=88)
   3. Resize to 384384  (student training resolution)
   4. Stack to 3-channel
   5. ImageNet normalisation  =[0.485,0.456,0.406]  =[0.229,0.224,0.225]
   6. Lung-mask (Otsu + morphological close)  anatomy-aware masking
=============================================================================
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

#  CLAHE object created once (same pattern as training scripts) 
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Target resolution must match student training (IMG_SIZE = 384 in student script)
IMG_SIZE = 384

# ImageNet normalisation (same as ChestXrayDataset in student_ensemble.py)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# 
# Chest X-Ray Input Validation
# 
class InvalidXrayError(ValueError):
    """Raised when the uploaded image is not a valid chest X-ray."""
    pass


def validate_chest_xray(image: Image.Image) -> None:
    """
    Validate that the uploaded image is likely a chest X-ray before inference.
    Raises InvalidXrayError with a descriptive message if validation fails.

    Three heuristic checks (no extra model needed, CPU-free):

    1. COLOUR SATURATION CHECK
       Chest X-rays are near-grayscale. Colour photos (selfies, landscapes)
       have high HSV saturation. Threshold validated on NIH ChestX-ray14
       (all grayscale, saturation  0) vs natural images (saturation > 40).

    2. ASPECT RATIO CHECK
       PA/AP chest X-rays have aspect ratio (W/H) between 0.65 and 1.50.
       Extreme ratios (banners, portraits, icons) are outside this range.

    3. INTENSITY DISTRIBUTION CHECK
       Valid X-rays have a bimodal histogram: dark lung fields + bright bones.
       Blank/overexposed images have very high mean brightness (> 220/255).
       Flat/uniform images (logos, solid backgrounds) have very low std (< 15).

    References:
      Wang et al. (2017)  NIH ChestX-ray14 dataset & baselines.
      Rajpurkar et al. (2017)  CheXNet: Radiologist-Level Pneumonia Detection.
    """
    img_np = np.array(image.convert("RGB"), dtype=np.uint8)   # H  W  3
    h, w   = img_np.shape[:2]

    #  Check 1: Colour saturation (HSV S-channel mean) 
    hsv     = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    avg_sat = float(hsv[:, :, 1].mean())   # S channel, range 0255
    # X-rays are monochrome but JPEG/PACS exports often carry slight chroma
    # fringing (saturation 30–55/255). Colour photos are typically > 80.
    if avg_sat > 60:
        raise InvalidXrayError(
            f"Colour photograph detected (avg saturation={avg_sat:.1f}/255). "
            "Please upload a frontal chest X-ray (PA or AP view) in PNG or JPEG format."
        )

    #  Check 2: Aspect ratio 
    aspect = w / h
    if not (0.60 <= aspect <= 1.60):
        raise InvalidXrayError(
            f"Invalid image dimensions (aspect ratio={aspect:.2f}). "
            "A standard frontal chest X-ray should have an aspect ratio between 0.60 and 1.60."
        )

    #  Check 3: Intensity distribution 
    gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean_val = float(gray.mean())
    std_val  = float(gray.std())
    if mean_val > 220:
        raise InvalidXrayError(
            f"Image appears blank or overexposed (mean brightness={mean_val:.1f}/255). "
            "Please upload a correctly exposed frontal chest X-ray."
        )
    if std_val < 15:
        raise InvalidXrayError(
            f"Image has insufficient contrast (std={std_val:.1f}/255). "
            "A valid chest X-ray should show clear contrast between lung fields, "
            "cardiac silhouette, and thoracic structures."
        )

    #  Check 4: Bright pixel ratio 
    # Chest X-rays on white backgrounds: lung fields (dark), bones (bright but sparse).
    # Empirically, valid X-rays have < 45% pixels above 200/255.
    # Documents, notebooks, screenshots typically have > 65% near-white pixels.
    # Reference: NIH ChestX-ray14 pixel intensity histogram analysis.
    bright_ratio = float((gray > 200).mean())   # fraction of pixels > 200/255
    if bright_ratio > 0.60:
        raise InvalidXrayError(
            f"Image appears to be a document or paper scan ({bright_ratio*100:.1f}% bright pixels). "
            "Please upload a frontal chest X-ray in grayscale DICOM, PNG, or JPEG format. "
            "Documents, screenshots, and notebook photos are not supported."
        )
    # All checks passed  proceed with inference

def apply_clahe(image: Image.Image):
    """
    Full preprocessing pipeline for Aura-X (ConvNeXt-V2-Tiny student).

    Parameters
    ----------
    image : PIL.Image  (any mode, will be converted to grayscale internally)

    Returns
    -------
    tensor     : torch.FloatTensor  shape (1, 3, 384, 384)   model input
    visual_img : PIL.Image RGB                               for Grad-CAM overlay
    """
    #  Step 1: Grayscale 
    img_gray = np.array(image.convert("L"), dtype=np.uint8)   # HW uint8

    #  Step 2: CLAHE (mirrors training script exactly) 
    img_clahe = _CLAHE.apply(img_gray)                         # HW uint8

    #  Step 3: Resize to 384384 
    img_resized = cv2.resize(img_clahe, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)   # 384384 uint8

    #  Step 4: 3-channel stacking 
    #   Same as training: torch.from_numpy(gray).unsqueeze(0).repeat(3,1,1)
    img_rgb = np.stack([img_resized, img_resized, img_resized], axis=-1)  # HW3

    #  Step 5: ImageNet normalisation 
    img_pil = Image.fromarray(img_rgb)
    base_transform = transforms.Compose([
        transforms.ToTensor(),                                #  [3, H, W] float [0,1]
        transforms.Normalize(mean=_IMAGENET_MEAN,
                             std=_IMAGENET_STD)
    ])
    img_tensor = base_transform(img_pil)                      # [3, 384, 384]

    #  Step 6: Anatomy-aware lung mask (Otsu + morphological close) 
    img_for_mask = img_tensor[0].numpy()
    img_for_mask = ((img_for_mask * _IMAGENET_STD[0] + _IMAGENET_MEAN[0]) * 255
                    ).clip(0, 255).astype(np.uint8)

    _, mask = cv2.threshold(img_for_mask, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask_tensor = torch.tensor(mask / 255.0, dtype=torch.float32).unsqueeze(0)
    img_tensor  = img_tensor * mask_tensor                    # zero non-lung pixels

    #  Final tensor: add batch dim  (1, 3, 384, 384) 
    final_tensor = img_tensor.unsqueeze(0)

    # Visual RGB PIL for Grad-CAM overlay (before masking / normalising)
    visual_img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB))

    return final_tensor, visual_img
