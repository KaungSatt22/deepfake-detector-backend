from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import requests
import cv2
import numpy as np
import os
import io
import re
import tempfile

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SE_USER = os.getenv("SIGHTENGINE_USER")
SE_SECRET = os.getenv("SIGHTENGINE_SECRET")

# Local model load လုပ်မယ်
processor = None
model = None

def get_deepfake_model():
    global processor, model
    if processor is None or model is None:
        print("Loading deepfake model...")
        processor = AutoImageProcessor.from_pretrained("Wvolf/ViT_Deepfake_Detection")
        model = AutoModelForImageClassification.from_pretrained("Wvolf/ViT_Deepfake_Detection")
        model.eval()
        print("Model loaded!")
    return processor, model

def verdict_from_score(score, threshold=0.5):
    score = max(0, min(float(score), 1))
    is_fake = score > threshold
    verdict_confidence = score if is_fake else 1 - score
    return is_fake, round(verdict_confidence * 100, 2)

def percent(score):
    return round(max(0, min(float(score), 1)) * 100, 2)

def extract_transaction_text(contents):
    try:
        import pytesseract
    except ImportError:
        return {
            "available": False,
            "status": "OCR unavailable",
            "text": "",
            "warning": "Install pytesseract and Tesseract OCR to enable transaction text checks.",
        }

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        scale = 2 if max(gray.shape) < 1400 else 1
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        thresholded = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )

        variants = [
            image,
            Image.fromarray(gray),
            Image.fromarray(thresholded),
        ]
        texts = [
            pytesseract.image_to_string(variant, config="--oem 3 --psm 6")
            for variant in variants
        ]
        text = max(texts, key=lambda value: len(value.strip())).strip()

        return {
            "available": True,
            "status": "OCR completed" if text else "No readable text found",
            "text": text,
            "warning": None if text else "OCR could not read transaction text clearly.",
        }
    except Exception as e:
        return {
            "available": False,
            "status": "OCR failed",
            "text": "",
            "warning": f"OCR failed: {str(e)}",
        }

def analyze_transaction_text(text):
    normalized = re.sub(r"\s+", " ", text).strip()
    lower_text = normalized.lower()

    links = sorted(set(re.findall(
        r"(?:https?://|www\.|t\.me/|telegram\.me/|wa\.me/|bit\.ly/|tinyurl\.com/|"
        r"forms\.gle/|docs\.google\.com/forms)[^\s,;)]*",
        normalized,
        flags=re.IGNORECASE,
    )))
    phones = sorted(set(re.findall(
        r"(?:\+?95|0)9[\s-]?\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}",
        normalized,
        flags=re.IGNORECASE,
    )))
    amounts = sorted(set(re.findall(
        r"(?:MMK|Ks?|Kyats?)\s*[\d,]+(?:\.\d{1,2})?|"
        r"[\d,]+(?:\.\d{1,2})?\s*(?:MMK|Ks?|Kyats?)\b",
        normalized,
        flags=re.IGNORECASE,
    )))
    dates = sorted(set(re.findall(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
        r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
        normalized,
        flags=re.IGNORECASE,
    )))
    references = sorted(set(re.findall(
        r"\b(?:ref(?:erence)?|txn|transaction|trace|voucher|receipt|id|no)\.?\s*[:#-]?\s*[A-Z0-9-]{5,}\b",
        normalized,
        flags=re.IGNORECASE,
    )))

    suspicious_keywords = [
        "otp", "password", "pin", "verify", "verification", "login", "click",
        "telegram", "airdrop", "crypto", "usdt", "bonus", "prize", "winner",
        "urgent", "limited", "fee", "tax", "unlock", "investment", "loan",
    ]
    matched_keywords = sorted({word for word in suspicious_keywords if word in lower_text})

    account_indicators = [
        "account", "acct", "recipient", "receiver", "beneficiary", "to ",
        "from ", "sender", "name", "payee",
    ]
    has_account_indicator = any(indicator in lower_text for indicator in account_indicators)

    warnings = []
    risk_score = 0.0

    if links:
        warnings.append("Links or social chat links found in the transaction screenshot.")
        risk_score += 0.35

    if matched_keywords:
        warnings.append(f"Suspicious scam keywords found: {', '.join(matched_keywords[:6])}.")
        risk_score += min(0.35, 0.08 * len(matched_keywords))

    if len(phones) > 1:
        warnings.append("Multiple phone numbers found; verify sender and recipient manually.")
        risk_score += 0.12

    if not amounts:
        warnings.append("No clear transaction amount found.")
        risk_score += 0.18

    if not dates:
        warnings.append("No clear transaction date found.")
        risk_score += 0.12

    if not references:
        warnings.append("No clear transaction/reference ID found.")
        risk_score += 0.14

    if not has_account_indicator:
        warnings.append("No clear account name, sender, or recipient label found.")
        risk_score += 0.12

    if len(normalized) < 30:
        warnings.append("Very little readable text found; screenshot may be cropped, blurred, or edited.")
        risk_score += 0.16

    risk_score = max(0, min(risk_score, 1))

    return {
        "risk_score": risk_score,
        "warnings": warnings,
        "fields": {
            "amounts": amounts[:5],
            "dates": dates[:5],
            "phones": phones[:5],
            "links": links[:5],
            "references": references[:5],
            "keywords": matched_keywords[:8],
            "has_account_indicator": has_account_indicator,
        },
        "text_preview": normalized[:700],
    }

def predict_frame(pil_image):
    processor, model = get_deepfake_model()
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    labels = model.config.id2label
    
    scores = {labels[i]: probs[i].item() for i in range(len(probs))}
    print(f"Frame scores: {scores}")
    
    fake_score = scores.get("Fake", scores.get("fake", scores.get("FAKE", 0)))
    real_score = scores.get("Real", scores.get("real", scores.get("REAL", 0)))
    return fake_score, real_score

@app.get("/")
def root():
    return {"message": "Deepfake Detector API running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()

    response = requests.post(
        "https://api.sightengine.com/1.0/check.json",
        files={"media": (file.filename, contents, file.content_type)},
        data={
            "models": "deepfake",
            "api_user": SE_USER,
            "api_secret": SE_SECRET,
        },
    )

    if response.status_code != 200:
        return {"error": "Model error", "detail": response.text}

    result = response.json()
    score = result.get("type", {}).get("deepfake", 0)
    is_fake, confidence = verdict_from_score(score)

    return {
        "is_fake": is_fake,
        "verdict": "AI Generated" if is_fake else "Likely Real",
        "confidence": confidence,
        "fake_score": round(score * 100, 2),
        "real_score": round((1 - score) * 100, 2),
        "raw": result
    }

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    contents = await file.read()

    # Temp file သွင်းမယ်
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        # 10 frame ပဲ sample ယူမယ် — fast ဖြစ်အောင်
        sample_count = min(10, total_frames)
        interval = max(1, total_frames // sample_count)

        fake_scores = []
        frames_analyzed = 0

        for i in range(sample_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if not ret:
                continue

            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            fake_score, real_score = predict_frame(pil_image)
            fake_scores.append(fake_score)
            frames_analyzed += 1

        cap.release()

        if not fake_scores:
            return {"error": "Could not extract frames from video"}

        avg_fake = sum(fake_scores) / len(fake_scores)
        is_fake, confidence = verdict_from_score(avg_fake)

        return {
            "is_fake": is_fake,
            "verdict": "AI Generated" if is_fake else "Likely Real",
            "confidence": confidence,
            "fake_score": round(avg_fake * 100, 2),
            "real_score": round((1 - avg_fake) * 100, 2),
            "frames_analyzed": frames_analyzed,
            "raw": {"per_frame_scores": fake_scores}
        }

    finally:
        os.unlink(tmp_path)

@app.post("/analyze-transaction")
async def analyze_transaction(file: UploadFile = File(...)):
    contents = await file.read()

    # Step 1 — Sightengine document/image forgery check
    response = requests.post(
        "https://api.sightengine.com/1.0/check.json",
        files={"media": (file.filename, contents, file.content_type)},
        data={
            "models": "genai,deepfake",
            "api_user": SE_USER,
            "api_secret": SE_SECRET,
        },
    )

    if response.status_code != 200:
        return {"error": "API error", "detail": response.text}

    result = response.json()

    # Scores ယူမယ်
    deepfake_score = result.get("type", {}).get("deepfake", 0)
    genai_score = result.get("type", {}).get("genai", 0)

    # ပိုမြင့်တဲ့ score ကိုယူမယ်
    manipulation_score = max(deepfake_score, genai_score)
    ocr_result = extract_transaction_text(contents)
    ocr_analysis = analyze_transaction_text(ocr_result["text"]) if ocr_result["text"] else {
        "risk_score": 0,
        "warnings": [],
        "fields": {},
        "text_preview": "",
    }
    evidence_score = max(manipulation_score, ocr_analysis["risk_score"])
    is_fake, confidence = verdict_from_score(evidence_score, threshold=0.4)

    warnings = list(ocr_analysis["warnings"])
    if ocr_result["warning"]:
        warnings.append(ocr_result["warning"])

    # Risk level သတ်မှတ်မယ်
    if evidence_score > 0.7:
        risk = "High Risk"
        risk_color = "red"
    elif evidence_score > 0.4:
        risk = "Medium Risk"
        risk_color = "orange"
    else:
        risk = "Low Risk"
        risk_color = "green"

    return {
        "is_fake": is_fake,
        "verdict": "Likely Forged" if is_fake else "Likely Authentic",
        "confidence": confidence,
        "evidence_score": percent(evidence_score),
        "manipulation_score": percent(manipulation_score),
        "ocr_risk_score": percent(ocr_analysis["risk_score"]),
        "authentic_score": percent(1 - evidence_score),
        "risk": risk,
        "risk_color": risk_color,
        "warnings": warnings,
        "ocr": {
            "status": ocr_result["status"],
            "available": ocr_result["available"],
            "fields": ocr_analysis["fields"],
            "text_preview": ocr_analysis["text_preview"],
        },
        "raw": result
    }
    
@app.post("/analyze-metadata")
async def analyze_metadata(file: UploadFile = File(...)):
    contents = await file.read()

    metadata = {}
    warnings = []

    try:
        image = Image.open(io.BytesIO(contents))
        exif_data = image._getexif()

        if exif_data:
            from PIL.ExifTags import TAGS, GPSTAGS

            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="ignore")
                    except:
                        value = str(value)
                metadata[tag] = str(value)

            # GPS check
            if "GPSInfo" in metadata:
                warnings.append("📍 GPS location data found — privacy risk")

            # Software check
            software = metadata.get("Software", "")
            suspicious_software = ["photoshop", "gimp", "lightroom", "affinity", "canva", "snapseed"]
            if any(s in software.lower() for s in suspicious_software):
                warnings.append(f"⚠️ Edited with: {software}")

            # DateTime check
            original = metadata.get("DateTimeOriginal", "")
            modified = metadata.get("DateTime", "")
            if original and modified and original != modified:
                warnings.append(f"⚠️ Modified after creation — Original: {original} / Modified: {modified}")

        else:
            warnings.append("⚠️ No EXIF data found — metadata may have been stripped (suspicious)")

        # Basic file info
        file_info = {
            "filename": file.filename,
            "format": image.format,
            "mode": image.mode,
            "size": f"{image.width}x{image.height}px",
            "file_size": f"{round(len(contents) / 1024, 2)} KB",
        }

        is_suspicious = len(warnings) > 0

        return {
            "is_suspicious": is_suspicious,
            "verdict": "Suspicious" if is_suspicious else "Clean",
            "warnings": warnings,
            "file_info": file_info,
            "metadata": metadata,
        }

    except Exception as e:
        return {"error": "Could not read metadata", "detail": str(e)}
