from fastapi import FastAPI, File, UploadFile, Request
from google import genai
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
import json
import re
from groq import Groq

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://deepfake-detector-sigma.vercel.app",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
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

    HIVE_API_KEY = os.getenv("HIVE_API_KEY")

    # Hive API call
    hive_response = requests.post(
        "https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection",
        headers={"Authorization": f"Bearer {HIVE_API_KEY}"},
        files={"media": (file.filename, contents, file.content_type)},
    )

    if hive_response.status_code != 200:
        return {"error": "Hive API error", "detail": hive_response.text}

    hive_result = hive_response.json()
    classes = hive_result.get("output", [{}])[0].get("classes", [])
    scores = {c["class"]: c["value"] for c in classes}

    ai_score = scores.get("ai_generated", 0)
    deepfake_score = scores.get("deepfake", 0)

    # Top source detect လုပ်မယ်
    exclude = {"ai_generated", "not_ai_generated", "none", "inconclusive",
               "inconclusive_video", "deepfake", "not_ai_generated_audio", "ai_generated_audio"}
    source_scores = {k: v for k, v in scores.items() if k not in exclude}
    top_source = max(source_scores, key=source_scores.get)
    top_source_score = source_scores[top_source]

    # Combined score
    combined_score = max(ai_score, deepfake_score, top_source_score if top_source_score > 0.1 else 0)
    is_fake = combined_score > 0.2
    confidence = round(combined_score * 100, 2)

    # Reasons
    reasons = []
    if ai_score > 0.2:
        reasons.append(f"AI generation detected ({round(ai_score*100)}%)")
    if deepfake_score > 0.2:
        reasons.append(f"Deepfake detected ({round(deepfake_score*100)}%)")
    if top_source_score > 0.1:
        reasons.append(f"Likely generated by: {top_source} ({round(top_source_score*100)}%)")

    return {
        "is_fake": is_fake,
        "verdict": "AI Generated / Manipulated" if is_fake else "Likely Authentic",
        "confidence": confidence,
        "reasons": reasons,
        "raw": scores
    }

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    contents = await file.read()

    HIVE_API_KEY = os.getenv("HIVE_API_KEY")

    # Hive API — video directly ပို့မယ်
    hive_response = requests.post(
        "https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection",
        headers={"Authorization": f"Bearer {HIVE_API_KEY}"},
        files={"media": (file.filename, contents, file.content_type)},
        timeout=120
    )

    if hive_response.status_code != 200:
        return {"error": "Hive API error", "detail": hive_response.text}

    hive_result = hive_response.json()
    frames = hive_result.get("output", [])

    if not frames:
        return {"error": "No frames analyzed"}

    # Frame တွေထဲက scores average ယူမယ်
    ai_scores = []
    deepfake_scores = []
    source_totals = {}

    exclude = {"ai_generated", "not_ai_generated", "none", "inconclusive",
               "inconclusive_video", "deepfake", "not_ai_generated_audio",
               "ai_generated_audio"}

    for frame in frames:
        classes = frame.get("classes", [])
        scores = {c["class"]: c["value"] for c in classes}

        ai_scores.append(scores.get("ai_generated", 0))
        deepfake_scores.append(scores.get("deepfake", 0))

        for k, v in scores.items():
            if k not in exclude:
                source_totals[k] = source_totals.get(k, 0) + v

    avg_ai = sum(ai_scores) / len(ai_scores)
    avg_deepfake = sum(deepfake_scores) / len(deepfake_scores)

    # Top source
    top_source = max(source_totals, key=source_totals.get) if source_totals else None
    top_source_avg = source_totals[top_source] / len(frames) if top_source else 0

    combined_score = max(avg_ai, avg_deepfake, top_source_avg if top_source_avg > 0.1 else 0)
    is_fake = combined_score > 0.2
    confidence = round(combined_score * 100, 2)

    reasons = []
    if avg_ai > 0.2:
        reasons.append(f"AI generation detected ({round(avg_ai*100)}%)")
    if avg_deepfake > 0.2:
        reasons.append(f"Deepfake detected ({round(avg_deepfake*100)}%)")
    if top_source and top_source_avg > 0.1:
        reasons.append(f"Likely generated by: {top_source} ({round(top_source_avg*100)}%)")

    return {
        "is_fake": is_fake,
        "verdict": "AI Generated / Manipulated" if is_fake else "Likely Authentic",
        "confidence": confidence,
        "frames_analyzed": len(frames),
        "reasons": reasons,
        "raw": {"ai_scores": ai_scores, "deepfake_scores": deepfake_scores}
    }

@app.post("/analyze-transaction")
async def analyze_transaction(file: UploadFile = File(...)):
    contents = await file.read()

    HIVE_API_KEY = os.getenv("HIVE_API_KEY")

    # Hive API call
    hive_response = requests.post(
        "https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection",
        headers={"Authorization": f"Bearer {HIVE_API_KEY}"},
        files={"media": (file.filename, contents, file.content_type)},
    )

    if hive_response.status_code != 200:
        return {"error": "Hive API error", "detail": hive_response.text}

    hive_result = hive_response.json()
    classes = hive_result.get("output", [{}])[0].get("classes", [])
    scores = {c["class"]: c["value"] for c in classes}

    ai_score = scores.get("ai_generated", 0)
    deepfake_score = scores.get("deepfake", 0)

    exclude = {"ai_generated", "not_ai_generated", "none", "inconclusive",
               "inconclusive_video", "deepfake", "not_ai_generated_audio", "ai_generated_audio"}
    source_scores = {k: v for k, v in scores.items() if k not in exclude}
    top_source = max(source_scores, key=source_scores.get)
    top_source_score = source_scores[top_source]

    combined_score = max(ai_score, deepfake_score, top_source_score if top_source_score > 0.1 else 0)
    is_fake = combined_score > 0.2
    confidence = round(combined_score * 100, 2)

    # Risk level
    if combined_score > 0.6:
        risk = "High Risk"
        risk_color = "red"
    elif combined_score > 0.2:
        risk = "Medium Risk"
        risk_color = "orange"
    else:
        risk = "Low Risk"
        risk_color = "green"

    reasons = []
    if ai_score > 0.2:
        reasons.append(f"AI generation detected ({round(ai_score*100)}%)")
    if deepfake_score > 0.2:
        reasons.append(f"Deepfake detected ({round(deepfake_score*100)}%)")
    if top_source_score > 0.1:
        reasons.append(f"Likely generated by: {top_source} ({round(top_source_score*100)}%)")

    # Metadata check ထပ်ထည့်မယ်
    try:
        image = Image.open(io.BytesIO(contents))
        exif_data = image._getexif()
        if not exif_data:
            reasons.append("⚠️ No EXIF metadata — may have been digitally processed")
    except:
        pass

    return {
        "is_fake": is_fake,
        "verdict": "Likely Forged" if is_fake else "Likely Authentic",
        "confidence": confidence,
        "risk": risk,
        "risk_color": risk_color,
        "reasons": reasons,
        "raw": scores
    }
    
@app.post("/analyze-metadata")
async def analyze_metadata(file: UploadFile = File(...)):
    contents = await file.read()

    HIVE_API_KEY = os.getenv("HIVE_API_KEY")
    warnings = []
    metadata = {}
    file_info = {}
    hive_score = 0

    # Step 1 — Hive AI Detection
    try:
        hive_response = requests.post(
            "https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection",
            headers={"Authorization": f"Bearer {HIVE_API_KEY}"},
            files={"media": (file.filename, contents, file.content_type)},
            timeout=30
        )
        if hive_response.status_code == 200:
            hive_result = hive_response.json()
            classes = hive_result.get("output", [{}])[0].get("classes", [])
            scores = {c["class"]: c["value"] for c in classes}

            ai_score = scores.get("ai_generated", 0)
            deepfake_score = scores.get("deepfake", 0)

            exclude = {"ai_generated", "not_ai_generated", "none", "inconclusive",
                      "inconclusive_video", "deepfake", "not_ai_generated_audio", "ai_generated_audio"}
            source_scores = {k: v for k, v in scores.items() if k not in exclude}
            top_source = max(source_scores, key=source_scores.get)
            top_source_score = source_scores[top_source]

            hive_score = max(ai_score, deepfake_score, top_source_score if top_source_score > 0.1 else 0)

            if ai_score > 0.2:
                warnings.append(f"🤖 AI generation detected ({round(ai_score*100)}%)")
            if deepfake_score > 0.2:
                warnings.append(f"👤 Deepfake detected ({round(deepfake_score*100)}%)")
            if top_source_score > 0.1:
                warnings.append(f"🔍 Likely generated by: {top_source} ({round(top_source_score*100)}%)")
    except Exception as e:
        warnings.append(f"⚠️ AI detection unavailable: {str(e)}")

    # Step 2 — EXIF Metadata Analysis
    try:
        image = Image.open(io.BytesIO(contents))
        exif_data = image._getexif()

        file_info = {
            "filename": file.filename,
            "format": image.format or "Unknown",
            "mode": image.mode,
            "size": f"{image.width}x{image.height}px",
            "file_size": f"{round(len(contents) / 1024, 2)} KB",
        }

        if exif_data:
            from PIL.ExifTags import TAGS
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="ignore")
                    except:
                        value = str(value)
                metadata[str(tag)] = str(value)

            # Software check
            software = metadata.get("Software", "").lower()
            suspicious_software = [
                "photoshop", "gimp", "lightroom", "affinity",
                "canva", "snapseed", "gemini", "midjourney",
                "dall-e", "stable diffusion", "firefly", "adobe"
            ]
            for s in suspicious_software:
                if s in software:
                    warnings.append(f"⚠️ Edited with suspicious software: {metadata.get('Software')}")
                    hive_score = max(hive_score, 0.6)
                    break

            # DateTime mismatch check
            original = metadata.get("DateTimeOriginal", "")
            modified = metadata.get("DateTime", "")
            if original and modified and original != modified:
                warnings.append(f"⚠️ Modified after creation — Original: {original} / Modified: {modified}")
                hive_score = max(hive_score, 0.4)

            # GPS check
            if "GPSInfo" in metadata:
                warnings.append("📍 GPS location data found — privacy risk")

            # Thumbnail mismatch check
            try:
                from PIL.ExifTags import TAGS
                if hasattr(image, '_getexif') and exif_data:
                    thumb_data = exif_data.get(513)  # JPEGInterchangeFormat
                    if thumb_data:
                        warnings.append("🖼️ Embedded thumbnail found — may differ from actual image")
            except:
                pass

        else:
            warnings.append("⚠️ No EXIF metadata found — may have been stripped or AI-generated")

    except Exception as e:
        warnings.append(f"⚠️ Could not read metadata: {str(e)}")

    # Final verdict
    # Final verdict
    is_suspicious = hive_score > 0.3

    if hive_score > 0.6:
        verdict = "Highly Suspicious"
    elif hive_score > 0.3:
        verdict = "Suspicious"
    else:
        verdict = "Clean"

    is_suspicious = hive_score > 0.3

    if hive_score > 0.6:
        verdict = "Highly Suspicious"
    elif hive_score > 0.3:
        verdict = "Suspicious"
    else:
        verdict = "Clean"

    is_suspicious = verdict != "Clean"
    confidence = round(hive_score * 100, 2)  # ← ဒါထည့်ပါ

    return {
        "is_suspicious": is_suspicious,
        "verdict": verdict,
        "confidence": confidence,
        "warnings": warnings,
        "file_info": file_info,
        "metadata": metadata,
    }

@app.post("/analyze-fakenews")
async def analyze_fakenews(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    url = body.get("url", "").strip()

    if not text and not url:
        return {"error": "Please provide text or URL"}

    # URL ဆိုရင် content fetch လုပ်မယ်
    content = text
    source_url = None

    if url:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            page = requests.get(url, headers=headers, timeout=10)
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in ["script", "style", "nav", "footer"]:
                        self.skip = True

                def handle_endtag(self, tag):
                    if tag in ["script", "style", "nav", "footer"]:
                        self.skip = False

                def handle_data(self, data):
                    if not self.skip and data.strip():
                        self.text.append(data.strip())

            parser = TextExtractor()
            parser.feed(page.text)
            content = " ".join(parser.text[:500])
            source_url = url
        except Exception as e:
            return {"error": f"Could not fetch URL: {str(e)}"}

    if not content:
        return {"error": "Could not extract content"}

    # Gemini analysis
    try:
        prompt = f"""
            You are a professional fact-checker and misinformation analyst.

            Analyze the following news content and determine if it's likely fake news or misinformation.

            Content to analyze:
            \"\"\"
            {content[:3000]}
            \"\"\"

            Respond ONLY in this exact JSON format (no markdown, no extra text):
            {{
                "verdict": "Fake News" or "Likely Fake" or "Uncertain" or "Likely Real" or "Real",
                "confidence": <number 0-100>,
                "reasons": [<list of 2-4 specific reasons for your assessment>],
                "red_flags": [<list of specific misinformation indicators found, empty if none>],
                "credibility_score": <number 0-100>,
                "summary": "<one sentence summary of the content>",
                "recommendation": "<what the user should do>"
            }}
            """
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional fact-checker and misinformation analyst. Always respond in valid JSON format only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content.strip()
        # JSON parse လုပ်မယ်
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            return {"error": "Could not parse AI response"}

        verdict = result.get("verdict", "Uncertain")
        confidence = result.get("confidence", 0)
        is_fake = verdict in ["Fake News", "Likely Fake"]

        return {
            "is_fake": is_fake,
            "verdict": verdict,
            "confidence": confidence,
            "reasons": result.get("reasons", []),
            "red_flags": result.get("red_flags", []),
            "credibility_score": result.get("credibility_score", 50),
            "summary": result.get("summary", ""),
            "recommendation": result.get("recommendation", ""),
            "source_url": source_url,
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}