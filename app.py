#-------cell1
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install diffusers==0.25.0 transformers==4.38.2 accelerate==0.27.2 \
             huggingface_hub==0.20.3 safetensors==0.4.2 controlnet_aux==0.0.6 \
             opencv-python ipywidgets Pillow requests flask flask-cors pyngrok --quiet



#---------cell2
!pip install -q \
    accelerate==0.30.0 \
    peft==0.10.0 \
    diffusers==0.25.0 \
    transformers==4.38.2


#----------cell3
import base64, io, re, threading, urllib.parse
from contextlib import nullcontext
import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.feature import canny as sk_canny

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
)

from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok, conf

# ── ngrok auth — get yours free at https://dashboard.ngrok.com ───────
NGROK_AUTH_TOKEN = "33MPZv1Pflzl7Kh0DLtGbaMG0jQ_4iRBJ6z1ULyAhvGfkmwcM"   # ← REPLACE THIS

# ── Device setup ─────────────────────────────────────────────────────
COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.float16 if COMPUTE_DEVICE == "cuda" else torch.float32
DETECTOR_DEVICE = 0 if torch.cuda.is_available() else -1
autocast       = torch.autocast("cuda") if COMPUTE_DEVICE == "cuda" else nullcontext()
print(f"Device: {COMPUTE_DEVICE}")

# ── Load datasets ─────────────────────────────────────────────────────
amazon_df = pd.read_csv("/content/amazon.csv")
ikea_df   = pd.read_csv("/content/ikea.csv")
print("Datasets loaded ✅")

# ── Load models ───────────────────────────────────────────────────────
print("Loading Object Detection (DETR)...")
detector = hf_pipeline(
    "object-detection",
    model="facebook/detr-resnet-101",
    device=DETECTOR_DEVICE
)
print("DETR ready ✅")

print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=DTYPE
).to(COMPUTE_DEVICE)

print("Loading Stable Diffusion (Realistic Vision)...")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    torch_dtype=DTYPE
).to(COMPUTE_DEVICE)
pipe.enable_attention_slicing()
print("Diffusion pipeline ready ✅")

print("Loading LLM (Flan-T5)...")
tokenizer  = AutoTokenizer.from_pretrained("google/flan-t5-large")
llm_model  = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to("cpu")
print("LLM ready ✅")



#-------------cell4
# ═════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════
import os
def b64_to_pil(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
VR_OUTPUT_DIR = "/content/vr_outputs"
os.makedirs(VR_OUTPUT_DIR, exist_ok=True)

def save_vr_image(img: Image.Image):
    filename = f"vr_{int(torch.randint(0,999999,(1,)).item())}.png"
    path = os.path.join(VR_OUTPUT_DIR, filename)
    img.save(path)
    return filename

def preprocess_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB").resize((512, 512))

def generate_canny(image: Image.Image) -> Image.Image:
    arr = np.array(image)
    gray = np.mean(arr, axis=2) if arr.ndim == 3 else arr
    edges = sk_canny(gray, low_threshold=50, high_threshold=150)
    return Image.fromarray((edges * 255).astype(np.uint8))

def generate_llm_text(prompt: str, max_length: int = 400) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compress_for_diffusion(text: str) -> str:
    prompt = f"""
    Summarize this interior design plan into a short
    diffusion-friendly description under 60 words.
    Focus only on visual changes.
    {text}
    """
    return generate_llm_text(prompt, max_length=120)

def generate_image(base_image: Image.Image, concept_text: str) -> Image.Image:
    canny_image = generate_canny(base_image)
    positive_prompt = f"""
    Ultra realistic DSLR interior photograph,
    preserve structure and layout,
    {concept_text}
    """
    negative_prompt = "cartoon, anime, distorted layout, unrealistic lighting"
    with autocast:
        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            control_image=canny_image,
            strength=0.75,
            guidance_scale=8.5,
            num_inference_steps=25,
            controlnet_conditioning_scale=0.6
        )
    return result.images[0]

def refine_image_fn(current_pil: Image.Image, current_concept: str, refinement_text: str):
    full_concept = current_concept + "\n" + refinement_text
    short_text   = compress_for_diffusion(full_concept)
    canny_image  = generate_canny(current_pil)
    positive_prompt = f"""
    Ultra realistic DSLR interior photograph,
    preserve structure and layout,
    preserve previously added elements,
    apply only these refinements:
    {short_text}
    """
    negative_prompt = "cartoon, anime, distorted layout, remove furniture, unrealistic lighting"
    with autocast:
        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=current_pil,
            control_image=canny_image,
            strength=0.6,
            guidance_scale=9.0,
            num_inference_steps=30,
            controlnet_conditioning_scale=0.7
        )
    return result.images[0], full_concept

# ═════════════════════════════════════════════════════════════════════
# AGENTS
# ═════════════════════════════════════════════════════════════════════

class DesignStrategistAgent:
    STYLE_MAP = {
        "Modern":             {"wall_color": "warm beige walls",            "materials": "matte black accents, light wood furniture", "patterns": "minimal patterns, clean lines",         "lighting": "warm recessed ceiling lights",        "decor": "abstract art frames"},
        "Indian Traditional": {"wall_color": "deep maroon or mustard walls", "materials": "teak wood furniture, brass decor",           "patterns": "floral or ethnic patterned textiles",   "lighting": "warm yellow lamps",                   "decor": "traditional framed artwork"},
        "Scandinavian":       {"wall_color": "pure white walls",             "materials": "light oak wood, neutral fabrics",            "patterns": "minimal geometric rug",                 "lighting": "soft diffused daylight",              "decor": "indoor green plants"},
        "Luxury":             {"wall_color": "taupe or cream walls",         "materials": "velvet upholstery, marble accents",          "patterns": "rich layered textiles",                 "lighting": "layered ambient + pendant lighting",  "decor": "large statement artwork"},
        "Minimal":            {"wall_color": "off white walls",              "materials": "simple wood furniture",                      "patterns": "very minimal patterns",                 "lighting": "soft natural lighting",               "decor": "limited decor pieces"},
        "Bohemian":           {"wall_color": "warm earthy tones",            "materials": "rattan, wood, mixed textures",               "patterns": "colorful layered textiles",             "lighting": "warm hanging lights",                 "decor": "plants and artistic decor"},
    }
    VIBE_EFFECT = {
        "Calm":    "soft neutral tones, diffused lighting",
        "Bold":    "high contrast colors, dramatic lighting",
        "Cozy":    "warm ambient lighting, soft textiles, layered fabrics",
        "Elegant": "refined finishes, subtle luxury details",
    }

    def generate_concept(self, style: str, purpose: str, vibe: str) -> str:
        s = self.STYLE_MAP.get(style, self.STYLE_MAP["Modern"])
        vibe_prompt = self.VIBE_EFFECT.get(vibe, "")
        return (
            f"Room Purpose: {purpose}\n"
            f"Desired Vibe: {vibe}\n\n"
            f"{vibe_prompt},\n"
            f"Change wall color to {s['wall_color']},\n"
            f"use {s['materials']},\n"
            f"add {s['patterns']},\n"
            f"install {s['lighting']},\n"
            f"include {s['decor']}"
        )


class SpaceOptimizationAgent:
    def analyze(self, image: Image.Image) -> dict:
        w, h = image.size
        area_score = (w * h) / 50000
        if area_score < 3:
            room_size, walking_space = "Compact", 0.8
        elif area_score < 6:
            room_size, walking_space = "Medium", 1.1
        else:
            room_size, walking_space = "Spacious", 1.5
        return {"room_size": room_size, "estimated_walking_space_m": walking_space, "is_space_sufficient": walking_space >= 0.9}

    def suggest_layout(self, room_type: str) -> list:
        layout_map = {
            "Bedroom":     ["Place bed against longest wall", "Maintain 0.9m minimum walking clearance", "Keep wardrobe near corner to maximize space", "Avoid blocking window light"],
            "Living Room": ["Place sofa facing focal wall or TV unit", "Maintain central walking path", "Keep coffee table within 45cm of sofa", "Avoid blocking balcony or window light"],
            "Study Room":  ["Place study desk near window for natural light", "Maintain ergonomic chair spacing", "Keep bookshelf against wall", "Maintain clutter-free walking area"],
            "Office":      ["Position desk facing entry or window", "Maintain 1m chair movement clearance", "Keep storage cabinets against walls", "Avoid blocking electrical outlets"],
            "Kitchen":     ["Maintain work triangle between sink, stove, fridge", "Keep minimum 1m circulation space", "Avoid blocking ventilation areas", "Place dining near natural light"],
        }
        return layout_map.get(room_type, ["Optimize layout for comfort and circulation"])


# ── Shopping constants ────────────────────────────────────────────────

JUNK_TOKENS = {
    "bedroom","living room","study room","office","kitchen","furniture","answer",
    "room","type","item","accents","materials","patterns","lighting","decor","style",
    "design","concept","color","scheme","wall","floor","ceiling","modern","minimal",
    "luxury","traditional","scandinavian","bohemian","indian","warm","cool","ambient",
    "natural","artificial","soft","bright","dark","light","matte","glossy","texture",
    "finish","wood","metal","glass","fabric","velvet","marble","oak","teak","brass",
    "black","white","beige","yes","no","none","null","n/a","na","list",
    "the","and","for","with","from","into","onto",
}

SYNONYM_MAP = {
    "couch":"sofa","loveseat":"sofa","sectional":"sofa","settee":"sofa",
    "armchair":"chair","accent chair":"chair","stool":"chair",
    "bookcase":"bookshelf","book shelf":"bookshelf",
    "cupboard":"wardrobe","closet":"wardrobe","almirah":"wardrobe",
    "dresser drawer":"dresser","chest of drawers":"dresser",
    "rug":"carpet","area rug":"carpet",
    "drapes":"curtains","drape":"curtains","curtain":"curtains",
    "floor lamp":"table lamp","desk lamp":"table lamp","lamp":"table lamp",
    "tv stand":"tv unit","tv cabinet":"tv unit","media unit":"tv unit","entertainment unit":"tv unit",
    "side table":"nightstand","bedside table":"nightstand",
    "study table":"desk","work desk":"desk","writing table":"desk",
    "dining set":"dining table",
    "plant":"indoor plant","potted plant":"indoor plant","indoor plants":"indoor plant",
    "picture frame":"art frame","wall art":"art frame","painting":"art frame",
}

FALLBACK_PRICE_RANGE = {
    "bed":(18000,20000),"wardrobe":(15000,60000),"sofa":(20000,30000),
    "curtains":(2000,8000),"carpet":(1000,2000),"chair":(1500,4000),
    "desk":(5000,25000),"coffee table":(4000,20000),"tv unit":(8000,35000),
    "cabinet":(10000,40000),"bookshelf":(6000,15000),"office chair":(1000,2000),
    "dining table":(5000,50000),"bar stool":(1000,2000),"refrigerator":(10000,60000),
    "nightstand":(3000,15000),"dresser":(3000,10000),"table lamp":(1000,3000),
    "indoor plant":(500,2000),"art frame":(800,5000),
}

STYLE_MULTIPLIER = {
    "Modern":1.0,"Luxury":1.4,"Minimal":0.9,
    "Indian Traditional":1.2,"Bohemian":1.1,"Scandinavian":1.0,
}

ROOM_ALLOWED_ITEMS = {
    "Bedroom":     {"bed","wardrobe","nightstand","dresser","table lamp","carpet","curtains","art frame"},
    "Living Room": {"sofa","tv unit","coffee table","carpet","curtains","chair","art frame","indoor plant"},
    "Study Room":  {"desk","chair","bookshelf","office chair","table lamp","carpet","curtains","indoor plant"},
    "Office":      {"desk","office chair","cabinet","bookshelf","table lamp"},
    "Kitchen":     {"dining table","bar stool","refrigerator","cabinet"},
}

DATASET_PRICE_MIN = 500
DATASET_PRICE_MAX = 500000


def _normalize_item(raw: str):
    item = raw.lower().strip()
    item = re.sub(r"[\[\]\"\'(){}<>]", "", item)
    item = re.sub(r"\s+", " ", item).strip()
    if len(item) < 3 or item.isdigit() or item in JUNK_TOKENS:
        return None
    return SYNONYM_MAP.get(item, item)

def normalize_item_list(raw_items: list) -> list:
    seen, cleaned = set(), []
    for raw in raw_items:
        for sub in raw.split(","):
            n = _normalize_item(sub)
            if n and n not in seen:
                seen.add(n); cleaned.append(n)
    return cleaned

def match_price_from_dataset(item: str) -> float:
    prices = []
    try:
        prices += pd.to_numeric(
            amazon_df[amazon_df["title"].str.contains(item, case=False, na=False, regex=False)]["price"],
            errors="coerce").dropna().tolist()
    except Exception: pass
    try:
        prices += pd.to_numeric(
            ikea_df[ikea_df["name"].str.contains(item, case=False, na=False, regex=False)]["price"],
            errors="coerce").dropna().tolist()
    except Exception: pass
    prices = [p for p in prices if DATASET_PRICE_MIN < p < DATASET_PRICE_MAX]
    return float(np.median(prices[:20])) if prices else None

def get_item_price(item: str, style: str):
    multiplier = STYLE_MULTIPLIER.get(style, 1.0)
    price = match_price_from_dataset(item)
    if price is not None:
        return round(price * multiplier)
    key = item
    if key not in FALLBACK_PRICE_RANGE:
        for k in FALLBACK_PRICE_RANGE:
            if k in item or item in k:
                key = k; break
    if key in FALLBACK_PRICE_RANGE:
        lo, hi = FALLBACK_PRICE_RANGE[key]
        return round(((lo + hi) / 2.0) * multiplier)
    return None

def scale_to_budget(price_map: dict, budget_min: float, budget_max: float) -> dict:
    total_raw = sum(price_map.values())
    if total_raw <= 0:
        return {k: 0 for k in price_map}
    scale  = ((budget_min + budget_max) / 2.0) / total_raw
    scaled = {item: max(1, round(p * scale)) for item, p in price_map.items()}
    final  = sum(scaled.values())
    top    = max(scaled, key=scaled.get)
    if final < budget_min:
        scaled[top] += (budget_min - final)
    elif final > budget_max:
        scaled[top] = max(1, scaled[top] - (final - budget_max))
    return scaled


class ShoppingAssistantAgent:
    LABEL_NORMALIZATION = {"couch":"sofa","chair":"chair","dining table":"dining table","bed":"bed","tv":"tv unit","potted plant":"indoor plant","refrigerator":"refrigerator"}
    FURNITURE_CLASSES   = ["couch","chair","dining table","bed","tv","potted plant","refrigerator"]

    def detect_furniture(self, image: Image.Image) -> list:
        return list({
            self.LABEL_NORMALIZATION[o["label"]]
            for o in detector(image)
            if o["score"] >= 0.5 and o["label"] in self.FURNITURE_CLASSES
        })

    def extract_furniture_from_concept(self, concept_text: str, room_type: str) -> list:
      prompt = f"""
      You are an interior design expert.

      Room type: {room_type}

      From the description below, extract ONLY tangible purchasable
      furniture and decor items appropriate for a {room_type}.

      Do NOT include:
      - room names
      - styles
      - materials
      - colors
      - abstract words

      Only include items like:
      bed, wardrobe, sofa, chair, desk, curtains,
      carpet, nightstand, dresser, tv unit, cabinet, art frame.

      Return strictly comma separated nouns only.
      No explanation.

      Description:
      {concept_text}
      """

      inputs = tokenizer(prompt, return_tensors="pt")
      with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_length=120)

      response = tokenizer.decode(outputs[0], skip_special_tokens=True)
      response = response.lower().replace("\n", ",").replace(" and ", ",")
      return [x.strip() for x in response.split(",")]

    def generate_store_links(self, item: str, style: str) -> dict:
        enc = urllib.parse.quote(f"{style} {item}")
        return {
            "Amazon":    f"https://www.amazon.in/s?k={enc}",
            "IKEA":      f"https://www.ikea.com/in/en/search/?q={enc}",
            "Pepperfry": f"https://www.pepperfry.com/site_product/search?q={enc}",
        }

    def generate_shopping_and_budget(self, image, style, concept_text, room_type, budget_min, budget_max):
        all_raw    = self.detect_furniture(image) + self.extract_furniture_from_concept(concept_text, room_type)
        allowed    = ROOM_ALLOWED_ITEMS.get(room_type, set())
        items      = [i for i in normalize_item_list(all_raw) if i in allowed]
        if not items:
            return [], 0
        price_map  = {i: p for i in items if (p := get_item_price(i, style)) is not None and p > 0}
        if not price_map:
            return [], 0
        scaled = scale_to_budget(price_map, budget_min, budget_max)
        return [{"item": i, "price": p, "stores": self.generate_store_links(i, style)} for i, p in scaled.items()], sum(scaled.values())


class InteriorCoordinator:
    def __init__(self):
        self.design = DesignStrategistAgent()
        self.space  = SpaceOptimizationAgent()
        self.shop   = ShoppingAssistantAgent()

    def run(self, image, style, room_type, purpose, vibe):
        concept  = self.design.generate_concept(style, purpose, vibe)
        geometry = self.space.analyze(image)
        layout   = self.space.suggest_layout(room_type)
        return concept, geometry, layout


coordinator = InteriorCoordinator()
print("All agents initialised ✅")




# ═════════════════════════════════════════════════════════════════════
# START SERVER + NGROK
# ═════════════════════════════════════════════════════════════════════

def start_flask():
    app.run(host="0.0.0.0", port=5000, use_reloader=False, debug=False)

# Configure ngrok
conf.get_default().auth_token = '2u1UpU39fulSYhpM6cLF4eMu3o3_49JxLmttZLTj38t8ZFqYY'
ngrok.kill()   # kill any stale tunnels

# Start Flask in background thread
threading.Thread(target=start_flask, daemon=True).start()

# Open public tunnel
public_url = ngrok.connect(5000, bind_tls=True).public_url

print("\n" + "═" * 60)
print("  ✅  InteriorAI Pro backend is LIVE!")
print(f"  🌐  Public URL : {public_url}")
print()
print("  📋  In your frontend HTML, set:")
print(f'      const BACKEND_URL = "{public_url}";')
print("      OR paste it into the Backend URL field in the header")
print("═" * 60 + "\n")
