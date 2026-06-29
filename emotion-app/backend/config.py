"""EmotionLens — shared configuration & field-name constants (single source of truth).

All field names used in mode_output dicts MUST match the JS frontend reads exactly.
When adding a new lens, register its id/title/icon here and in the frontend LENS_DEFS.
"""
import os

# ═══════════════════════════════════════════════════════════════
# 0. Emotion ↔ Valence/Arousal mapping
# ═══════════════════════════════════════════════════════════════
EMOTION_VA = {
    "happiness": {"valence": 1.0,  "arousal": 0.5},
    "surprise":  {"valence": 0.2,  "arousal": 0.8},
    "neutral":   {"valence": 0.0,  "arousal": 0.1},
    "sadness":   {"valence": -0.6, "arousal": 0.3},
    "fear":      {"valence": -0.7, "arousal": 0.85},
    "disgust":   {"valence": -0.7, "arousal": 0.6},
    "anger":     {"valence": -0.9, "arousal": 0.9},
}

EMO_CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]

# ═══════════════════════════════════════════════════════════════
# 1. Engine parameters
# ═══════════════════════════════════════════════════════════════
DET_CONF   = 0.6
MIN_FACE_PX = 60
EMA_GAMMA  = 0.6         # temporal smoothing weight for face tracker

# ═══════════════════════════════════════════════════════════════
# 1.5 Model backend (switch via env var EMO_MODEL)
# ═══════════════════════════════════════════════════════════════
#   classmate     — your custom ResNet18 (default)
#   efficientnet  — your EfficientNet-B0
#   <path>        — any custom checkpoint path
MODEL_BACKEND = os.getenv("EMO_MODEL", "classmate")

# Built-in model registry: name → {architecture, checkpoint_relative_path, state_dict_key}
# Relative paths are resolved from engine_fer.py's directory (backend/).
MODEL_REGISTRY = {
    "classmate": {
        "architecture": "resnet18",
        "path": os.path.join(os.path.dirname(__file__), "..", "..", "runs", "classmate_model", "best.pt"),
        "state_key": "model_state",   # key in checkpoint dict
        "label": "Classmate (ResNet18)",
    },
    "efficientnet": {
        "architecture": "efficientnet_b0",
        "path": os.path.join(os.path.dirname(__file__), "..", "..", "runs", "efficientnet_b0_20260615_235918", "best.pt"),
        "state_key": "model",
        "label": "EfficientNet-B0",
    },
}

# ═══════════════════════════════════════════════════════════════
# 2. Lens registry  (id → {title, icon, has_timer, durations})
# ═══════════════════════════════════════════════════════════════
LENS_DEFS = {
    "m0": {"title": "Live",                 "icon": "📊", "has_timer": False},
    "m1": {"title": "Cafeteria Mood",       "icon": "🍽️", "has_timer": True,  "durations": [30, 60, 180, 600]},
    "m2": {"title": "CODE RED",             "icon": "🚨", "has_timer": False},
    "m3": {"title": "Audience Reactions",   "icon": "🎬", "has_timer": False},
    "m4": {"title": "Speech Coach",         "icon": "🎤", "has_timer": True,  "durations": [10, 30, 60, 180]},
    "m5": {"title": "Mimic Game",           "icon": "🎮", "has_timer": False},
}

# ═══════════════════════════════════════════════════════════════
# 3. L1 · Cafeteria Mood
# ═══════════════════════════════════════════════════════════════
L1_MIN_FACES = 1          # minimum faces to consider session valid

# Verdict thresholds (English)
L1_VERDICT_HI = 75        # >= 75  → "Very satisfied 👍"
L1_VERDICT_MD = 55        # >= 55  → "Satisfied"
L1_VERDICT_LO = 40        # >= 40  → "Mixed" ;  < 40 → "Unsatisfied"

# ═══════════════════════════════════════════════════════════════
# 4. L2 · CODE RED  (more sensitive than original M2)
# ═══════════════════════════════════════════════════════════════
L2_W1        = 0.6        # weight for negative-emotion probability
L2_W2        = 0.4        # weight for arousal
L2_R_SPIKE   = 0.12       # spike threshold  (lower = easier trigger)
L2_R_HI      = 0.30       # sustained-high threshold
L2_SUSTAIN_N = 3          # consecutive frames for sustained trigger
L2_COOLDOWN  = 2.5        # cooldown seconds between alarms

# Banner lines — pool of English tactical phrases
L2_BANNER_POOL = [
    "Draw — take the shot!",
    "Suspect lunging — restrain!",
    "High threat — defend now!",
    "Officer needs assistance!",
    "Drop the weapon — now!",
]
# Milder pool (swap in for sensitive audiences)
L2_BANNER_MILD = [
    "Heads up — agitation spike",
    "Warning — elevated stress detected",
    "Alert — sudden mood change",
    "Attention — emotional surge",
]

# ═══════════════════════════════════════════════════════════════
# 5. L3 · Audience Reactions
# ═══════════════════════════════════════════════════════════════
L3_DEBOUNCE_K = 4         # consecutive frames to confirm dominant switch

# ═══════════════════════════════════════════════════════════════
# 6. L4 · Speech Coach
# ═══════════════════════════════════════════════════════════════
L4_POS_V_THRESHOLD = 0.2  # valence > this → "positive" frame

# Rule-based fallback advice (English)
def rule_based_advice(metrics: dict) -> str:
    """Generate rule-based English coaching advice from metrics."""
    anx = metrics.get("anxiety", 0)
    exp = metrics.get("expressiveness", 0)
    pos = metrics.get("positivity", 0)
    tips = []
    if anx > 0.3:
        tips.append("Slow down and breathe — your anxiety is showing.")
    if exp < 0.2:
        tips.append("Add more facial expression and energy to your delivery.")
    if pos > 0.5:
        tips.append("Great presence and positivity — keep it up!")
    if not tips:
        tips.append("Keep practicing — you're building a solid foundation.")
    return " ".join(tips)

# ═══════════════════════════════════════════════════════════════
# 7. L5 · Mimic Game
# ═══════════════════════════════════════════════════════════════
L5_HIT_THRESHOLD = 0.6    # p_target > this → hit
L5_TIME_PER_TARGET = 5    # seconds per round
L5_TOTAL_ROUNDS    = 10   # game over after this many rounds
L5_TOTAL_TIME      = 60   # … or this many seconds total
L5_MAX_SCORE       = 100  # capped max score
L5_SCORE_MULTIPLIER = 2   # 2× score boost (hits: + (10+combo*2)*2)
