"""
main.py - Akciğer Kanseri Risk Analiz API'si (FastAPI + XGBoost + SHAP + SQLAlchemy + JWT + Vision + RAG Asistan)

Çalıştırma:   uvicorn main:app --reload
Swagger:      http://127.0.0.1:8000/docs
"""
import json
import logging
import os
import pickle
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
import jwt
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from database import PredictionRecord, SessionLocal, UserInDB, init_db
from fusion import late_fusion
from xai_shap import RiskExplainer
from assistant import ClinicalAssistant

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ImageValidationError = ValueError
try:
    from vision import ImageRiskModel, ImageValidationError
except ImportError as _e:  # torch / Pillow yoksa tabular sistem çalışmaya devam eder
    ImageRiskModel = None
    VISION_IMPORT_ERROR = str(_e)
else:
    VISION_IMPORT_ERROR = None

log = logging.getLogger("akciger")
BASE_DIR = Path(__file__).resolve().parent

# ───────────────────────── Ayarlar ─────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(48)
    log.warning("SECRET_KEY tanımlı değil; geçici anahtar üretildi.")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
DOCTOR_INVITE_CODE = os.getenv("DOCTOR_INVITE_CODE")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

VISION_WEIGHTS_PATH = os.getenv("VISION_WEIGHTS", str(BASE_DIR / "vision_model.pt"))
FUSION_W_TABULAR = float(os.getenv("FUSION_W_TABULAR", "0.6"))
FUSION_W_IMAGE = float(os.getenv("FUSION_W_IMAGE", "0.4"))
FINAL_THRESHOLD = float(os.getenv("FINAL_THRESHOLD", "0.5"))

FEATURE_NAMES = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC_DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING",
    "COUGHING", "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN",
]

FIELD_TO_COLUMN = {
    "gender": "GENDER", "age": "AGE", "smoking": "SMOKING",
    "yellow_fingers": "YELLOW_FINGERS", "anxiety": "ANXIETY",
    "peer_pressure": "PEER_PRESSURE", "chronic_disease": "CHRONIC_DISEASE",
    "fatigue": "FATIGUE", "allergy": "ALLERGY", "wheezing": "WHEEZING",
    "alcohol": "ALCOHOL_CONSUMING", "coughing": "COUGHING",
    "shortness_of_breath": "SHORTNESS_OF_BREATH",
    "swallowing": "SWALLOWING_DIFFICULTY", "chest_pain": "CHEST_PAIN",
}


# ───────────────────────── Şemalar ─────────────────────────
class PatientData(BaseModel):
    gender: int = Field(ge=0, le=1)
    age: int = Field(ge=1, le=120)
    smoking: int = Field(ge=1, le=2)
    yellow_fingers: int = Field(ge=1, le=2)
    anxiety: int = Field(ge=1, le=2)
    peer_pressure: int = Field(ge=1, le=2)
    chronic_disease: int = Field(ge=1, le=2)
    fatigue: int = Field(ge=1, le=2)
    allergy: int = Field(ge=1, le=2)
    wheezing: int = Field(ge=1, le=2)
    alcohol: int = Field(ge=1, le=2)
    coughing: int = Field(ge=1, le=2)
    shortness_of_breath: int = Field(ge=1, le=2)
    swallowing: int = Field(ge=1, le=2)
    chest_pain: int = Field(ge=1, le=2)


class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=72)
    role: str = Field(default="patient", pattern="^(patient|doctor)$")
    invite_code: str | None = None


# ───────────────────────── Uygulama ömrü ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(BASE_DIR / "model.pkl", "rb") as f:
        app.state.model = pickle.load(f)
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        app.state.scaler = pickle.load(f)
    app.state.explainer = RiskExplainer(app.state.model, FEATURE_NAMES)
    app.state.vision = ImageRiskModel(VISION_WEIGHTS_PATH) if ImageRiskModel else None
    if app.state.vision is None:
        log.warning("Görüntü modülü devre dışı: %s", VISION_IMPORT_ERROR)

    # Faz 3: Akıllı Klinik Asistanı yükle (LLM yoksa şablon modu)
    app.state.assistant = ClinicalAssistant.from_env(BASE_DIR / "knowledge_base.json")

    init_db()
    yield


app = FastAPI(title="Akciğer Kanseri Risk Analiz API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────── Yardımcılar ─────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": username, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def get_optional_user(
    token: str | None = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> UserInDB | None:
    if not token:
        return None
    try:
        username = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]).get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Geçersiz veya süresi dolmuş token.",
                            headers={"WWW-Authenticate": "Bearer"})
    user = db.query(UserInDB).filter(UserInDB.username == username).first()
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Kullanıcı bulunamadı.")
    return user


def get_current_user(user: UserInDB | None = Depends(get_optional_user)) -> UserInDB:
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Giriş yapmanız gerekiyor.",
                            headers={"WWW-Authenticate": "Bearer"})
    return user


def to_frames(request: Request, data: PatientData):
    raw = data.model_dump()
    row = {FIELD_TO_COLUMN[k]: v for k, v in raw.items()}
    df = pd.DataFrame([row], columns=FEATURE_NAMES)
    scaled = pd.DataFrame(request.app.state.scaler.transform(df), columns=FEATURE_NAMES)
    return row, scaled


# ───────────────────────── Kimlik doğrulama ─────────────────────────
@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    if payload.role == "doctor" and (
        not DOCTOR_INVITE_CODE
        or not secrets.compare_digest(payload.invite_code or "", DOCTOR_INVITE_CODE)
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Doktor kaydı için geçerli davet kodu gerekli.")
    if db.query(UserInDB).filter(UserInDB.username == payload.username).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Bu kullanıcı adı zaten alınmış.")
    user = UserInDB(username=payload.username,
                    hashed_password=hash_password(payload.password), role=payload.role)
    db.add(user)
    db.commit()
    return {"message": "Kullanıcı oluşturuldu.", "username": user.username, "role": user.role}


@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserInDB).filter(UserInDB.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Hatalı kullanıcı adı veya şifre.",
                            headers={"WWW-Authenticate": "Bearer"})
    return {"access_token": create_access_token(user.username), "token_type": "bearer",
            "role": user.role}


# ───────────────────────── Tahmin ve açıklama ─────────────────────────
def build_explanation(request: Request, row: dict, scaled: pd.DataFrame) -> dict:
    result = request.app.state.explainer.explain_prediction(scaled)
    for item in result["contributions"]:
        item["value"] = row[item["feature"]]
    return result


@app.post("/predict")
def predict(data: PatientData, request: Request, db: Session = Depends(get_db),
            user: UserInDB | None = Depends(get_optional_user)):
    row, scaled = to_frames(request, data)
    model = request.app.state.model
    pred = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]
    risk_status = "YÜKSEK RİSK" if pred == 1 else "DÜŞÜK RİSK"

    saved = False
    if user is not None:
        db.add(PredictionRecord(
            user_id=user.id, risk_status=risk_status, risk_probability=float(proba[1]),
            inputs=row,
            shap_contributions=build_explanation(request, row, scaled)["contributions"],
        ))
        db.commit()
        saved = True

    cancer_probability = float(proba[1])
    return {
        "status": risk_status,
        "probability": f"%{cancer_probability * 100:.1f}",
        "risk_probability": round(cancer_probability, 4),
        "saved": saved,
    }


@app.post("/explain")
def explain(data: PatientData, request: Request):
    row, scaled = to_frames(request, data)
    return build_explanation(request, row, scaled)


# ───────────────────────── Faz 2: çok modlu tahmin + Faz 3 Asistan Entegrasyonu ─────────────────────────
@app.post("/predict-multimodal")
def predict_multimodal(
    request: Request,
    data: str = Form(..., description="Anket verisi: PatientData alanlarını içeren JSON metni"),
    image: UploadFile | None = File(None, description="PNG/JPEG röntgen veya BT (isteğe bağlı)"),
    db: Session = Depends(get_db),
    user: UserInDB | None = Depends(get_optional_user),
):
    try:
        patient = PatientData.model_validate_json(data)
    except ValidationError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=json.loads(e.json()))
    
    row, scaled = to_frames(request, patient)
    tabular_prob = float(request.app.state.model.predict_proba(scaled)[0][1])

    image_info = {"provided": image is not None, "available": False,
                  "risk_probability": None, "note": None}
    image_prob = None
    if image is not None:
        raw = image.file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Dosya 10 MB sınırını aşıyor.")
        vision = request.app.state.vision
        if vision is None or not vision.available:
            reason = vision.reason if vision else (VISION_IMPORT_ERROR or "görüntü modülü yüklenemedi")
            image_info["note"] = (
                "Görüntü modeli kullanılamıyor; nihai skor yalnızca anket verisinden hesaplandı. "
                + (reason or "")
            )
        else:
            try:
                image_prob = vision.predict(raw)
            except ImageValidationError as e:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
            image_info.update(available=True, risk_probability=round(image_prob, 4))

    fusion = late_fusion(tabular_prob, image_prob, FUSION_W_TABULAR, FUSION_W_IMAGE, FINAL_THRESHOLD)
    explanation = build_explanation(request, row, scaled)

    # Faz 3: Asistan Tavsiyesi Üretimi
    assistant = request.app.state.assistant
    assistant_advice = assistant.advise(fusion, explanation, row, image_info) if assistant else None

    saved = False
    if user is not None:
        db.add(PredictionRecord(
            user_id=user.id, risk_status=fusion["final_status"], risk_probability=tabular_prob,
            inputs=row, shap_contributions=explanation["contributions"],
            image_probability=image_prob, final_risk_score=fusion["final_risk_score"],
            fusion_info={k: fusion[k] for k in ("mode", "weights", "threshold")},
        ))
        db.commit()
        saved = True

    return {
        "final_risk_score": fusion["final_risk_score"],
        "final_status": fusion["final_status"],
        "final_risk_percent": f"%{fusion['final_risk_score'] * 100:.1f}",
        "mode": fusion["mode"],
        "weights": fusion["weights"],
        "components": {
            "tabular": {"risk_probability": round(tabular_prob, 4)},
            "image": image_info,
        },
        "explanation": explanation,
        "explanation_scope": "tabular",
        "ai_assistant_advice": assistant_advice,
        "saved": saved,
        "disclaimer": "Bu karar destek amaçlıdır; tanı yerine geçmez. Bir hekime başvurun.",
    }


@app.get("/health")
def health(request: Request):
    vision = request.app.state.vision
    assistant = request.app.state.assistant
    return {
        "status": "ok",
        "vision_available": bool(vision and vision.available),
        "vision_note": (vision.reason if vision else VISION_IMPORT_ERROR),
        "assistant_mode": "template" if (assistant and not assistant.provider) else ("llm" if assistant else "disabled"),
    }


# ───────────────────────── Geçmiş ─────────────────────────
def _serialize(records):
    return [{
        "id": r.id, "created_at": r.created_at.isoformat(), "risk_status": r.risk_status,
        "risk_probability": r.risk_probability,
        "risk_score": r.final_risk_score if r.final_risk_score is not None else r.risk_probability,
        "image_used": r.image_probability is not None,
    } for r in records]


@app.get("/history")
def my_history(user: UserInDB = Depends(get_current_user), db: Session = Depends(get_db)):
    records = (db.query(PredictionRecord).filter(PredictionRecord.user_id == user.id)
               .order_by(PredictionRecord.created_at).all())
    return _serialize(records)


@app.get("/patients/{username}/history")
def patient_history(username: str, user: UserInDB = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    if user.role != "doctor":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Bu işlem yalnızca doktorlar içindir.")
    patient = db.query(UserInDB).filter(UserInDB.username == username).first()
    if not patient:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Hasta bulunamadı.")
    records = (db.query(PredictionRecord).filter(PredictionRecord.user_id == patient.id)
               .order_by(PredictionRecord.created_at).all())
    return _serialize(records)
