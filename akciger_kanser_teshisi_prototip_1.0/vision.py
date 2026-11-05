"""
vision.py - Görüntü (röntgen/BT) risk modeli: PyTorch ResNet50 + güvenli yükleme doğrulaması.

ÖNEMLİ: ImageNet ağırlıklı ResNet50 tek başına akciğer kanseri tespit ETMEZ.
Bu modül, akciğer verisiyle İNCE AYARLI (fine-tuned) bir ağırlık dosyası bekler
(`vision_model.pt`, bkz. train_vision.py). Dosya yoksa model "kullanılamaz" sayılır ve
sistem uydurma bir skor üretmek yerine yalnızca tabular skora döner.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

log = logging.getLogger("akciger.vision")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024          # 10 MB
ALLOWED_FORMATS = {"PNG", "JPEG"}
Image.MAX_IMAGE_PIXELS = 50_000_000          # "decompression bomb" koruması

# Eğitimde (train_vision.py) ve çıkarımda AYNI dönüşüm kullanılmalıdır.
# Tüm akciğer alanını korumak için kırpma yok, doğrudan 224x224'e yeniden boyutlandırılır.
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageValidationError(ValueError):
    """Yüklenen dosya geçerli bir PNG/JPEG değil veya sınırları aşıyor."""


def build_resnet50(pretrained: bool = False) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1))  # tek logit
    return model


def load_and_validate_image(data: bytes) -> Image.Image:
    """Dosya uzantısına/Content-Type'a güvenmez; içeriği gerçekten çözerek doğrular."""
    if not data:
        raise ImageValidationError("Boş dosya.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("Dosya 10 MB sınırını aşıyor.")
    try:
        probe = Image.open(io.BytesIO(data))
        if probe.format not in ALLOWED_FORMATS:
            raise ImageValidationError("Yalnızca PNG veya JPEG kabul edilir.")
        probe.verify()                                   # bozuk dosyayı yakalar
        img = Image.open(io.BytesIO(data))               # verify() sonrası yeniden aç
        return img.convert("RGB")                        # gri tonlu röntgeni 3 kanala çevir
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as e:
        raise ImageValidationError("Görüntü okunamadı veya geçersiz.") from e


class ImageRiskModel:
    """Uygulama açılışında BİR KEZ oluşturulur."""

    def __init__(self, weights_path: str | Path):
        self.available = False
        self.reason: str | None = None
        self.model: nn.Module | None = None
        path = Path(weights_path)
        if not path.exists():
            self.reason = f"Ağırlık dosyası bulunamadı: {path.name}"
            log.warning("Görüntü modeli devre dışı: %s", self.reason)
            return
        try:
            model = build_resnet50(pretrained=False)
            # weights_only=True: pickle çalıştırmadan yalnızca tensörleri yükler (daha güvenli).
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self.model = model
            self.available = True
        except Exception as e:  # noqa: BLE001
            self.reason = f"Ağırlık dosyası yüklenemedi: {type(e).__name__}"
            log.error("Görüntü modeli yüklenemedi: %s", e)

    @torch.inference_mode()
    def predict(self, data: bytes) -> float:
        """Görüntüdeki 'pozitif' sınıf olasılığını (0-1) döndürür."""
        if not self.available or self.model is None:
            raise RuntimeError("Görüntü modeli kullanılamıyor.")
        img = load_and_validate_image(data)
        x = EVAL_TRANSFORM(img).unsqueeze(0)
        logit = self.model(x).squeeze()
        return float(torch.sigmoid(logit).item())
