"""
fusion.py - Geç Entegrasyon (Late Fusion): tabular XGBoost skoru + görüntü CNN skoru.

Saf Python, framework bağımlılığı yok -> kolay test edilir.
UYARI: Ağırlıklı ortalama, iki skorun KALİBRE (gerçek olasılığı yansıtan) olduğunu varsayar.
Ağırlıkları keyfi seçmek yerine doğrulama verisinde (ör. lojistik regresyonla) öğrenmek idealdir.
"""
from __future__ import annotations


def late_fusion(
    tabular_prob: float,
    image_prob: float | None,
    w_tabular: float = 0.6,
    w_image: float = 0.4,
    threshold: float = 0.5,
) -> dict:
    """Ağırlıklı ortalama ile nihai riski üretir.

    image_prob None ise (görüntü yok / model kullanılamıyor) yalnızca tabular skor döner.
    """
    if w_tabular < 0 or w_image < 0 or (w_tabular + w_image) == 0:
        raise ValueError("Füzyon ağırlıkları negatif olamaz ve toplamı sıfır olamaz.")

    if image_prob is None:
        final, mode = tabular_prob, "tabular_only"
        weights = {"tabular": 1.0, "image": 0.0}
    else:
        total = w_tabular + w_image
        weights = {"tabular": w_tabular / total, "image": w_image / total}
        final = weights["tabular"] * tabular_prob + weights["image"] * image_prob
        mode = "multimodal"

    final = min(max(final, 0.0), 1.0)
    return {
        "final_risk_score": round(final, 4),
        "final_status": "YÜKSEK RİSK" if final >= threshold else "DÜŞÜK RİSK",
        "mode": mode,
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "threshold": threshold,
    }
