import numpy as np
import shap


class RiskExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = list(feature_names)
        # XGBoost için TreeExplainer (log-odds / raw margin).
        self.explainer = shap.TreeExplainer(self.model)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def explain_prediction(self, scaled_input_df):
        """Ölçeklenmiş tek satır için SHAP katkılarını (pozitif sınıf) döndürür."""
        sv = self.explainer(scaled_input_df)
        values = np.asarray(sv.values[0], dtype=float)
        # shap sürümüne göre (özellik,) ya da (özellik, sınıf) gelebilir.
        if values.ndim == 2:
            values = values[:, 1] if values.shape[1] > 1 else values[:, 0]
        values = np.ravel(values)

        base_raw = np.ravel(np.asarray(sv.base_values[0], dtype=float))
        base_value = float(base_raw[-1] if base_raw.size else 0.0)
        proba = float(self.model.predict_proba(scaled_input_df)[0][1])

        if len(values) != len(self.feature_names):
            raise ValueError(
                f"SHAP boyutu ({len(values)}) özellik sayısıyla ({len(self.feature_names)}) uyuşmuyor."
            )

        contributions = [
            {
                "feature": name,
                "shap_value": round(float(v), 4),
                "direction": "risk_artirir" if v > 0 else "risk_azaltir",
            }
            for name, v in zip(self.feature_names, values)
        ]
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

        return {
            "risk_probability": round(proba, 4),
            "base_value": round(base_value, 4),
            "base_probability": round(self._sigmoid(base_value), 4),
            "unit": "log-odds",
            "contributions": contributions,
        }
