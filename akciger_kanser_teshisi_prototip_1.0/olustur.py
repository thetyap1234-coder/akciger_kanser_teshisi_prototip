import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
df = pd.DataFrame(np.random.randint(1, 3, (100, 15)), columns=[
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC_DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING",
    "COUGHING", "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN"
])
y = np.random.randint(0, 2, 100)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Dosyalar basariyla olusturuldu!")