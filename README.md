# Heart Disease Prediction — End-to-End ML Deployment

A machine learning system that predicts whether a patient is at risk of heart
disease based on clinical parameters, exposed as a REST API built with
Flask, version-controlled on GitHub, and deployed live on Render.

**Live App URL:** `https://<your-app-name>.onrender.com`  ← replace after deploying (see Task 4 below)

---

## 📊 Dataset

Heart Disease dataset (UCI Cleveland heart-disease dataset — same schema as
the Kaggle "Heart Disease Dataset" by johnsmith88), 303 patient records,
13 clinical features + 1 target column.

| Feature | Description |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| restecg | Resting electrocardiographic results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes, 0 = no) |
| oldpeak | ST depression induced by exercise relative to rest |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels (0–3) colored by fluoroscopy |
| thal | 1 = normal, 2 = fixed defect, 3 = reversible defect |
| **target** | 1 = heart disease present, 0 = no heart disease |

---

## 🗂 Repository Structure

```
HeartDiseaseDeployment/
│
├── app.py                 # Flask REST API
├── model.pkl               # Trained Random Forest model
├── feature_names.pkl       # Feature order used at training time
├── requirements.txt        # Python dependencies
├── README.md                # This file
├── train_model.py          # Data preprocessing + model training script
├── heart.csv                 # Dataset
├── Procfile                  # Render/Gunicorn start command
├── templates/
│   └── index.html            # Optional simple web UI
└── static/                   # (unused, reserved for assets)
```

---

## Task 1 — Data Understanding and Preprocessing

`train_model.py` performs the following:
1. Loads `heart.csv` with Pandas.
2. Prints the first five records (`df.head()`).
3. Identifies the 13 numerical/clinical features and the target variable (`target`).
4. Checks for missing values — the dataset has **zero** missing values.
5. Splits the data into 80% training / 20% testing using `train_test_split`
   (stratified on the target to preserve class balance).

## Task 2 — Model Development

- Algorithm: **Random Forest Classifier** (`n_estimators=200, max_depth=6`).
- Evaluation metric: **Accuracy Score** → **83.61%** on the held-out test set.
- The trained model is serialized with **Joblib** to `model.pkl`
  (feature ordering is saved separately to `feature_names.pkl` so the API
  can reconstruct input rows correctly).

To retrain:
```bash
pip install -r requirements.txt
python train_model.py
```

## Task 3 — API Development

`app.py` is a Flask REST API with three routes:

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Landing page / basic info |
| GET | `/health` | Health check (used by Render/monitoring) |
| POST | `/predict` | Accepts patient JSON, returns prediction JSON |

### Example request
```bash
curl -X POST https://<your-app-name>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
      }'
```

### Example response
```json
{
  "prediction": "Heart Disease Detected",
  "confidence": 0.6865
}
```

Run locally:
```bash
pip install -r requirements.txt
python app.py
# App runs on http://127.0.0.1:5000
```

## Task 4 — GitHub and Cloud Deployment

### Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Heart Disease Prediction ML deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/HeartDiseaseDeployment.git
git push -u origin main
```

### Deploy on Render
1. Sign in to [Render](https://render.com) and click **New → Web Service**.
2. Connect your GitHub repository (`HeartDiseaseDeployment`).
3. Configure:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Click **Create Web Service**. Render will build and deploy automatically.
5. Once live, copy the public URL (e.g. `https://heartdiseasedeployment.onrender.com`)
   and paste it at the top of this README, replacing the placeholder.
6. Verify the deployment is reachable and `/predict` returns predictions
   (Render free-tier services sleep after inactivity — the first request
   after idling may take ~30–50 seconds to wake up).

## Task 5 — Conclusion

The Random Forest classifier achieved **83.6% accuracy** on the held-out
test set, with strong recall (0.97) for identifying patients who do have
heart disease — an encouraging result for a screening-style tool, though
in a real clinical setting further tuning and a larger, more diverse
dataset would be needed before deployment. The main challenges during
deployment were ensuring the Flask API validated and ordered incoming
JSON fields exactly as the model was trained on, keeping the model
artifact and dependency versions in sync between the local environment
and Render's build environment, and configuring Render's free-tier
service so it stays reachable for evaluation despite periodic sleep on
inactivity. This project highlighted why MLOps practices matter: version
control, reproducible training scripts, and automated cloud deployment
turn a one-off notebook model into a reliable, testable service that
others can actually call. Without these practices, model handoffs
between training and production become fragile, error-prone, and hard to
reproduce — exactly what MLOps pipelines are designed to prevent.

---

## 🛠 Tech Stack
Python · Pandas · Scikit-learn · Joblib · Flask · Gunicorn · GitHub · Render
