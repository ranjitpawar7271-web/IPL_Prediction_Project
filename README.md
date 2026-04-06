# 🏏 IPL Match Winner Prediction

A Machine Learning project to predict IPL match winners using historical data.
Includes full pipeline development, model deployment, and a web-based interface.

---

## 🚀 Features

* End-to-end ML pipeline (Data → Model → Deployment)
* Binary classification (Team1 vs Team2)
* REST API using FastAPI + BentoML
* Interactive frontend (HTML + JavaScript)
* Real-time prediction with confidence score

---

## 🧠 How It Works

* User selects match details (teams, venue, toss)
* Model predicts:

  * 🏆 Winner
  * 📊 Confidence %

---

## 📂 Dataset

* Sample dataset included (`IPL_small.csv`)
* Full dataset can be added at:

```
data/raw/IPL.csv
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/ranjitpawar7271-web/IPL_Prediction_Project.git
cd IPL_Prediction_Project

conda create -n ipl_env python=3.9 -y
conda activate ipl_env

pip install -r requirements.txt
python main.py

bentoml serve service:svc --port 3001
python -m http.server 5500
```

Open: http://localhost:5500

---

## 📊 Example Output

```
🏆 Mumbai Indians (74% confidence)
```

---
## 📸 Screenshot

![IPL Prediction UI](assets/ui_result.png)
## 👨‍💻 Author

**Ranjit Pawar**
[ranjitpawar7271@gmail.com](mailto:ranjitpawar7271@gmail.com)

---

⭐ Star the repo if you like it!
