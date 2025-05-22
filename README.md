# assignment_4
Predicting Crop Production Based on Agricultural Data
# 🌾 Predicting Crop Production Based on Agricultural Data

Hi there! 👋 I'm super excited to share my data science project that dives into predicting crop production using global agricultural data. This isn’t just another regression task — it's a full pipeline from raw CSVs to real-time predictions in a web app, all built from scratch.

---

## 🚜 Why I Built This

Agriculture is the backbone of many economies, especially in developing countries. But predicting how much crop will be produced in a given year is tricky and influenced by many factors like area harvested, yield, and climate. I wanted to explore how well we can use machine learning to make sense of this and offer future predictions that could actually help farmers, researchers, or policymakers.

---

## 🔍 The Data

I used open-source data from [FAOSTAT](https://www.fao.org/faostat/en/), which includes:

- **Country/Region**
- **Year**
- **Crop type**
- **Area harvested (ha)**
- **Yield (hg/ha)**
- **Production (tonnes)**

> File used: `FAOSTAT_data_en_12-29-2024.csv`

---

## 🧠 What I Did

Here's a quick walkthrough of what this project includes:

- 📦 Loaded and transformed raw data using `pandas`.
- 🧹 Cleaned missing values and reshaped the dataset using `pivot_table()`.
- 📊 Visualized data using `matplotlib`, `seaborn`, and built-in Streamlit charts.
- 🤖 Trained two machine learning models:
  - `Linear Regression`
  - `Random Forest Regressor`
- ⚙️ Evaluated model performance using R² Score, MAE, and MSE.
- 🎛 Built a real-time prediction UI using **Streamlit**.
- 💾 Logged predictions into a local **SQLite database** (`crop_predictions.db`).

---

## 🖥 How to Use It

### ✅ Run the App

```bash
streamlit run crop_prediction_app.py

Requirements:

pip install pandas numpy scikit-learn matplotlib seaborn streamlit
