import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Database setup
conn = sqlite3.connect('crop_predictions.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    area_harvested REAL,
    yield REAL,
    year INTEGER,
    model TEXT,
    predicted_production REAL
)
''')
conn.commit()

# Load data
file_path = 'FAOSTAT_data - FAOSTAT_data_en_12-29-2024.csv'
data = pd.read_csv(file_path)

st.title('🌾 Crop Production Prediction App')

st.write('### Initial Data Overview:')
st.write(data.head())

# Transform data
data_transformed = data.pivot_table(index=['Area', 'Year', 'Item'],
                                    columns='Element',
                                    values='Value').reset_index()

# Fill missing values
data_transformed[['Production', 'Area harvested', 'Yield']] = data_transformed[['Production', 'Area harvested', 'Yield']].fillna(0)

st.write('### Transformed Data:')
st.write(data_transformed.head())

st.write('### Missing Values:')
st.write(data_transformed.isnull().sum())

st.write('✅ Shape after handling missing values:', data_transformed.shape)

# 📊 EDA
st.subheader('📌 Area-wise Crop Production')
area_prod = data_transformed.groupby('Area')['Production'].sum().sort_values(ascending=False)
st.bar_chart(area_prod)

st.subheader('📌 Crop Type Distribution (Avg. Yield)')
crop_yield = data_transformed.groupby('Item')['Yield'].mean().sort_values(ascending=False)
st.bar_chart(crop_yield)

st.subheader('📌 Yearly Production Trends')
yearly_trend = data_transformed.groupby('Year')['Production'].sum()
st.line_chart(yearly_trend)

st.subheader('📌 Feature Correlation Heatmap')
corr = data_transformed[['Area harvested', 'Yield', 'Production']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='Blues', ax=ax)
st.pyplot(fig)

# ML model training
X = data_transformed[['Area harvested', 'Yield', 'Year']]
y = data_transformed['Production']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Model performance
st.subheader('📊 Model Performance')

st.write('**Linear Regression**')
st.write('R2 Score:', round(r2_score(y_test, y_pred_lr), 2))
st.write('Mean Absolute Error:', round(mean_absolute_error(y_test, y_pred_lr), 2))
st.write('Mean Squared Error:', round(mean_squared_error(y_test, y_pred_lr), 2))

st.write('**Random Forest Regressor**')
st.write('R2 Score:', round(r2_score(y_test, y_pred_rf), 2))
st.write('Mean Absolute Error:', round(mean_absolute_error(y_test, y_pred_rf), 2))
st.write('Mean Squared Error:', round(mean_squared_error(y_test, y_pred_rf), 2))

# User input and prediction
st.subheader("🌿 Predict Crop Production")

area_harvested_input = st.number_input('Enter Area Harvested (in hectares)', min_value=0.0, format="%.2f")
yield_input = st.number_input('Enter Yield (hg/ha)', min_value=0.0, format="%.2f")
year_input = st.number_input('Enter Year', min_value=1900, max_value=2100, step=1)
model_choice = st.selectbox('Choose a Model for Prediction', ['Linear Regression', 'Random Forest'])

if st.button('Predict Production'):
    input_data = np.array([[area_harvested_input, yield_input, year_input]])

    if model_choice == 'Linear Regression':
        predicted = lr.predict(input_data)[0]
    else:
        predicted = rf.predict(input_data)[0]

    st.success(f'✅ Predicted Production: {round(predicted, 2)} tonnes')

    cursor.execute('''
        INSERT INTO predictions (area_harvested, yield, year, model, predicted_production)
        VALUES (?, ?, ?, ?, ?)
    ''', (area_harvested_input, yield_input, year_input, model_choice, predicted))
    conn.commit()

    st.info("📥 Prediction saved to database!")
