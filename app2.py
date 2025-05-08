import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Loading data
url = 'https://drive.google.com/file/d/13Fb0d_zRK5oP-3g1MKS1XAxkClsPFsUH/view?usp=drive_link'
df = pd.read_csv(url)
#df = pd.read_csv('Transformed_dataset.csv')  

# Setup
st.set_page_config(page_title="Air Pollution Project", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Data Overview", "Exploratory Data Analysis", "Model Building"])

# Welcome Page
if page == "Welcome":
    st.title("Air Pollution Prediction Assessment")
    st.write("""
    Welcome to the Air Quality Analysis and Prediction App!

    Use the sidebar to explore:
    - Data Overview 
    - Exploratory Data Analysis 
    - Model Building and Prediction 
    """)

# Data Overview Page
elif page == "Data Overview":
    st.title("Data Overview ")
    st.write("### First Look at the Data")
    st.dataframe(df)

    st.write("### Basic Information")
    st.write(df.describe())

    st.write("### Checking any missing Values")
    st.write(df.isnull().sum())

# EDA Page
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis ")

    # Shape & Size
    st.header("Shape and Size of Data")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    # Univariate Analysis
    st.header("Univariate Analysis:")

    df = df.drop(columns=['No'])  # Drop target and non-numeric columns like 'Time'

    col = st.selectbox("Select column for Univariate Analysis", df.select_dtypes(include=['float64', 'int64']).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    # Bivariate Analysis
    st.header("Bivariate Analysis:")

    col1 = st.selectbox("Select X-axis", df.select_dtypes(include=['float64', 'int64']).columns)
    col2 = st.selectbox("Select Y-axis", df.select_dtypes(include=['float64', 'int64']).columns)
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
    st.pyplot(fig)

    # Multivariate Analysis
    st.header("Multivariate Analysis:")
    st.write("Correlation Heatmap between numerical features")

    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Time-Series Analysis
    st.header("Time-Series Analysis 🕓")
    if 'datetime' in df.columns:
        time_col = st.selectbox("Select Time Column", [col for col in df.columns if 'datetime' in col.lower()])
        value_col = st.selectbox("Select Value Column", df.select_dtypes(include=['float64', 'int64']).columns)
        fig, ax = plt.subplots()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df_sorted = df.sort_values(by=time_col)
        ax.plot(df_sorted[time_col], df_sorted[value_col])
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No time-related column found in the dataset.")

# Model Building Page
elif page == "Model Building":
    st.title("Model Building and Evaluation 🚀")

    # Splitting Data
    X = df.drop(columns=['PM2.5'])
    X = X.select_dtypes(include=['float64', 'int64'])
    y = df['PM2.5']

    st.header("1. Splitting Data into Train and Test Sets")
    #features = df.drop('PM2.5', axis=1)
    target = df['PM2.5']

    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

    st.success("Data Split Successful!")

    # Distribution Plot
    st.header("2. PM2.5 Distribution Plot (Train vs Test)")
    fig, ax = plt.subplots()
    sns.kdeplot(y_train, label="Train", shade=True)
    sns.kdeplot(y_test, label="Test", shade=True)
    plt.legend()
    st.pyplot(fig)

    # Linear Regression
    st.header("3. Linear Regression Model")
    X = df.drop(columns=['datetime'])  # Drop target and non-numeric columns like 'Time'

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lr = lin_reg.predict(X_test)


    st.subheader("Linear Regression Metrics:")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred_lr):.2f}")

    # KNN Regression
    st.header("4. KNN Regression Model")

    X = df.drop(columns=['datetime'])  # Drop target and non-numeric columns like 'Time'

    k = st.slider("Select K Value", 1, 20, 5)
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    st.subheader("KNN Regression Metrics:")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred_knn):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_knn)):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred_knn):.2f}")
