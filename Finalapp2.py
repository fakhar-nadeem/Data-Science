import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the merged dataset
# Loading through google drive because csv was too big to upload on github
import requests

file_id = "13Fb0d_zRK5oP-3g1MKS1XAxkClsPFsUH"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
merged_df = pd.read_csv(url)

# App config
st.set_page_config(page_title="Air Pollution App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Data Overview", "Exploratory Data Analysis", "Model Building"])

# Welcome Page
if page == "Welcome":
    st.title("Air Pollution Prediction App")
    st.write("""
        Welcome to the Air Pollution Analysis and Prediction App!

        Use the sidebar to navigate:
        - View Data Overview
        - Perform Exploratory Data Analysis (EDA)
        - Build and evaluate predictive models
    """)

# Data Overview
elif page == "Data Overview":
    st.title("Data Overview")
    st.dataframe(merged_df)
    st.subheader("Basic Info")
    st.write(merged_df.describe())
    st.write("Missing Values:")
    st.write(merged_df.isnull().sum())

# Exploratory Data Analysis
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    st.subheader("Shape and Size")
    st.write(f"Rows: {merged_df.shape[0]}")
    st.write(f"Columns: {merged_df.shape[1]}")

    st.subheader("Univariate Analysis")
    numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    univariate_col = st.selectbox("Select Column", numeric_cols)
    fig1, ax1 = plt.subplots()
    sns.histplot(merged_df[univariate_col], kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Bivariate Analysis")
    x_col = st.selectbox("X-Axis", numeric_cols, key="biv_x")
    y_col = st.selectbox("Y-Axis", numeric_cols, key="biv_y")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=merged_df[x_col], y=merged_df[y_col], ax=ax2)
    st.pyplot(fig2)

    st.subheader("Multivariate Analysis")

    # Define the specific columns to include
    selected_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']
    
    # Filter numeric columns that are also in the selected list
    filtered_cols = [col for col in selected_cols if col in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[col])]
    
    # Compute correlation matrix
    corr = merged_df[filtered_cols].corr()
    
    # Plot full square heatmap
    fig3, ax3 = plt.subplots(figsize=(18, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax3, annot_kws={"size": 8})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    st.pyplot(fig3)



    st.subheader("Time-Series Analysis")
    if 'datetime' in merged_df.columns:
        merged_df['datetime'] = pd.to_datetime(merged_df['datetime'], errors='coerce')
        ts_col = st.selectbox("Select Value Column", numeric_cols, key="ts")
        yearly_df = merged_df.groupby(merged_df['datetime'].dt.year)[ts_col].mean().reset_index()
        fig4, ax4 = plt.subplots()
        sns.lineplot(x=yearly_df['datetime'], y=yearly_df[ts_col], ax=ax4)
        ax4.set_title(f"Yearly Trend of {ts_col}")
        st.pyplot(fig4)
    else:
        st.warning("No datetime column found.")

    st.subheader("Derived Feature Analysis: PM Ratio & SO Ratio")
    
    # Ensure datetime is parsed
    if 'datetime' in merged_df.columns:
        merged_df['datetime'] = pd.to_datetime(merged_df['datetime'], errors='coerce')
    
        # PM Ratio Calculation and Plot
        merged_df['PM_ratio'] = merged_df['PM2.5'] / merged_df['PM10']
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        ax5.plot(merged_df['datetime'], merged_df['PM_ratio'], label='PM Ratio', color='red')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('PM Ratio')
        ax5.set_title('PM2.5 / PM10 Ratio Over Time')
        ax5.legend()
        st.pyplot(fig5)
        st.caption("High PM2.5/PM10 ratios indicate dominance of fine particulate matter, which is more hazardous (Zhang and Cao, 2015).")
    
        # SO Ratio Calculation and Plot
        merged_df['SO_ratio'] = merged_df['SO2'] / merged_df['NO2']
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        ax6.plot(merged_df['datetime'], merged_df['SO_ratio'], label='SO Ratio', color='green')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('SO Ratio')
        ax6.set_title('SO2 / NO2 Ratio Over Time')
        ax6.legend()
        st.pyplot(fig6)
        st.caption("SO2/NO2 ratios can indicate pollution source types: vehicle vs industrial (Nirel, 2001).")
    
    else:
        st.warning("Datetime column is missing or not in proper format.")
    

# Model Building
elif page == "Model Building":
    st.title("Model Building")

    merged_df = merged_df.dropna()
    X = merged_df.select_dtypes(include=['float64', 'int64']).drop(columns=['PM2.5'])
    y = merged_df['PM2.5']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("PM2.5 Train vs Test Distribution")
    fig5, ax5 = plt.subplots()
    sns.kdeplot(y_train, label='Train', fill=True)
    sns.kdeplot(y_test, label='Test', fill=True)
    ax5.legend()
    st.pyplot(fig5)

    metrics = {}

    # Linear Regression
    st.header("Linear Regression Model:")
    X = merged_df.drop(columns=['datetime'])  # Drop target and non-numeric columns like 'Time'

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lr = lin_reg.predict(X_test)


    st.subheader("Linear Regression Metrics:")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred_lr):.2f}")

    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)
    metrics['Linear'] = [mean_absolute_error(y_test, y_pred_lin),
                         np.sqrt(mean_squared_error(y_test, y_pred_lin)),
                         r2_score(y_test, y_pred_lin)]

# KNN Regression
    st.header("KNN Regression Model:")

    X = merged_df.drop(columns=['datetime'])  # Drop target and non-numeric columns like 'Time'

    k = st.slider("Select K Value", 1, 20, 5)
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    st.subheader("KNN Regression Metrics:")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred_knn):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_knn)):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred_knn):.2f}")

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    metrics['KNN'] = [mean_absolute_error(y_test, y_pred_knn),
                      np.sqrt(mean_squared_error(y_test, y_pred_knn)),
                      r2_score(y_test, y_pred_knn)]

    # SVM
    st.header("Support Vector Regression (SVM)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr = SVR(kernel='rbf')
    svr.fit(X_train_scaled, y_train)
    y_pred_svr = svr.predict(X_test_scaled)

    st.subheader("SVM Metrics:")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred_svr):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_svr)):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred_svr):.2f}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svr = SVR(kernel='rbf')
    svr.fit(X_train_scaled, y_train)
    y_pred_svr = svr.predict(X_test_scaled)
    metrics['SVM'] = [mean_absolute_error(y_test, y_pred_svr),
                      np.sqrt(mean_squared_error(y_test, y_pred_svr)),
                      r2_score(y_test, y_pred_svr)]

    # Random Forest
    st.header("Random Forest Model")
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    st.subheader("Random Forest Metrics:")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred_rf):.2f}")

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    metrics['Random Forest'] = [mean_absolute_error(y_test, y_pred_rf),
                                np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                                r2_score(y_test, y_pred_rf)]


    # Comparison Graphs
    st.subheader("Model Comparison")
    model_names = list(metrics.keys())
    mae_vals = [v[0] for v in metrics.values()]
    rmse_vals = [v[1] for v in metrics.values()]
    r2_vals = [v[2] for v in metrics.values()]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].bar(model_names, mae_vals, color='skyblue')
    axs[0].set_title('MAE')
    axs[1].bar(model_names, rmse_vals, color='orange')
    axs[1].set_title('RMSE')
    axs[2].bar(model_names, r2_vals, color='green')
    axs[2].set_title('R² Score')
    st.pyplot(fig)
