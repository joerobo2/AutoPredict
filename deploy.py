import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, \
    roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt


# Load the vehicle maintenance dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


vehicle_data = load_data("vehicle_maintenance_data.csv")  # replace with your vehicle maintenance dataset path

# Embed the banner image
banner_path = "vehicle.png"  # Update to the correct path
st.markdown(
    f"""
    <style>
    .full-width-banner {{
        width: 100%;
        height: auto;
        margin-bottom: 20px;
    }}
    </style>
    <img class="full-width-banner" src="data:image/png;base64,{base64.b64encode(open(banner_path, "rb").read()).decode()}">
    """,
    unsafe_allow_html=True
)

st.title("AutoPredict")

# Introductory Summary (Executive Overview)
st.write("""
### Project Overview
This project aims to develop a machine learning model that predicts the maintenance needs of vehicles based on several features such as mileage, vehicle age, maintenance history, and fuel type. By using this model, we hope to assist fleet managers and vehicle owners in scheduling timely maintenance to prevent breakdowns and reduce operational costs.
""")

# Organize layout using Streamlit Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data Overview", "Preprocessing", "Analysis", "Prediction", "Evaluation"])

# Developer information
st.sidebar.markdown("### Explore the Tabs to Dive into Vehicle Maintenance Prediction")
st.sidebar.markdown("### Developed by Joseph Robinson")

# Data Overview Tab
with tab1:
    st.write("### Data Overview")
    st.write("""
    Understanding the data is key to building an effective predictive model. The dataset contains several vehicle-specific attributes such as **Vehicle_Model**, **Mileage**, **Vehicle_Age**, and **Maintenance_History**, which are vital for predicting maintenance needs.
    """)
    st.write(vehicle_data.info())
    st.write(vehicle_data.head())

    # Display summary statistics
    st.write("### Summary Statistics")
    st.write(vehicle_data.describe(include='all'))

    # Display data dictionary
    st.write("### Data Dictionary")
    st.write(""" 
    **Features**:
    - **Vehicle_Model**: Type of the vehicle (Car, SUV, Van, Truck, Bus, Motorcycle)
    - **Mileage**: Total mileage of the vehicle
    - **Maintenance_History**: Maintenance history of the vehicle (Good, Average, Poor)
    - **Reported_Issues**: Number of reported issues
    - **Vehicle_Age**: Age of the vehicle in years
    - **Fuel_Type**: Type of fuel used (Diesel, Petrol, Electric)
    - **Transmission_Type**: Transmission type (Automatic, Manual)
    - **Engine_Size**: Size of the engine in cc (Cubic Centimeters)
    - **Odometer_Reading**: Current odometer reading of the vehicle
    - **Last_Service_Date**: Date of the last service
    - **Warranty_Expiry_Date**: Date when the warranty expires
    - **Owner_Type**: Type of vehicle owner (First, Second, Third)
    - **Insurance_Premium**: Insurance premium amount
    - **Service_History**: Number of services done
    - **Accident_History**: Number of accidents the vehicle has been involved in
    - **Fuel_Efficiency**: Fuel efficiency of the vehicle in km/l (Kilometers per liter)
    - **Tire_Condition**: Condition of the tires (New, Good, Worn Out)
    - **Brake_Condition**: Condition of the brakes (New, Good, Worn Out)
    - **Battery_Status**: Status of the battery (New, Good, Weak)
    - **Need_Maintenance**: Target variable indicating whether the vehicle needs maintenance (1 = Yes, 0 = No)
    """)

# Preprocessing Tab
with tab2:
    st.write("### Data Preprocessing")
    st.write("""
    Preprocessing involves preparing the data for model training. Here, we handle missing values, encode categorical variables, and standardize the dataset to ensure consistency across different features.
    """)

    # Create a table for missing values
    missing_values = vehicle_data.isnull().sum().reset_index()
    missing_values.columns = ['Feature', 'Missing Values']
    missing_values = missing_values[missing_values['Missing Values'] > 0]

    st.write("### Missing Values")
    st.dataframe(missing_values)

    # Check for duplicates
    duplicate_rows = vehicle_data.duplicated().sum()

    # Create a uniform data quality table
    quality_metrics = pd.DataFrame({
        'Metric': ['Duplicate Rows', 'Total Rows'],
        'Count': [duplicate_rows, vehicle_data.shape[0]]
    })

    st.write("### Data Quality Metrics")
    st.dataframe(quality_metrics)


    # Preprocess the dataset
    def preprocess_data(df):
        # Label encoding for categorical features
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Dropping rows with any missing values
        initial_shape = df.shape
        df.dropna(inplace=True)
        final_shape = df.shape

        st.write(f"### Rows Dropped Due to Missing Values: {initial_shape[0] - final_shape[0]}")

        return df, label_encoders


    vehicle_data_processed, encoders = preprocess_data(vehicle_data)

    st.write("### Processed Data Overview")
    st.write(vehicle_data_processed.head())

# Analysis Tab
with tab3:
    st.write("### Univariate and Multivariate Analysis")
    st.write("""
    **Univariate Analysis**: Understanding the distribution of individual features helps us gain insights into the overall condition of the vehicle fleet and the factors contributing to the need for maintenance.

    **Multivariate Analysis**: Examining the relationships between features such as **Brake_Condition** and **Reported_Issues** gives us a deeper understanding of how different vehicle attributes impact maintenance needs.
    """)

    # Univariate analysis - distribution of numerical features
    for col in vehicle_data_processed.select_dtypes(include=['float64', 'int64']).columns:
        st.write(f"#### Distribution of {col}")
        fig = px.histogram(vehicle_data_processed, x=col, nbins=30)
        st.plotly_chart(fig)

    st.write("### Correlation Heatmap")
    st.write("""
    By analyzing the correlation between features, we can understand how various factors influence maintenance needs. For example, **Reported_Issues** is strongly correlated with **Need_Maintenance**, highlighting the importance of addressing vehicle problems early.
    """)

    corr = vehicle_data_processed.corr()
    fig_corr = px.imshow(corr, color_continuous_scale='Viridis')
    st.plotly_chart(fig_corr)

# Prediction Tab
with tab4:
    st.write("### Vehicle Maintenance Prediction")
    st.write("""
    In this section, we allow users to input vehicle attributes, and the machine learning model predicts whether the vehicle will need maintenance.

    **Models Used**: Logistic Regression, Random Forest, XGBoost
    """)

    # Split the dataset for prediction
    X = vehicle_data_processed.drop(columns=['Need_Maintenance'])
    y = vehicle_data_processed['Need_Maintenance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Sidebar for input features
    with st.sidebar.expander("Input Vehicle Data"):
        def user_input_vehicle_features():
            data = {}
            for column in X.columns:
                if column in vehicle_data_processed.select_dtypes(include=['object']).columns:
                    unique_values = vehicle_data_processed[column].unique()
                    data[column] = st.sidebar.selectbox(column, unique_values)
                else:
                    data[column] = st.sidebar.slider(column, float(vehicle_data_processed[column].min()),
                                                     float(vehicle_data_processed[column].max()),
                                                     float(vehicle_data_processed[column].mean()))
            input_vehicle_df = pd.DataFrame(data, index=[0])
            return input_vehicle_df


        input_vehicle_df = user_input_vehicle_features()

    # Model Selection for Vehicle Maintenance
    vehicle_model_choice = st.selectbox("Choose Vehicle Model", ("Logistic Regression", "Random Forest", "XGBoost"))


    def train_vehicle_model(model_name):
        """Train the selected vehicle model."""
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=500)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_name == "XGBoost":
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        model.fit(X_train, y_train)
        return model


    vehicle_model = train_vehicle_model(vehicle_model_choice)

    # Preprocess user input for vehicle prediction
    preprocessed_vehicle_input_df = scaler.transform(input_vehicle_df)

    # Make prediction based on user input
    vehicle_prediction = vehicle_model.predict(preprocessed_vehicle_input_df)
    vehicle_prediction_proba = vehicle_model.predict_proba(preprocessed_vehicle_input_df)[:, 1]

    st.write(f"### Predicted Need for Maintenance: {'Yes' if vehicle_prediction[0] == 1 else 'No'}")
    st.write(f"### Prediction Probability: {vehicle_prediction_proba[0]:.2f}")

    # Display Feature Importance (if applicable)
    st.write("### Feature Importance")
    if vehicle_model_choice == "Logistic Regression":
        st.write("**Logistic Regression doesn't provide feature importance.**")
    elif vehicle_model_choice == "Random Forest":
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': vehicle_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.dataframe(feature_importances)
        fig = px.bar(feature_importances, x='Importance', y='Feature', orientation='h',
                     title='Random Forest Feature Importance')
        st.plotly_chart(fig)
    elif vehicle_model_choice == "XGBoost":
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': vehicle_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.dataframe(feature_importances)
        fig = px.bar(feature_importances, x='Importance', y='Feature', orientation='h',
                     title='XGBoost Feature Importance')
        st.plotly_chart(fig)

# Evaluation Tab
with tab5:
    st.write("### Model Evaluation Metrics")
    st.write("""
    After training the models, it's crucial to evaluate their performance on unseen data. This helps ensure that the model generalizes well and isn't overfitting.
    """)

    if 'Need_Maintenance' in vehicle_data_processed.columns:
        y_pred = vehicle_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, vehicle_model.predict_proba(X_test)[:, 1])

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Display evaluation metrics
        metrics = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            'Score': [accuracy, precision, recall, f1, roc_auc]
        })

        st.dataframe(metrics)

        # Confusion matrix visualization
        fig_conf_matrix = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['No Need', 'Need'],
            y=['No Need', 'Need'],
            colorscale='Viridis',
            colorbar=dict(title='Count'),
        ))
        fig_conf_matrix.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
        st.plotly_chart(fig_conf_matrix)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, vehicle_model.predict_proba(X_test)[:, 1])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
        fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig_roc)

    st.write("""
    **Conclusion**:
    Both Random Forest and XGBoost models performed exceptionally well, with high accuracy and AUC scores. However, it's essential to validate these models on unseen real-world data to ensure they generalize well.
    """)

