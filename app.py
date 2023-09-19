import streamlit as st
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report



# Title and description
st.title("Machine Learning Web App")
st.write("This app uses PyCaret to train and evaluate machine learning models.")

# Dictionary to keep track of trained models
trained_models = {}

# Sidebar for user input
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file:", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.subheader("Uploaded Data:")
    st.write(data.head())

    # Select target variable
    target_column_name = st.sidebar.selectbox("Select the target column:", data.columns)

    # Choose between classification and regression
    problem_type = st.sidebar.radio("Select the problem type:", ["Classification", "Regression"])

    # EDA
    st.sidebar.header("Exploratory Data Analysis")
    if st.sidebar.checkbox("Show EDA"):
        st.subheader("Exploratory Data Analysis")
        st.write(data.describe())
        profile_df = data.profile_report()
        st_profile_report(profile_df)


    # Model Training
    st.sidebar.header("Model Training")
    if st.sidebar.checkbox("Train Model"):
        st.subheader("Train Machine Learning Model")

        # Perform PyCaret setup
        if problem_type == "Classification":
            setup(data, target=target_column_name, session_id=123)
        else:
            setup(data, target=target_column_name, session_id=123, silent=True, data_split_shuffle=False)
        
        if st.checkbox("Compare Models"):
            compare_models()

        # Allow user to choose models
        available_models = models()
        model_ids = available_models['ID'].tolist()
        model_names = available_models['Name'].tolist()
        selected_models_names = st.multiselect(
            "Select machine learning models to use:",
            model_names,
            default=model_names[:2]  # Default selection (first two models)
        )

        # Get corresponding model IDs for selected model names
        selected_models_ids = [model_id for model_id, model_name in zip(model_ids, model_names) if model_name in selected_models_names]

        # Train and evaluate only the new models
        for model_id, model_name in zip(selected_models_ids, selected_models_names):
            if model_name not in trained_models:
                st.write(f"Training and evaluating {model_name}...")
                model = create_model(model_id)
                evaluate_model(model)
                trained_models[model_name] = model  # Add the trained model to the dictionary

    # Make Predictions
    st.sidebar.header("Make Predictions")
    if st.sidebar.checkbox("Make Predictions"):
        st.subheader("Make Predictions")
        # Upload a CSV file for predictions
        uploaded_file = st.file_uploader("Upload a CSV file for predictions:", type=["csv"])
        if uploaded_file is not None:
            new_data = pd.read_csv(uploaded_file)
            st.write("New data for predictions:")
            st.write(new_data.head())
            for model_name, model in trained_models.items():
                predictions = predict_model(model, data=new_data)
                st.write(f"Predictions using {model_name} (Model ID: {model_ids[model_names.index(model_name)]}):")
                st.write(predictions)

    # Save Model
    st.sidebar.header("Save Model")
    if st.sidebar.button("Save Model"):
        final_model = trained_models.get(selected_models_names[0])  # Use the first selected model for saving
        if final_model:
            save_model(final_model, 'final_model')
            st.write("Model saved as 'final_model.pkl'.")

# Footer
st.sidebar.write("By OpenAI's GPT-3")
