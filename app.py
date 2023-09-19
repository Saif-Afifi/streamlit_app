import streamlit as st
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


# Title and description
st.title("Auto Machine Learning Web App")
st.write("This is an automated app to train and evaluate machine learning models.")

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
            setup_df = pull()
            st.dataframe(setup_df)


        else:
            setup(data, target=target_column_name, session_id=123, silent=True, data_split_shuffle=False)
            setup_df = pull()
            st.dataframe(setup_df)


        if st.checkbox("Compare Models"):
            best_model =compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')


        # Allow user to choose models
        available_models = models()
        model_ids = available_models.index.tolist()
        model_names = available_models['Name'].tolist()
        selected_models_names = st.multiselect(
            "Select machine learning models to use:",
            model_names,
            default=model_names[:2]  # Default selection (first two models)
        )

        # Get corresponding model IDs for selected model names
        selected_models_ids = [model_id for model_id, model_name in zip(model_ids, model_names) if model_name in selected_models_names]

        # Check if any new models have been added
        new_models_ids = [model_id for model_id in selected_models_ids if model_id not in st.session_state.get('trained_models', [])]
        st.session_state.trained_models = selected_models_ids

        # Train and evaluate the selected models
        for model_id, model_name in zip(new_models_ids, selected_models_names):
            st.write(f"Training and evaluating {model_name}...")
            model = create_model(model_id)
            evaluate_model(model)
            create_df = pull()
            st.dataframe(create_df)

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
            for model_id, model_name in zip(selected_models_ids, selected_models_names):
                predictions = predict_model(model_id, data=new_data)
                st.write(f"Predictions using {model_name} (Model ID: {model_id}):")
                st.write(predictions)

    # Save Model
    st.sidebar.header("Save Model")
    if st.sidebar.button("Save Model"):
        final_model = create_model(selected_models_ids[0])  # Use the first selected model for saving
        save_model(final_model, 'final_model')
        st.write("Model saved as 'final_model.pkl'.")

# Footer
st.sidebar.write("By Saif Gamal")
