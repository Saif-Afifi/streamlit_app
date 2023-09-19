import streamlit as st
import pandas as pd

# Function to automate data preprocessing
def automate_preprocessing(data):
    # Automatically detect column types
    categorical_cols = data.select_dtypes(include=['object']).columns
    numerical_cols = data.select_dtypes(exclude=['object']).columns

    # Automatically detect null values
    null_values = data.isnull().sum()

    return categorical_cols, numerical_cols, null_values

# Function to handle missing values and data transformation
def handle_missing_and_transform(data, categorical_cols, numerical_cols):
    st.subheader("Handling Missing Values")
    for col in data.columns:
        if data[col].isna().any():
            st.subheader(f"Handling missing values for column '{col}'")
            if col in categorical_cols:
                technique = st.radio(f"What do you want to do with '{col}'?", ('Most Frequent', 'Additional Class'))
                if technique == 'Most Frequent':
                    most_frequent = data[col].mode()[0]
                    data[col].fillna(most_frequent, inplace=True)
                elif technique == 'Additional Class':
                    data[col].fillna('Missing', inplace=True)
            elif col in numerical_cols:
                technique = st.radio(f"What do you want to do with '{col}'?", ('Mean', 'Median', 'Mode'))
                if technique == 'Mean':
                    mean_value = data[col].mean()
                    data[col].fillna(mean_value, inplace=True)
                elif technique == 'Median':
                    median_value = data[col].median()
                    data[col].fillna(median_value, inplace=True)
                elif technique == 'Mode':
                    mode_value = data[col].mode()[0]
                    data[col].fillna(mode_value, inplace=True)
    return data

# Streamlit app
st.title("Data Preprocessing and Model Training App")

# Upload data
st.subheader("Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader('Your Data')
    st.write(data.head())

    categorical_cols, numerical_cols, null_values = automate_preprocessing(data)

    st.subheader("Data Info")
    st.write(f"Number of Rows: {data.shape[0]}")
    st.write(f"Number of Columns: {data.shape[1]}")
    st.write("Column Types:")
    column_types = data.dtypes
    st.write(column_types)
    st.write("Missing Values:")
    st.write(null_values)
    st.subheader('Data Description')
    st.write(data.describe())

    st.subheader("Choose Columns to Drop")
    columns_to_drop = st.multiselect("Select columns to drop", data.columns)
    if columns_to_drop:
        data.drop(columns=columns_to_drop, inplace=True)

    st.subheader("Column Selection")
    target_col = st.selectbox("Select the Target Column", data.columns)
    st.write(f"Target Column: {target_col}")

    st.subheader("Data Transformation")
    data = handle_missing_and_transform(data, categorical_cols, numerical_cols)



    st.subheader("Model Training and Evaluation")

    # Detect task type (classification or regression)
    if st.button('Run Modelling'): 
        if data[target_col].dtype == 'object' or len(data[target_col].unique()) <= 2:
            task_type = 'Classification'
            st.write(f'This task is a {task_type} task')
            from pycaret.classification import *
            model = setup(data, target=target_col, train_size=0.7)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')

        else:
            task_type = 'Regression'
            st.write(f'This task is a {task_type} task')
            from pycaret.regression import *
            model = setup(data, target=target_col, train_size=0.7)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')
