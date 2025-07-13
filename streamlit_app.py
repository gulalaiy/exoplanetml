import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor # Already imported
from sklearn.gaussian_process.kernels import RBF, ConstantKernel # New import for GPR kernel
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile

# Page title
st.set_page_config(page_title='ExoplanetML', page_icon=':alien:')
st.title(':alien: ExoplanetML: Machine Learning Model for Target Variable Prediction')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to build a machine learning (ML) model for Exoplanet target variable prediction in an end-to-end workflow. This encompasses data upload, data pre-processing, ML model building and post-model analysis. It now supports both **Random Forest Regression** and **Gaussian Process Regression** models.') # Added GPR mention
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
    Here's a useful tool for data curation [CSV only]: <a href="https://aivigoratemitotool.streamlit.app/" target="_blank">AI-powered Data Curation Tool</a>. Tip: Ensure that your CSV file doesn't have any NaNs.
    </div>
    <br>
    """, unsafe_allow_html=True)

    st.markdown('**How to use the app?**')
    st.warning('To work with the app, go to the sidebar and select a dataset. Choose your ML model, adjust its parameters, which will initiate the ML model building process, display the model results, and allow users to download the generated models and accompanying data.') # Updated instructions

# Sidebar for input
with st.sidebar:
    # Load data
    st.header('1. Input data')

    st.markdown('**1.1 Use custom data**')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)

    # Download example data
    @st.cache_data
    def convert_df(input_df):
        return input_df.to_csv(index=False).encode('utf-8')

    example_csv = pd.read_csv('https://drive.google.com/uc?export=download&id=1J1f_qSHCYdfqiqQtpoda_Km_VmQuI4UX')
    csv = convert_df(example_csv)
    st.download_button(
        label="Download example CSV",
        data=csv,
        file_name='hwc-pesi.csv',
        mime='text/csv',
    )

    # Select example data
    st.markdown('**1.2. Use example data**')
    example_data = st.toggle('PHL Habitable Worlds Catalog (HWC)')
    if example_data:
        df = pd.read_csv('https://drive.google.com/uc?export=download&id=1J1f_qSHCYdfqiqQtpoda_Km_VmQuI4UX')

    # --- New: Model Selection ---
    st.header('2. Choose ML Model')
    model_choice = st.selectbox("Select ML Model", ("Random Forest Regressor", "Gaussian Process Regressor"))
    # --- End New: Model Selection ---

    st.header('3. Set Parameters') # Changed header to 3
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    # --- New: Conditional Learning Parameters ---
    if model_choice == "Random Forest Regressor":
        st.subheader('3.1. Random Forest Learning Parameters')
        with st.expander('See parameters'):
            parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
            parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])
            parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
            parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

        st.subheader('3.2. Random Forest General Parameters')
        with st.expander('See parameters', expanded=False):
            parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
            parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse'])
            parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
            parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

    elif model_choice == "Gaussian Process Regressor":
        st.subheader('3.1. Gaussian Process Learning Parameters')
        with st.expander('See parameters'):
            # Basic GPR parameters
            parameter_alpha = st.slider('Noise level (alpha)', 1e-10, 1e-1, 1e-10, format="%.10f") # Alpha for noise
            parameter_n_restarts_optimizer = st.slider('Number of restarts for optimizer (n_restarts_optimizer)', 0, 10, 0, 1)
            # For simplicity, we'll use a fixed kernel for now. Customizing kernels is more complex.
            st.markdown("*(Using default RBF kernel)*")

        st.subheader('3.2. Gaussian Process General Parameters')
        with st.expander('See parameters', expanded=False):
            # GPR also has a random_state
            parameter_random_state_gpr = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
    # --- End New: Conditional Learning Parameters ---

    sleep_time = st.slider('Sleep time', 0, 3, 0)


# Model building process
if uploaded_file is not None or example_data: # Use is not None instead of simply 'or example_data' for robustness
    with st.status("Running ...", expanded=True) as status:
    
        st.write("Loading data ...")
        time.sleep(sleep_time)

        st.write("Preparing data ...")
        time.sleep(sleep_time)
        # --- Start Existing Modification for X and Y selection ---
        # The original lines (commented out) assumed the last column was the target.
        # X = df.iloc[:,:-1] # g - all columns except for last on for features
        # y = df.iloc[:,-1] # g - last variable for the target

        # g - If 'P_ESI' is the ESI column name and it's not the last column
        target_column = 'P_ESI' # Or whatever the ESI column name is in your CSV. MAKE SURE THIS IS CORRECT FOR YOUR DATASET
        
        # Check if the target_column exists in the DataFrame
        if target_column not in df.columns:
            st.error(f"Error: Target column '{target_column}' not found in the uploaded data. Please check your CSV file or the 'target_column' variable in the code.")
            st.stop() # Stop execution if target column is missing

        y = df[target_column]
        
        # Drop the target column and any other irrelevant columns to get features (X)
        # Add any columns you've identified as irrelevant (e.g., IDs, strings) to this list
        irrelevant_columns = ['P_ESI'] # Start with the target column
        
        # You might need to add more irrelevant columns here based on your data analysis, e.g.:
        # irrelevant_columns.extend(['name', 'star_name', 'last_update', '__row', 'other_string_identifiers_etc'])

        # Filter X to only include numeric columns after dropping irrelevant ones, as ML models often require numeric input
        X = df.drop(columns=[col for col in irrelevant_columns if col in df.columns])
        X = X.select_dtypes(include=np.number) # Select only numeric columns for X

        if X.empty:
            st.error("Error: No numeric feature columns found after preprocessing. Please check your data and column selection.")
            st.stop() # Stop execution if no features are left
        # --- End Existing Modification for X and Y selection ---

        st.write("Splitting data ...")
        time.sleep(sleep_time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state if model_choice == "Random Forest Regressor" else parameter_random_state_gpr) # Use correct random_state based on model
    
        st.write(f"Training {model_choice} model ...") # Dynamic message
        time.sleep(sleep_time)

        # --- New: Conditional Model Training ---
        trained_model = None # Initialize model variable
        if model_choice == "Random Forest Regressor":
            if parameter_max_features == 'all':
                parameter_max_features = None
            
            trained_model = RandomForestRegressor(
                    n_estimators=parameter_n_estimators,
                    max_features=parameter_max_features,
                    min_samples_split=parameter_min_samples_split,
                    min_samples_leaf=parameter_min_samples_leaf,
                    random_state=parameter_random_state,
                    criterion=parameter_criterion,
                    bootstrap=parameter_bootstrap,
                    oob_score=parameter_oob_score)
            model_filename = 'rf_model.joblib' # Define filename for RF

        elif model_choice == "Gaussian Process Regressor":
            kernel = ConstantKernel(1.0) * RBF(1.0) # A common default kernel
            trained_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=parameter_alpha,
                n_restarts_optimizer=parameter_n_restarts_optimizer,
                random_state=parameter_random_state_gpr # Use GPR specific random_state
            )
            model_filename = 'gpr_model.joblib' # Define filename for GPR

        if trained_model: # Ensure a model was selected and instantiated
            trained_model.fit(X_train, y_train)
        else:
            st.error("No model selected or initialized. Please choose a model.")
            st.stop()
        # --- End New: Conditional Model Training ---
        
        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)
        y_train_pred = trained_model.predict(X_train)
        y_test_pred = trained_model.predict(X_test)
            
        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # --- Update results DataFrame to be dynamic ---
        model_results_df = pd.DataFrame({
            'Method': [model_choice], # Dynamic method name
            'Training MSE': [train_mse],
            'Training R2': [train_r2],
            'Test MSE': [test_mse],
            'Test R2': [test_r2]
        }).round(3)
        # --- End Update ---
        
    status.update(label="Status", state="complete", expanded=False)

    # Display data info
    st.header('Input data', divider='rainbow')
    col = st.columns(4)
    col[0].metric(label="No. of samples", value=X.shape[0], delta="")
    col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
    col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
    col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")
    
    with st.expander('Initial dataset', expanded=True):
        st.dataframe(df, height=210, use_container_width=True)
    with st.expander('Train split', expanded=False):
        train_col = st.columns((3,1))
        with train_col[0]:
            st.markdown('**X**')
            st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
        with train_col[1]:
            st.markdown('**y**')
            st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
    with st.expander('Test split', expanded=False):
        test_col = st.columns((3,1))
        with test_col[0]:
            st.markdown('**X**')
            st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
        with test_col[1]:
            st.markdown('**y**')
            st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

    # --- New: Conditional Feature Importance Display ---
    performance_col = st.columns((2, 0.2, 3))
    with performance_col[0]:
        st.header('Model performance', divider='rainbow')
        st.dataframe(model_results_df.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'})) # Dynamic results df

    with performance_col[2]:
        st.header('Feature importance', divider='rainbow')
        if model_choice == "Random Forest Regressor": # Only show for Random Forest
            importances = trained_model.feature_importances_
            feature_names = list(X.columns)
            forest_importances = pd.Series(importances, index=feature_names)
            df_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'value'})
            
            bars = alt.Chart(df_importance).mark_bar(size=40).encode(
                     x='value:Q',
                     y=alt.Y('feature:N', sort='-x')
                   ).properties(height=250)
            st.altair_chart(bars, theme='streamlit', use_container_width=True)
        else:
            st.info("Feature importance plot is typically not applicable or directly available for Gaussian Process Regressors in the same way as tree-based models like Random Forest.")
    # --- End New: Conditional Feature Importance Display ---

    # Prediction results
    st.header(f'Prediction results for {model_choice}', divider='rainbow') # Dynamic header
    s_y_train = pd.Series(y_train, name='actual').reset_index(drop=True)
    s_y_train_pred = pd.Series(y_train_pred, name='predicted').reset_index(drop=True)
    df_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
    df_train['class'] = 'train'
        
    s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
    s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
    df_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
    df_test['class'] = 'test'
    
    df_prediction = pd.concat([df_train, df_test], axis=0)
    
    prediction_col = st.columns((2, 0.2, 3))
    
    # Display dataframe
    with prediction_col[0]:
        st.dataframe(df_prediction, height=320, use_container_width=True)

    # Display scatter plot of actual vs predicted values
    with prediction_col[2]:
        scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
                        x='actual',
                        y='predicted',
                        color='class'
                  )
        st.altair_chart(scatter, theme='streamlit', use_container_width=True)

    # Save trained model
    # model_filename is already defined in the conditional training block
    joblib.dump(trained_model, model_filename) # Use trained_model variable

    with open(model_filename, 'rb') as f:
        st.download_button(
            label=f'Download Trained {model_choice} Model', # Dynamic label
            data=f,
            file_name=model_filename,
            mime='application/octet-stream'
        )

    # Apply to new dataset
    st.header('Apply Trained Model to New Dataset')
    new_file = st.file_uploader("Upload a new CSV for prediction", type=["csv"], key='predict')
    
    if new_file is not None:
        new_data = pd.read_csv(new_file)

        # Load the trained model
        with open(model_filename, 'rb') as f:
            saved_model = joblib.load(f)

        # Get the feature names that the model was trained on. This is safer.
        # Ensure 'model_features' is correctly retrieved from the saved model if possible.
        # If the model doesn't store feature_names_in_ directly, you might need to store them separately.
        # For this current setup, we assume feature_names_in_ works for both or the order is consistent.
        # A more robust way might be to save X.columns.tolist() alongside the model.
        
        # Check if the saved_model has feature_names_in_ attribute
        if hasattr(saved_model, 'feature_names_in_') and saved_model.feature_names_in_ is not None:
             model_features = saved_model.feature_names_in_
        else:
             # Fallback if feature_names_in_ is not available (e.g., older scikit-learn or certain models)
             # You might need to manually ensure the new_data has the same columns as X_train
             st.warning("Could not retrieve feature names from the saved model. Ensure your prediction data columns match training data.")
             model_features = X.columns # Fallback to columns from the *current* training data (risky if not consistent)


        # Check if the new dataset has the required input features
        missing_features = set(model_features).difference(new_data.columns)
        
        if len(missing_features) == 0:
            # Reorder the columns in the new dataset to match the model's expected input features
            new_X = new_data[model_features]
            
            # Predict using the loaded model
            predictions = saved_model.predict(new_X)
            
            # Add the predictions as a new column to the new dataset
            new_data['Predictions'] = predictions
            st.write(new_data.head())
            
            # Allow download of the new dataset with predictions
            csv_pred = convert_df(new_data)
            st.download_button(
                label="Download Predictions",
                data=csv_pred,
                file_name='predictions.csv',
                mime='text/csv'
            )
        else:
            # Show an error if the dataset is missing any required features
            st.error("The dataset is missing the following features: " + ", ".join(missing_features) + ". Please ensure the prediction CSV has the same columns used for training.")
else:
    st.warning('👈 Upload a CSV file or click *"Load example data"* to get started!')