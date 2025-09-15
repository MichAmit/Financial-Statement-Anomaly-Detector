import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set a professional and modern Streamlit UI
st.set_page_config(layout="wide", page_title="Financial Anomaly Detector", page_icon="📈")

# --- UI COMPONENTS ---
st.title("💰 Financial Statement Anomaly Detector")
st.markdown("""
    This app identifies **anomalies** in financial data using the **Isolation Forest** machine learning algorithm.
    Anomalies could indicate fraud, errors, or significant business events.
""")

# --- DATA UPLOAD AND PROCESSING ---
st.header("1. Upload Your Financial Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.write(df.head())
    
    st.header("2. Configure Anomaly Detection")
    # User selects columns for analysis
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    features_to_analyze = st.multiselect(
        "Select numerical features for analysis",
        options=numerical_cols,
        default=numerical_cols
    )

    if features_to_analyze:
        # Get user-defined contamination parameter
        contamination = st.slider(
            "Expected proportion of anomalies (Contamination)",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01
        )
        
        # --- MODEL TRAINING AND PREDICTION ---
        st.subheader("Running Anomaly Detection...")
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(df[features_to_analyze])
        
        # Predict anomalies. Isolation Forest returns -1 for anomalies and 1 for normal
        df['anomaly_score'] = model.decision_function(df[features_to_analyze])
        df['is_anomaly'] = model.predict(df[features_to_analyze])
        df['is_anomaly'] = df['is_anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
        
        # --- VISUALIZATION AND RESULTS ---
        st.header("3. Analysis Results")
        
        # Display the number of anomalies found
        anomalies_count = df[df['is_anomaly'] == 'Anomaly'].shape[0]
        st.write(f"**Total records analyzed:** {df.shape[0]}")
        st.write(f"**Anomalies detected:** {anomalies_count} ({anomalies_count/df.shape[0]:.2%})")

        # Display detected anomalies
        st.subheader("Detected Anomalies")
        anomalies_df = df[df['is_anomaly'] == 'Anomaly'].sort_values(by='anomaly_score')
        st.write(anomalies_df)
        
        # Visualize anomalies
        st.subheader("Interactive Anomaly Plot")
        if len(features_to_analyze) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x=features_to_analyze[0],
                y=features_to_analyze[1],
                hue='is_anomaly',
                style='is_anomaly',
                data=df,
                palette={'Normal': 'skyblue', 'Anomaly': 'red'},
                ax=ax
            )
            ax.set_title("2D Anomaly Plot")
            st.pyplot(fig)
        else:
            st.info("Select at least two numerical features to generate a 2D plot.")
            
        # Optional: Add a simple bar chart of anomaly scores
        st.subheader("Anomaly Scores Distribution")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(df['anomaly_score'], kde=True, ax=ax2)
        ax2.set_title("Distribution of Anomaly Scores")
        st.pyplot(fig2)

# --- INSTRUCTIONS FOR USE ---
st.sidebar.title("App Instructions")
st.sidebar.markdown("""
1.  **Upload** a `.csv` file with your financial data.
2.  **Select** the numerical features you want to analyze.
3.  **Adjust** the "Contamination" slider to set the expected percentage of anomalies.
4.  **Review** the analysis and visualizations to gain insights.
""")

st.sidebar.markdown("---")
st.sidebar.info("This is a simple demo. For real-world applications, you would need to use more complex feature engineering and a larger, labeled dataset for training.")