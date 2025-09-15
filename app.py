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
            st.markdown("""
            This scatter plot visualizes the data points based on the two selected features. **Red data points** indicate the anomalies detected by the model. These are the transactions that are most different from the rest of the dataset, often due to their extreme values in one or more dimensions.
            """)
        else:
            st.info("Select at least two numerical features to generate a 2D plot.")
            
        # Add a simple bar chart of anomaly scores
        st.subheader("Anomaly Scores Distribution")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(df['anomaly_score'], kde=True, ax=ax2)
        ax2.set_title("Distribution of Anomaly Scores")
        ax2.axvline(x=0, color='red', linestyle='--', label='Anomaly Threshold')
        ax2.legend()
        st.pyplot(fig2)
        st.markdown("""
        This histogram shows the distribution of anomaly scores assigned by the model. Scores below zero (to the left of the **red dashed line**) are considered anomalies, while scores above zero are considered normal. The further a data point is to the left, the more likely it is to be a significant anomaly.
        """)

# --- CONCLUSION AND NEXT STEPS ---
st.header("4. Conclusion and Next Steps")
st.markdown("""
Based on the analysis, this tool has successfully identified transactions that deviate significantly from the norm. These outliers, whether due to an unusually high amount, a strange combination of features, or a rare transaction type, warrant further investigation.

### Ways to Fix and Probable Steps Going Forward:

1.  **Manual Review:** The most critical next step is to manually review each flagged anomaly. A human analyst should examine the details of these transactions to determine if they are legitimate, fraudulent, or simply data entry errors.
2.  **Root Cause Analysis:** For each confirmed anomaly, investigate the root cause. Was it a one-time event, or does it point to a systemic issue? For example, an anomalous expense could indicate a mis-categorized transaction or a new type of vendor relationship that needs to be documented.
3.  **Refine the Model:** Continuously improve the model by incorporating feedback. If the model flags too many false positives (legitimate transactions), adjust the `contamination` parameter or add more features to the analysis. If it misses clear anomalies, it might be time to collect more data or try a different algorithm.
4.  **Automate Alerts:** Integrate this anomaly detection logic into a real-time system. Instead of running a batch process, set up an automated alert system that notifies the relevant team (e.g., fraud, finance, or operations) as soon as a suspicious transaction occurs.
5.  **Expand Features:** For a more robust analysis, include additional features in the model, such as the time of day, day of the week, user ID, or location. These variables can help the model learn more complex patterns and improve its detection accuracy.
""")

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
