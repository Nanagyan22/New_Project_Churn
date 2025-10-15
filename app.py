import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Streamlit Page configuration
st.set_page_config(
    page_title="Reder Telecom Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Header
st.markdown("""
<div class="main-header">
    <h1>📊 REDER TELECOM</h1>
    <h3>Customer Churn Prediction System</h3>
    <p>Proactive Customer Retention Platform</p>
    <p style="font-size: 14px; margin-top: 10px;">Applied Machine Learning Project</p>
</div>
<style>
.main-header {
    text-align: center;
    padding: 20px 0;
    background: linear-gradient(90deg, #1f4e79, #2e6da4);
    color: white;
    margin: -30px -30px 30px -30px;
    border-radius: 0 0 10px 10px;
}
.stAlert > div {
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Will Load Model, Feature Names, Feature Types, and Preprocessing Pipeline
@st.cache_resource
def load_assets() -> Tuple[Any, List[str], Dict[str, str], Any]:
    """Load model, feature names, feature types, and preprocessing pipeline."""
    try:
        # Load model
        if not os.path.exists('model.pkl'):
            st.error("❌ Model file (model.pkl) not found.")
            st.info("💡 Run `python sample assets for testing.")
            st.stop()
        
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load feature names
        if not os.path.exists('feature_names.json'):
            st.error("❌ Feature names file (feature_names.json) not found.")
            st.stop()
        
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        # Load feature types
        if not os.path.exists('feature_types.json'):
            st.error("❌ Feature types file (feature_types.json) not found.")
            st.stop()
            
        with open('feature_types.json', 'r') as f:
            feature_types = json.load(f)
        
        # Load preprocessing pipeline
        preprocessing_pipeline = None
        if os.path.exists('preprocessing_pipeline.pkl'):
            try:
                with open('preprocessing_pipeline.pkl', 'rb') as f:
                    preprocessing_pipeline = pickle.load(f)
                st.sidebar.success(f"✅ Preprocessing pipeline loaded")
            except AttributeError as e:
                preprocessing_pipeline = None
        
        # Validate that feature_names is a list
        if not isinstance(feature_names, list):
            st.error("❌ Feature names should be a list of strings.")
            st.stop()
        
        # Validate that feature_types is a dictionary
        if not isinstance(feature_types, dict):
            st.error("❌ Feature types should be a dictionary mapping feature names to data types.")
            st.stop()
        
        st.sidebar.success(f"✅ Model loaded successfully")
        st.sidebar.info(f"📊 Features: {len(feature_names)}")
        
        return model, feature_names, feature_types, preprocessing_pipeline
        
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("💡 Make sure all required files (model.pkl, feature_names.json, feature_types.json) are loaded")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading assets: {e}")
        st.stop()

# Load assets
model, feature_names, feature_types, preprocessing_pipeline = load_assets()

# Sidebar: Model Status
st.sidebar.markdown("## 📊 Model Status")
st.sidebar.markdown("🌳 Using: **Logistic Regression**")

# Top 10 features from the model
top_features = [
    "TotalInteractionType_Email",
    "AVGLatePayment", 
    "customer_segment_premium",
    "customer_segment_need_attention",
    "TotalInteractionType_Call",
    "customer_segment_at_risk",
    "TotalInteractionType_Chat|Email",
    "NPS",
    "most_recent_action_date_Year",
    "End_Date_Year"
]

def prepare_input_data(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """Prepare input data for model prediction with proper data types and feature alignment."""
    try:
        # Create a DataFrame with all required features
        prepared_dict = {}
        
        # Initialize all features with default values
        for feature in feature_names:
            dtype = feature_types.get(feature, "float64")
            if "int" in dtype.lower():
                prepared_dict[feature] = 0
            elif "float" in dtype.lower():
                prepared_dict[feature] = 0.0
            else:
                prepared_dict[feature] = 0  # For categorical encoded as numeric
        
        # Update with provided values
        for key, value in input_dict.items():
            if key in prepared_dict:
                prepared_dict[key] = value
        
        # Create DataFrame
        input_df = pd.DataFrame([prepared_dict], columns=feature_names)
        
        # Apply proper data types
        for feature, dtype in feature_types.items():
            if feature in input_df.columns:
                try:
                    if "int" in dtype.lower():
                        input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce').fillna(0).astype(int)
                    elif "float" in dtype.lower():
                        input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce').fillna(0.0)
                    else:
                        input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce').fillna(0)
                except Exception as e:
                    st.warning(f"⚠️ Could not convert feature '{feature}' to {dtype}, using default value")
                    input_df[feature] = 0
        
        return input_df
    
    except Exception as e:
        st.error(f"❌ Error preparing input data: {e}")
        return None

def make_prediction(input_df: pd.DataFrame) -> Tuple[int, float, np.ndarray]:
    """Make prediction using the loaded model."""
    try:
        # Ensure we have the right number of features
        if len(input_df.columns) != len(feature_names):
            st.warning(f"⚠️ Feature count mismatch. Expected: {len(feature_names)}, Got: {len(input_df.columns)}")
        
        # Apply preprocessing pipeline if available
        if preprocessing_pipeline is not None:
            input_processed = preprocessing_pipeline.transform(input_df)
            # If the pipeline returns a numpy array, convert back to DataFrame
            if isinstance(input_processed, np.ndarray):
                input_processed = pd.DataFrame(input_processed, columns=feature_names)
        else:
            input_processed = input_df
        
        # Make predictions
        prediction = model.predict(input_processed)[0]
        prediction_proba = model.predict_proba(input_processed)[0]
        churn_probability = prediction_proba[1]
        
        return prediction, churn_probability, prediction_proba
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        import traceback
        st.error(f"**Traceback:** {traceback.format_exc()}")
        return None, None, None

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Introduction & Objectives",
    "🎯 Single Customer Prediction", 
    "📈 Batch Processing",
    "🧠 Model Details"
])

#  Tab 1: Introduction & Objectives
with tab1:
    st.header("Project Overview: Predicting Customer Churn at Reder Telecom")
    st.markdown("""
    - **Problem Statement:**
    Reder Telecom, a major player in the telecommunications industry, faces a significant challenge with customer churn. 
    This application leverages a machine learning model to identify customers who are most likely to leave, enabling targeted and effective retention strategies.
    """)

    st.subheader("Key Business Drivers")
    st.markdown("""
    - **Revenue Protection:** Safeguarding income by retaining existing customers is more cost-effective than acquiring new ones.
    - **Marketing Efficiency:** Optimize efforts by targeting customers with the highest churn risk for specific campaigns.
    - **Customer Satisfaction:** Proactively address issues to improve service quality and customer loyalty.
    """)

    st.subheader("Project Objectives")
    st.markdown("""
    1. **Develop a Classification Model:** To predict churn using historical customer data.
    2. **Analyze Key Factors:** Identify the top contributing factors that drive customer churn.
    3. **Create an Interactive Application:** Provide a user-friendly interface for business users to make data-driven decisions.
    """)

    st.subheader("How to Use This App")
    st.info("""
    1. **Single Customer Prediction:** Go to the 'Single Customer Prediction' tab, enter a customer's details, and get an instant churn risk assessment.
    2. **Batch Processing:** Navigate to the 'Batch Processing' tab to upload a CSV file of multiple customers and get a full analysis.
    3. **Model Details:** Explore the 'Model Details' tab to learn more about the model's capabilities and the features it uses.
    """)

# Tab 2: Single Customer Prediction & Assessment
with tab2:
    st.header("Single Customer Churn Assessment")

    with st.form("churn_assessment"):
        col1, col2, col3 = st.columns(3)
        
        # Personal Information
        with col1:
            st.subheader("📋 Personal Information")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            location = st.text_input("Location", value="")
            segment = st.selectbox("Segment", ["A", "B", "C"])
        
        # Service & Usage
        with col2:
            st.subheader("💼 Service & Usage")
            plan = st.selectbox("Plan", ["Basic", "Standard", "Premium"])
            nps = st.number_input("NPS (Net Promoter Score)*", min_value=-100, max_value=100, value=-50)
            total_purchase_value = st.number_input("Total Purchase Value", min_value=0.0, value=200.0)
            num_emails = st.number_input("Number of Emails*", min_value=0, value=25)
            num_calls = st.number_input("Number of Calls*", min_value=0, value=15)
            total_interactiontype_chat_email = st.number_input("Chat|Email Interactions*", min_value=0, value=18)
        
        # Churn Risk Factors
        with col3:
            st.subheader("📈 Churn Risk Factors")
            avg_late_payment = st.number_input("Average Late Payment*", min_value=0.0, value=10.0)
            customer_segment_premium = st.selectbox("Is Premium Segment?*", ["No", "Yes"])
            customer_segment_need_attention = st.selectbox("Needs Attention Segment?*", ["No", "Yes"], index=1)
            customer_segment_at_risk = st.selectbox("At Risk Segment?*", ["No", "Yes"], index=1)
            most_recent_action_date_year = st.number_input("Most Recent Action Year*", min_value=2000, max_value=2030, value=2020)
            end_date_year = st.number_input("End Date Year*", min_value=2000, max_value=2030, value=2021)

        submitted = st.form_submit_button("🎯 Assess Churn Risk", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing customer data..."):
            # Input dictionary with user inputs mapped to model features
            input_dict = {
                "Age": age,
                "NPS": nps,
                "TotalPurchaseValue": total_purchase_value,
                "numEmails": num_emails,
                "numCalls": num_calls,
                "AVGLatePayment": avg_late_payment,
                "TotalInteractionType_Call": num_calls,
                "TotalInteractionType_Chat|Email": total_interactiontype_chat_email,
                "TotalInteractionType_Email": num_emails,
                "customer_segment_at_risk": 1 if customer_segment_at_risk == "Yes" else 0,
                "customer_segment_need_attention": 1 if customer_segment_need_attention == "Yes" else 0,
                "customer_segment_premium": 1 if customer_segment_premium == "Yes" else 0,
                "most_recent_action_date_Year": most_recent_action_date_year,
                "End_Date_Year": end_date_year,
                "Gender_Male": 1 if gender == "Male" else 0,
                "Gender_Female": 1 if gender == "Female" else 0,
                "Segment": ord(segment) - ord('A'),
                "Plan": 0 if plan == "Basic" else (1 if plan == "Standard" else 2)
            }
            
            # Add additional context for display (used for prediction)
            customer_context = {
                "age": age,
                "gender": gender,
                "location": location,
                "segment": segment,
                "plan": plan,
                "total_purchase_value": total_purchase_value
            }
            
            # Prepare input data
            input_df = prepare_input_data(input_dict)
            
            if input_df is not None:
                # Make prediction
                prediction, churn_probability, prediction_proba = make_prediction(input_df)
                
                if prediction is not None:
                    # Display results
                    st.subheader("📊 Assessment Results")
                    
                    # Display customer context
                    with st.expander("👤 Customer Information", expanded=True):
                        info_col1, info_col2 = st.columns(2)
                        with info_col1:
                            st.write(f"**Age:** {customer_context['age']}")
                            st.write(f"**Gender:** {customer_context['gender']}")
                            st.write(f"**Location:** {customer_context['location'] if customer_context['location'] else 'N/A'}")
                        with info_col2:
                            st.write(f"**Segment:** {customer_context['segment']}")
                            st.write(f"**Plan:** {customer_context['plan']}")
                            st.write(f"**Total Purchase Value:** ${customer_context['total_purchase_value']:.2f}")
                    
                    st.markdown("---")
                    
                    # Determine churn status
                    if prediction == 1:
                        churn_status = "🚨 Will Churn"
                    else:
                        churn_status = "✅ Will Not Churn"
                    
                    # Recommendation based on probability
                    if churn_probability >= 0.7:
                        recommendation = "Immediate action needed to retain this customer"
                    elif churn_probability >= 0.5:
                        recommendation = "This customer needs attention soon"
                    elif churn_probability >= 0.3:
                        recommendation = "Keep a close watch on this customer"
                    else:
                        recommendation = "Customer appears stable"
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Churn Prediction", churn_status)
                    with col2:
                        st.metric("Churn Probability", f"{churn_probability:.1%}")
                    
                    st.info(f"**Recommendation:** {recommendation}")
                    
                    # Top Contributing Factors
                    st.subheader("🔍 What's Influencing This Prediction")
                    if hasattr(model, "coef_"):
                        try:
                            customer_values = input_df.iloc[0].values
                            coefs = model.coef_[0]
                            contributions = customer_values * coefs
                            
                            # Create feature contribution analysis
                            feature_contributions = list(zip(feature_names, customer_values, coefs, contributions))
                            # Sort by absolute contribution
                            feature_contributions.sort(key=lambda x: abs(x[3]), reverse=True)
                            
                            # Display top 5 factors
                            st.write("**Here are the main reasons for this prediction:**")
                            st.write("")
                            
                            # Map technical names to simple descriptions
                            feature_descriptions = {
                                "TotalInteractionType_Email": "number of email contacts",
                                "AVGLatePayment": "average late payments",
                                "customer_segment_premium": "premium customer status",
                                "customer_segment_need_attention": "needs attention status",
                                "TotalInteractionType_Call": "number of phone calls",
                                "customer_segment_at_risk": "at risk status",
                                "TotalInteractionType_Chat|Email": "chat and email contacts",
                                "NPS": "customer satisfaction score",
                                "most_recent_action_date_Year": "year of last activity",
                                "End_Date_Year": "contract end year"
                            }
                            
                            for i, (feat, val, coef, contrib) in enumerate(feature_contributions[:5], 1):
                                feature_desc = feature_descriptions.get(feat, feat.replace("_", " ").lower())
                                
                                if contrib > 0:
                                    impact = "increases the chance of leaving"
                                else:
                                    impact = "decreases the chance of leaving"
                                
                                st.write(f"{i}. The customer's **{feature_desc}** (value: {val:.1f}) {impact}.")
                            
                        except Exception as e:
                            st.error(f"❌ Could not analyze contributing factors: {e}")
                    else:
                        st.info("Contributing factor analysis is not available.")

#Tab 3: Batch Processing
with tab3:
    st.header("📈 Batch Churn Risk Assessment")
    st.markdown("**Upload a CSV file with customer data for batch processing.**")
    
    #template download 
    sample_row = {
        "TotalInteractionType_Email": 5,
        "AVGLatePayment": 1.2,
        "customer_segment_premium": 1,
        "customer_segment_need_attention": 0,
        "TotalInteractionType_Call": 3,
        "customer_segment_at_risk": 0,
        "TotalInteractionType_Chat|Email": 2,
        "NPS": 7,
        "most_recent_action_date_Year": 2024,
        "End_Date_Year": 2024
    }
    template_df = pd.DataFrame([sample_row])
    
    st.download_button(
        "📥 Download CSV Template",
        template_df.to_csv(index=False),
        "churn_template.csv",
        "text/csv",
        help="Download a template CSV file with sample data"
    )

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload your customer data CSV file")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} customer records.")
            
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), width='stretch')
            
            # Show data info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            
            # Check for missing required columns
            missing_cols = [col for col in top_features if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {missing_cols}")
                st.info("Missing columns will be filled with default values (0)")
            
            if st.button("🚀 Process Batch", width='stretch', type="primary"):
                with st.spinner("Processing batch... This may take a moment."):
                    try:
                        # Create a copy for processing
                        process_df = df.copy()
                        
                        # Prepare data for all rows
                        processed_data = []
                        for idx, row in process_df.iterrows():
                            # Create input dict from row
                            input_dict = {}
                            for feature in top_features:
                                if feature in row:
                                    input_dict[feature] = row[feature]
                                else:
                                    input_dict[feature] = 0
                            
                            # Add to processed data
                            processed_data.append(input_dict)
                        
                        # Convert to DataFrame and prepare for model
                        batch_df = pd.DataFrame(processed_data)
                        
                        # Add missing features with defaults
                        for feature in feature_names:
                            if feature not in batch_df.columns:
                                dtype = feature_types.get(feature, "float64")
                                if "int" in dtype.lower():
                                    batch_df[feature] = 0
                                elif "float" in dtype.lower():
                                    batch_df[feature] = 0.0
                                else:
                                    batch_df[feature] = 0
                        
                        # Ensure column order matches model expectations
                        batch_df = batch_df[feature_names]
                        
                        # Apply data types
                        for feature, dtype in feature_types.items():
                            if feature in batch_df.columns:
                                try:
                                    if "int" in dtype.lower():
                                        batch_df[feature] = pd.to_numeric(batch_df[feature], errors='coerce').fillna(0).astype(int)
                                    elif "float" in dtype.lower():
                                        batch_df[feature] = pd.to_numeric(batch_df[feature], errors='coerce').fillna(0.0)
                                    else:
                                        batch_df[feature] = pd.to_numeric(batch_df[feature], errors='coerce').fillna(0)
                                except:
                                    batch_df[feature] = 0
                        
                        # Make predictions
                        predictions = model.predict(batch_df)
                        probabilities = model.predict_proba(batch_df)[:, 1]
                        
                        # Add results to original dataframe
                        results_df = df.copy()
                        results_df['churn_prediction'] = predictions
                        results_df['churn_probability'] = probabilities
                        results_df['churn_status'] = ['Will Churn' if p == 1 else 'Will Not Churn' for p in predictions]
                        
                        # Assign recommendations
                        def get_recommendation(prob):
                            if prob >= 0.7:
                                return "Immediate action needed"
                            elif prob >= 0.5:
                                return "Needs attention soon"
                            elif prob >= 0.3:
                                return "Keep a close watch"
                            else:
                                return "Customer appears stable"
                        
                        results_df['recommendation'] = results_df['churn_probability'].apply(get_recommendation)
                        
                        # Display results summary
                        st.subheader("📈 Batch Processing Results")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Customers", len(results_df))
                        with col2:
                            churn_count = (results_df['churn_prediction'] == 1).sum()
                            st.metric("Predicted Churners", churn_count)
                        with col3:
                            churn_rate = (churn_count / len(results_df)) * 100
                            st.metric("Churn Rate", f"{churn_rate:.1f}%")
                        
                        # Show results table
                        st.subheader("📊 Detailed Results")
                        display_cols = ['churn_status', 'churn_probability', 'recommendation']
                        if len(results_df.columns) > 20:
                            st.info("Showing key results columns. Download full results to see all data.")
                            st.dataframe(results_df[display_cols + list(df.columns)[:5]], width='stretch')
                        else:
                            st.dataframe(results_df, width='stretch')
                        
                        # Download results
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Full Results as CSV",
                            csv_data,
                            "churn_batch_results.csv",
                            "text/csv",
                            help="Download complete results with predictions and risk analysis"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error processing batch: {str(e)}")
                        st.info("💡 Please check that your CSV format matches the expected structure.")
                        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("💡 Please ensure your file is a valid CSV format.")

# Tab 4: Model Details
with tab4:
    st.header("🧠 Model Details & Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Information")
        st.markdown(f"""
        **Model Type:** Logistic Regression
        **Total Features:** {len(feature_names)}
        **Model Status:** ✅ Loaded Successfully
        """)
        
        st.subheader("🎯 Key Capabilities")
        st.markdown("""
        - ✅ Binary classification (Churn/No Churn)
        - ✅ Probability estimates (0-100%)
        - ✅ Feature importance analysis
        - ✅ Handles numerical and categorical features
        - ✅ Batch processing support
        """)
    
    with col2:
        st.subheader("📊 Model Performance")
        st.info("""
        **Training Data:** Historical customer data from Reder Telecom
        
        **Validation Method:** Cross-validation and test set evaluation
        
        **Key Metrics Evaluated:**
        - Accuracy
        - Precision  
        - Recall
        - Confusion Matrix
        - ROC-AUC
        """)
    
    st.subheader("💼 Business Impact")
    st.markdown("""
    **Why Customer Churn Prediction Matters:**
    
    1. **💰 Revenue Protection** - Retaining customers is 5x more cost-effective than acquiring new ones
    2. **🎯 Targeted Marketing** - Focus retention efforts on high-risk customers
    3. **📈 Customer Lifetime Value** - Maximize long-term customer relationships
    4. **⚡ Proactive Action** - Address issues before customers decide to leave
    5. **🔍 Data-Driven Decisions** - Base retention strategies on predictive insights
    """)
    
    st.info("💡 **Tip:** Use the Single Customer Prediction tab for real-time assessments, or the Batch Processing tab for analyzing multiple customers at once.")
