import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
from typing import Dict, Any, Optional
import logging
import os
import sys
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    MBartForConditionalGeneration, MBart50TokenizerFast,
    BartForConditionalGeneration, BartTokenizer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================
# CONFIGURATION
# =============================================

# Model paths - updated to use relative paths
MODEL_PATHS = {
    "fraud_model": "env/Scripts/insurance_risk_claims_model.pkl",
    "sentiment_model": "env/Scripts/sentiment_analysis_model.pkl",
    "fraudulent_claims_model": "env/Scripts/fraudulent_claims_model.pkl",
    "customer_segmentation_model": "env/Scripts/customer_segmentation_model (1).pkl"
}

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Language code map for translation
LANG_CODES = {
    "French": "fr_XX",
    "German": "de_DE",
    "Spanish": "es_XX"
}

# =============================================
# MODEL LOADING UTILITIES
# =============================================

@st.cache_resource
def safe_load_model(path: str) -> Optional[Any]:
    """Enhanced model loader with multiple fallback strategies"""
    try:
        if not os.path.exists(path):
            logger.warning(f"Model file not found at {path}")
            return None

        # Try different loading methods
        for loader in [_try_pytorch_load, _try_sklearn_load, _try_generic_pickle]:
            try:
                model = loader(path)
                if model is not None:
                    logger.info(f"Successfully loaded {os.path.basename(path)} with {loader.__name__}")
                    return model
            except Exception as e:
                logger.debug(f"{loader.__name__} failed: {str(e)}")
                continue

        logger.warning(f"All loading methods failed for {path}")
        return None

    except Exception as e:
        logger.error(f"Failed to load {os.path.basename(path)}: {str(e)}")
        return None

def _try_pytorch_load(path: str) -> Optional[Any]:
    """Attempt PyTorch model loading"""
    try:
        return torch.load(path, map_location=torch.device('cpu'))
    except:
        return None

def _try_sklearn_load(path: str) -> Optional[Any]:
    """Attempt sklearn model loading"""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
            if hasattr(model, 'predict') or hasattr(model, 'transform'):
                return model
            return None
    except:
        return None

def _try_generic_pickle(path: str) -> Optional[Any]:
    """Generic pickle loading fallback"""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_models() -> Dict[str, Any]:
    """Load all models with progress tracking"""
    models = {}
    
    # Check if we're in development mode (no models present)
    dev_mode = all(not os.path.exists(path) for path in MODEL_PATHS.values())
    
    if dev_mode:
        st.warning("Running in development mode - using fallback rules for all models")
        return {name: None for name in MODEL_PATHS.keys()}
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (model_name, path) in enumerate(MODEL_PATHS.items()):
        progress = (i + 1) / len(MODEL_PATHS)
        progress_bar.progress(progress)
        status_text.text(f"Loading {model_name.replace('_', ' ')}...")

        try:
            models[model_name] = safe_load_model(path)
            if models[model_name] is None:
                logger.warning(f"Failed to load {model_name}")
        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            models[model_name] = None

    progress_bar.empty()
    status_text.empty()
    return models

# =============================================
# NLP MODELS LOADING
# =============================================

@st.cache_resource
def load_translation_model():
    """Load mBART model for translation"""
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading translation model: {str(e)}")
        return None, None

@st.cache_resource
def load_summarization_model():
    """Load BART model for summarization"""
    try:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading summarization model: {str(e)}")
        return None, None

# =============================================
# PREPROCESSING FUNCTIONS
# =============================================

def preprocess_risk_input(inputs: Dict) -> pd.DataFrame:
    """Prepare risk prediction inputs"""
    mapping = {
        'Gender': {'Male': 0, 'Female': 1},
        'Policy_Type': {'Basic': 0, 'Standard': 1, 'Premium': 2},
        'Claim_History': {'None': 0, '1-2 claims': 1, '3-5 claims': 2, '5+ claims': 3},
        'Fraudulent_Claim': {'No': 0, 'Yes': 1}
    }
    
    df = pd.DataFrame([inputs])
    for col, map_dict in mapping.items():
        if col in df.columns:
            df[col] = df[col].map(map_dict)
    return df

def preprocess_fraud_input(inputs: Dict) -> pd.DataFrame:
    """Prepare fraud detection inputs"""
    inputs['incident_severity'] = {"Minor": 0, "Moderate": 1, "Severe": 2}[inputs['incident_severity']]
    inputs['suspicious_docs'] = int(inputs['suspicious_docs'])
    return pd.DataFrame([inputs])

def preprocess_segmentation_input(inputs: Dict) -> pd.DataFrame:
    """Prepare customer segmentation inputs"""
    education_map = {"High School": 0, "College": 1, "Graduate": 2}
    inputs['education'] = education_map[inputs['education']]
    return pd.DataFrame([inputs])

# =============================================
# PREDICTION FUNCTIONS
# =============================================

def predict_risk(model, inputs: Dict) -> Optional[Dict]:
    """Run risk prediction"""
    try:
        df = preprocess_risk_input(inputs)
        
        if hasattr(model, 'predict_proba'):
            prediction = model.predict(df)[0]
            proba = model.predict_proba(df)[0]
            confidence = max(proba) * 100
        else:
            prediction = model.predict(df)[0]
            proba = [0, 0]
            confidence = 100
        
        return {
            'result': 'High Risk' if prediction == 1 else 'Low Risk',
            'confidence': confidence,
            'proba': proba
        }
    except Exception as e:
        logger.error(f"Risk prediction error: {str(e)}")
        return None

def detect_fraud(model, inputs: Dict) -> Optional[Dict]:
    """Run fraud detection"""
    try:
        df = preprocess_fraud_input(inputs)
        
        if hasattr(model, 'predict_proba'):
            prediction = model.predict(df)[0]
            proba = model.predict_proba(df)[0]
            confidence = max(proba) * 100
        else:
            prediction = model.predict(df)[0]
            proba = [0, 0]
            confidence = 100
        
        return {
            'result': 'Fraudulent' if prediction == 1 else 'Legitimate',
            'confidence': confidence,
            'proba': proba
        }
    except Exception as e:
        logger.error(f"Fraud detection error: {str(e)}")
        return None

def analyze_sentiment(model, text: str) -> Optional[Dict]:
    """Analyze text sentiment"""
    try:
        if model is None:
            return None
            
        if hasattr(model, 'predict'):
            # Handle case where model.predict returns array
            prediction = model.predict([text])[0]
            if isinstance(prediction, (np.ndarray, list)):
                prediction = prediction[0]  # Take first element if array
            return {
                'sentiment': "Positive" if prediction == 1 else "Negative",
                'confidence': 100  # Assume full confidence
            }
        return None
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        return None

def segment_customer(model, inputs: Dict) -> Optional[Dict]:
    """Segment customer with enhanced features"""
    try:
        if model is None:
            return None
            
        # Create DataFrame with proper feature names
        input_df = pd.DataFrame([{
            'Age': inputs['age'],
            'Income': inputs['income'],
            'Education': inputs['education'],
            'Policies': inputs['policy_count'],
            'Claims': inputs['claims_history'],
            'Years_with_Company': inputs['loyalty_years']
        }])
        
        # Preprocessing
        education_map = {"High School": 0, "College": 1, "Graduate": 2}
        input_df['Education_Num'] = input_df['Education'].map(education_map)
        
        # Feature engineering
        input_df['Policy_to_Claim_Ratio'] = input_df['Policies'] / (input_df['Claims'] + 1e-5)
        input_df['Customer_Value'] = (input_df['Income'] * input_df['Years_with_Company']) / 1000
        
        # Select features
        features = ['Age', 'Income', 'Education_Num', 'Policies', 
                   'Claims', 'Years_with_Company',
                   'Policy_to_Claim_Ratio', 'Customer_Value']
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(input_df[features])
        
        # Predict segment
        segment = model.predict(X)
        st.write(f"Segment: {segment}")
        if isinstance(segment, (np.ndarray, list)):
            segment = segment[0]  # Take first element if array
        
        segments = {
            0: {"name": "Low-Risk Value", "color": "#4CAF50", "icon": "🛡️", 
                "desc": "Customers with moderate spending and low claims",
                "strategy": "Target with loyalty programs and basic upsells"},
            1: {"name": "High-Value Loyal", "color": "#2196F3", "icon": "💎", 
                "desc": "High-income customers with multiple policies",
                "strategy": "Offer premium services and personalized solutions"},
            2: {"name": "Risk-Prone", "color": "#FF9800", "icon": "⚠️", 
                "desc": "Customers with higher claim frequency",
                "strategy": "Consider risk mitigation and education"},
            3: {"name": "Stable Long-Term", "color": "#9C27B0", "icon": "🏛️", 
                "desc": "Long-term customers with stable history",
                "strategy": "Maintain relationship with value-added services"}
        }
        
        return {
            'segment': segments.get(int(segment)),
            'segment_code': int(segment),
            'input_data': input_df
        }
    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        return None

# =============================================
# NLP FUNCTIONS
# =============================================

def translate_text(text: str, target_lang: str) -> Optional[str]:
    """Translate text to target language"""
    try:
        mbart_tokenizer.src_lang = "en_XX"
        encoded = mbart_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        lang_code = LANG_CODES[target_lang]

        with torch.no_grad():
            generated = mbart_model.generate(
                **encoded,
                forced_bos_token_id=mbart_tokenizer.lang_code_to_id[lang_code],
                max_length=512
            )

        return mbart_tokenizer.decode(generated[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return None

def summarize_text(text: str) -> Optional[str]:
    """Summarize English text"""
    try:
        inputs = bart_tokenizer([text], return_tensors="pt", max_length=1024, truncation=True).to(device)
        with torch.no_grad():
            summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=150)

        return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return None

# =============================================
# STREAMLIT UI COMPONENTS
# =============================================

def display_segment_card(segment_info: Dict):
    """Display segment information card"""
    if segment_info is None:
        st.warning("No segment information available")
        return
        
    st.markdown(f"""
    <div style='border-left: 5px solid {segment_info['color']}; padding: 10px; margin-bottom: 20px;'>
        <h3 style='color: {segment_info['color']};'>{segment_info['icon']} {segment_info['name']}</h3>
        <p><strong>Profile:</strong> {segment_info['desc']}</p>
        <p><strong>Strategy:</strong> {segment_info['strategy']}</p>
    </div>
    """, unsafe_allow_html=True)

def risk_prediction_tab(models: Dict):
    """Risk prediction interface"""
    st.header("📊 Insurance Risk Prediction")
    
    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            inputs = {
                'Customer_Age': st.slider('Age', 18, 100, 35),
                'Annual_Income': st.number_input('Income ($)', 0, 1000000, 50000, 1000),
                'Premium_Amount': st.number_input('Premium ($)', 0, 10000, 1000, 100),
                'Claim_Amount': st.number_input('Claim ($)', 0, 100000, 0, 1000)
            }
        
        with col2:
            inputs.update({
                'Gender': st.selectbox('Gender', ['Male', 'Female']),
                'Policy_Type': st.selectbox('Policy Type', ['Basic', 'Standard', 'Premium']),
                'Claim_History': st.selectbox('Claim History', ['None', '1-2', '3-5', '5+']),
                'Fraudulent_Claim': st.selectbox('Previous Fraud', ['No', 'Yes']),
                'Risk_Score': st.slider('Risk Score', 0, 100, 50)
            })
        
        if st.form_submit_button('Predict Risk'):
            if models.get("fraud_model") is not None:
                results = predict_risk(models["fraud_model"], inputs)
                if results:
                    st.success(f"**Result:** {results['result']} ({results['confidence']:.1f}% confidence)")
                    if sum(results['proba']) > 0:
                        st.bar_chart(pd.DataFrame({
                            'Risk': ['Low', 'High'],
                            'Probability': [results['proba'][0]*100, results['proba'][1]*100]
                        }).set_index('Risk'))
                else:
                    st.warning("Risk prediction failed")
            else:
                
                risk_score = (inputs['Claim_Amount'] / 1000) + (inputs['Risk_Score'] / 2)
                if inputs['Fraudulent_Claim'] == 'Yes':
                    risk_score += 30
                risk = "High Risk" if risk_score > 60 else "Low Risk"
                st.info(f"Fallback Risk Assessment: {risk} (Score: {risk_score:.1f})")

def fraud_detection_tab(models: Dict):
    """Fraud detection interface"""
    st.header("🕵️ Fraud Detection")
    
    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            inputs = {
                'claim_amount': st.number_input("Claim Amount ($)", 0, 100000, 5000),
                'incident_severity': st.selectbox("Severity", ["Minor", "Moderate", "Severe"]),
                'policy_duration': st.number_input("Policy Duration (months)", 0, 120, 12)
            }
        
        with col2:
            inputs.update({
                'days_to_report': st.number_input("Days to Report", 0, 365, 3),
                'previous_claims': st.number_input("Previous Claims", 0, 20, 0),
                'suspicious_docs': st.checkbox("Suspicious Docs")
            })
        
        if st.form_submit_button('Check for Fraud'):
            if models.get("fraudulent_claims_model") is not None:
                results = detect_fraud(models["fraudulent_claims_model"], inputs)
                if results:
                    color = "red" if results['result'] == 'Fraudulent' else "green"
                    st.markdown(f"**Result:** <span style='color:{color}'>{results['result']}</span> ({results['confidence']:.1f}%)", 
                               unsafe_allow_html=True)
                    if sum(results['proba']) > 0:
                        st.bar_chart(pd.DataFrame({
                            'Outcome': ['Legitimate', 'Fraudulent'],
                            'Probability': [results['proba'][0]*100, results['proba'][1]*100]
                        }).set_index('Outcome'))
                else:
                    st.warning("Fraud detection failed")
            else:
                
                score = 0
                if inputs['claim_amount'] > 10000: score += 1
                if inputs['days_to_report'] > 7: score += 1
                if inputs['suspicious_docs']: score += 1
                if inputs['previous_claims'] > 2: score += 1
                result = "Likely Fraudulent" if score >= 2 else "Likely Legitimate"
                st.info(f"Fallback Result: {result} (Score: {score}/4)")

def text_processing_tab():
    """Translation and summarization interface"""
    st.header("🌍 policy Anaysis ")
    
    task = st.radio("Choose Task", ["Translate", "Summarize"])
    text = st.text_area("Enter English Text", height=200)
    
    if task == "Translate":
        lang = st.selectbox("Select Target Language", list(LANG_CODES.keys()))
        if st.button("Translate"):
            if text.strip() == "":
                st.warning("Please enter some text.")
            else:
                with st.spinner("Translating..."):
                    translation = translate_text(text, lang)
                    if translation:
                        st.success(f"**Translated to {lang}:**")
                        st.write(translation)
                    else:
                        st.error("Translation failed. Please try again.")
    
    elif task == "Summarize":
        if st.button("Summarize"):
            if text.strip() == "":
                st.warning("Please enter some text.")
            else:
                with st.spinner("Summarizing..."):
                    summary = summarize_text(text)
                    if summary:
                        st.success("📝 Summary:")
                        st.write(summary)
                    else:
                        st.error("Summarization failed. Please try again.")

def sentiment_analysis_tab(models: Dict):
    """Sentiment analysis interface"""
    st.header("😊 Customer Sentiment Analysis")
    
    feedback = st.text_area("Customer Feedback", height=150)
    
    if st.button("Analyze Sentiment"):
        if not feedback.strip():
            st.warning("Please enter feedback text")
            return
            
        if models.get("sentiment_model") is not None:
            results = analyze_sentiment(models["sentiment_model"], feedback)
            if results:
                st.success(f"**Sentiment:** {results['sentiment']} ({results['confidence']:.1f}% confidence)")
            else:
                st.warning("Sentiment analysis failed")
        else:
            
            positive = sum(feedback.lower().count(word) for word in ["good", "great", "excellent", "happy"])
            negative = sum(feedback.lower().count(word) for word in ["bad", "poor", "terrible", "unhappy"])
            if positive > negative:
                st.success("**Sentiment:** Positive")
            elif negative > positive:
                st.error("**Sentiment:** Negative")
            else:
                st.info("**Sentiment:** Neutral")

def customer_segmentation_tab(models: Dict):
    """Enhanced Customer Segmentation Interface"""
    st.header("👥 Customer Segmentation Analysis")
    
    with st.expander("ℹ️ About Customer Segmentation"):
        st.write("""
        Our segmentation model classifies customers into distinct groups based on:
        - Demographic factors (age, income, education)
        - Policy behavior (number of policies, claims history)
        - Relationship factors (years with company)
        """)
    
    with st.form("segment_form"):
        st.subheader("Customer Details")
        
        col1, col2 = st.columns(2)
        with col1:
            inputs = {
                'age': st.number_input("Age", 18, 100, 35),
                'income': st.number_input("Income ($)", 0, 1000000, 50000),
                'education': st.selectbox("Education Level", ["High School", "College", "Graduate"])
            }
        
        with col2:
            inputs.update({
                'policy_count': st.number_input("Number of Policies", 1, 10, 1),
                'claims_history': st.number_input("Claims (5 years)", 0, 10, 1),
                'loyalty_years': st.number_input("Years with Company", 0, 50, 3)
            })
        
        submitted = st.form_submit_button("Analyze Customer Segment")
    
    if submitted:
        try:
            if models.get("customer_segmentation_model") is not None:
                with st.spinner("Analyzing customer profile..."):
                    results = segment_customer(models["customer_segmentation_model"], inputs)
                    
                    if results and results['segment']:
                        segment_info = results['segment']
                        input_df = results['input_data']
                        
                        st.success("Analysis Complete")
                        display_segment_card(segment_info)
                        
                        # Display metrics
                        st.subheader("Customer Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            risk_score = min(100, max(0, (inputs['claims_history']*20 + (100-inputs['age'])/2)))
                            st.metric("Risk Score", f"{risk_score}/100", 
                                    help="Higher claims and lower age increase risk")
                        with col2:
                            customer_value = (inputs['income'] * inputs['loyalty_years']) / 1000
                            st.metric("Customer Value", f"${customer_value:.1f}K", 
                                    help="Income multiplied by loyalty years")
                        with col3:
                            st.metric("Policy Efficiency", 
                                    f"{(inputs['policy_count'] / (inputs['claims_history'] + 1)):.1f}",
                                    help="Policies per claim")
                        
                        # Visualization
                        st.subheader("Segmentation Visualization")
                        fig, ax = plt.subplots(figsize=(8, 5))
                        
                        # Sample data for context
                        sample_data = pd.DataFrame({
                            'Age': np.random.normal(45, 15, 100),
                            'Income': np.random.normal(75000, 25000, 100)
                        })
                        
                        sns.scatterplot(
                            x='Age', 
                            y='Income',
                            data=sample_data,
                            color='lightgray',
                            alpha=0.5,
                            ax=ax
                        )
                        
                        # Plot current customer
                        sns.scatterplot(
                            x=[inputs['age']],
                            y=[inputs['income']],
                            color=segment_info['color'],
                            s=200,
                            ax=ax
                        )
                        
                        ax.set_title("Customer Position in Segmentation Space")
                        ax.set_xlabel("Age")
                        ax.set_ylabel("Annual Income ($)")
                        st.pyplot(fig)
                        
                        # Show raw data
                        with st.expander("View Processed Data"):
                            st.dataframe(input_df)
                    else:
                        st.error("Segmentation failed - no results returned")
            else:
                
                
                # Fallback segmentation logic
                if inputs['claims_history'] > 2:
                    segment = {
                        'name': "Risk-Prone", 
                        'color': "#FF9800", 
                        'icon': "⚠️",
                        'desc': "Customers with higher claim frequency",
                        'strategy': "Monitor closely and consider risk mitigation"
                    }
                elif inputs['income'] > 80000 and inputs['loyalty_years'] > 3:
                    segment = {
                        'name': "High-Value", 
                        'color': "#2196F3", 
                        'icon': "💎",
                        'desc': "High-income long-term customers",
                        'strategy': "Offer premium services and personalized solutions"
                    }
                elif inputs['loyalty_years'] > 5:
                    segment = {
                        'name': "Loyal", 
                        'color': "#9C27B0", 
                        'icon': "🏛️",
                        'desc': "Long-term customers with stable history",
                        'strategy': "Maintain relationship with value-added services"
                    }
                else:
                    segment = {
                        'name': "Standard", 
                        'color': "#4CAF50", 
                        'icon': "🛡️",
                        'desc': "Typical customers with average risk profile",
                        'strategy': "Target with standard offerings"
                    }
                
                display_segment_card(segment)
                
        
        except Exception as e:
            st.error(f"Segmentation error: {str(e)}")

# =============================================
# MAIN APPLICATION
# =============================================

def main():
    # Configure page
    st.set_page_config(
        page_title="InsureAnalytics Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .st-bb { background-color: #f0f2f6; }
        .st-at { background-color: #ffffff; }
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stButton>button { width: 100%; }
        .stAlert { padding: 20px; }
        div[data-testid="stExpander"] div[role="button"] p {
            font-size: 1.2rem;
            font-weight: bold;
        }
        .stMarkdown h1 {
            color: #2c3e50;
        }
        .stMarkdown h2 {
            color: #3498db;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    # Load NLP models
    global mbart_tokenizer, mbart_model, bart_tokenizer, bart_model
    mbart_tokenizer, mbart_model = load_translation_model()
    bart_tokenizer, bart_model = load_summarization_model()
    
    # Main interface
    st.title("Insurance Analytics Dashboard")
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <p style='margin: 0;'>Analyze customer risk, detect fraud, and segment your customer base.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tabs = st.tabs([
        "🔮 Risk Prediction",
        "🕵️ Fraud Detection", 
        "🌍 Policy Analysis",
        "😊 Sentiment Analysis",
        "👥 Customer Segments"
    ])
    
    # Render tabs
    with tabs[0]:
        risk_prediction_tab(models)
    
    with tabs[1]:
        fraud_detection_tab(models)
    
    with tabs[2]:
        text_processing_tab()
    
    with tabs[3]:
        sentiment_analysis_tab(models)
    
    with tabs[4]:
        customer_segmentation_tab(models)
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        
        # Check if we're in development mode
        dev_mode = all(model is None for model in models.values())
        
        if dev_mode:
            st.warning("Development Mode Active")
            st.info("Using fallback rules for all predictions")
        else:
            st.success("Production Mode Active")
            
            model_status = {
                "Risk Model": models.get("fraud_model"),
                "Fraud Model": models.get("fraudulent_claims_model"),
                "Sentiment Model": models.get("sentiment_model"),
                "Segmentation Model": models.get("customer_segmentation_model")
            }
            
            for name, model in model_status.items():
                status = "✅ Loaded" if model is not None else "⚠️ Using Fallback"
                color = "green" if model is not None else "orange"
                st.markdown(f"<span style='color:{color}'>{status}</span> - {name}", unsafe_allow_html=True)
        
        st.markdown("---")
       

if __name__ == "__main__":
    main()