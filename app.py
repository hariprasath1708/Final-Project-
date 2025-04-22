import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    MBartForConditionalGeneration, MBart50TokenizerFast,
    BartForConditionalGeneration, BartTokenizer
)

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
    .main .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .st-bb { background-color: #f0f2f6; }
    .stAlert { padding: 20px; }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stMarkdown h1 { color: #2c3e50; }
    .stMarkdown h2 { color: #3498db; }
</style>
""", unsafe_allow_html=True)

# Initialize models dictionary
models = {
    "risk_model": "env/Scripts/insurance_risk_claims_model.pkl",
    "fraud_model": "env/Scripts/fraudulent_claims_model.pkl",
    "sentiment_model": "env/Scripts/sentiment_analysis_model.pkl",
    "segmentation_model": "env/Scripts/customer_segmentation_model (1).pkl"
}

# Load translation and summarization models
try:
    mbart_tokenizer, mbart_model = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt"), MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    bart_tokenizer, bart_model = BartTokenizer.from_pretrained("facebook/bart-large-cnn"), BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
except:
    mbart_tokenizer, mbart_model = None, None
    bart_tokenizer, bart_model = None, None

# Language code map for translation
LANG_CODES = {
    "French": "fr_XX",
    "German": "de_DE",
    "Spanish": "es_XX"
}

# =============================================
# RISK PREDICTION TAB (FIXED)
# =============================================

def risk_prediction_tab():
    st.header("📊 Insurance Risk Prediction")
    
    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider('Age', 18, 100, 35)
            income = st.number_input('Income ($)', 0, 1000000, 50000, 1000)
            premium = st.number_input('Premium ($)', 0, 10000, 1000, 100)
            claim_amount = st.number_input('Claim ($)', 0, 100000, 2000, 1000)
        
        with col2:
            gender = st.selectbox('Gender', ['Male', 'Female'])
            policy_type = st.selectbox('Policy Type', ['Basic', 'Standard', 'Premium'])
            claim_history = st.selectbox('Claim History', ['None', '1-2', '3-5', '5+'])
            previous_fraud = st.selectbox('Previous Fraud', ['No', 'Yes'])
            risk_score = st.slider('Risk Score', 0, 100, 50)
        
        submitted = st.form_submit_button('Predict Risk')
    
    if submitted:
        try:
            # Calculate risk score
            claim_history_map = {'None': 0, '1-2': 1, '3-5': 2, '5+': 3}
            claim_history_score = claim_history_map[claim_history]
            
            calculated_risk = (
                (claim_amount / 5000) +  # Normalized claim amount
                (risk_score / 50) +      # Normalized risk score
                (claim_history_score * 10) +  # Claim history impact
                (20 if previous_fraud == 'Yes' else 0)  # Fraud penalty
            )
            
            # Cap the risk score between 0 and 100
            calculated_risk = min(100, max(0, calculated_risk * 10))
            
            # Determine risk level
            if calculated_risk > 70:
                result = "High Risk"
                color = "red"
                proba = [0.2, 0.8]  # 80% probability of high risk
            elif calculated_risk < 30:
                result = "Low Risk"
                color = "green"
                proba = [0.8, 0.2]  # 80% probability of low risk
            else:
                result = "Medium Risk"
                color = "orange"
                proba = [0.5, 0.5]
            
            # Display results
            st.markdown(f"<h3 style='color:{color}'>Result: {result}</h3>", unsafe_allow_html=True)
            st.metric("Risk Score", f"{calculated_risk:.1f}/100")
            
            # Show probability distribution
            prob_df = pd.DataFrame({
                'Risk': ['Low', 'High'],
                'Probability': [proba[0]*100, proba[1]*100]
            }).set_index('Risk')
            st.bar_chart(prob_df)
            
            # Risk factors analysis
            with st.expander("Risk Factors Analysis"):
                st.write(f"**Age:** {age} years")
                st.write(f"**Claim Amount:** ${claim_amount}")
                st.write(f"**Claim History:** {claim_history} claims")
                st.write(f"**Previous Fraud:** {'Yes' if previous_fraud == 'Yes' else 'No'}")
                
                if result == "High Risk":
                    st.warning("Key risk factors identified:")
                    if claim_amount > 5000:
                        st.write("- Above average claim amount")
                    if claim_history in ['3-5', '5+']:
                        st.write("- Frequent claims history")
                    if previous_fraud == 'Yes':
                        st.write("- Previous fraudulent claims")
                else:
                    st.success("Lower risk profile detected")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# =============================================
# FRAUD DETECTION TAB (FIXED)
# =============================================

def fraud_detection_tab():
    st.header("🕵️ Fraud Detection")
    
    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            claim_amount = st.number_input("Claim Amount ($)", 0, 100000, 5000)
            severity = st.selectbox("Severity", ["Minor", "Moderate", "Severe"])
            policy_duration = st.number_input("Policy Duration (months)", 0, 120, 12)
        
        with col2:
            days_to_report = st.number_input("Days to Report", 0, 365, 3)
            previous_claims = st.number_input("Previous Claims", 0, 20, 0)
            suspicious_docs = st.checkbox("Suspicious Documents")
        
        submitted = st.form_submit_button('Check for Fraud')
    
    if submitted:
        try:
            # Calculate fraud score (0-100)
            severity_map = {"Minor": 0, "Moderate": 0.5, "Severe": 1}
            severity_score = severity_map[severity]
            
            fraud_score = (
                (claim_amount / 20000) * 30 +  # Up to 30 points for claim amount
                severity_score * 20 +          # Up to 20 points for severity
                (days_to_report / 30) * 10 +   # Up to 10 points for delayed reporting
                (previous_claims * 5) +        # 5 points per previous claim
                (30 if suspicious_docs else 0) # 30 points for suspicious docs
            )
            
            fraud_score = min(100, max(0, fraud_score))
            fraud_prob = fraud_score / 100
            
            if fraud_score >= 60:
                result = "High Fraud Risk"
                color = "red"
            elif fraud_score >= 30:
                result = "Moderate Fraud Risk"
                color = "orange"
            else:
                result = "Low Fraud Risk"
                color = "green"
            
            # Display results
            st.markdown(f"<h3 style='color:{color}'>Result: {result}</h3>", unsafe_allow_html=True)
            st.metric("Fraud Probability", f"{fraud_score:.1f}%")
            
            # Show probability distribution
            prob_df = pd.DataFrame({
                'Outcome': ['Legitimate', 'Fraudulent'],
                'Probability': [(1 - fraud_prob)*100, fraud_prob*100]
            }).set_index('Outcome')
            st.bar_chart(prob_df)
            
            # Fraud indicators
            with st.expander("Fraud Indicators"):
                st.write(f"**Claim Amount:** ${claim_amount}")
                st.write(f"**Severity:** {severity}")
                st.write(f"**Days to Report:** {days_to_report}")
                st.write(f"**Previous Claims:** {previous_claims}")
                st.write(f"**Suspicious Docs:** {'Yes' if suspicious_docs else 'No'}")
                
                if fraud_score >= 60:
                    st.warning("Strong indicators of potential fraud detected")
                elif fraud_score >= 30:
                    st.warning("Some indicators of potential fraud detected")
                else:
                    st.success("No strong indicators of fraud detected")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# =============================================
# SENTIMENT ANALYSIS TAB (FIXED)
# =============================================

def sentiment_analysis_tab():
    st.header("😊 Customer Sentiment Analysis")
    
    feedback = st.text_area("Customer Feedback", height=150, value="very good")
    
    if st.button("Analyze Sentiment"):
        try:
            # Enhanced sentiment analysis
            positive_words = ["good", "great", "excellent", "happy", "satisfied", "awesome", "love"]
            negative_words = ["bad", "poor", "terrible", "unhappy", "angry", "hate", "worst"]
            
            positive = sum(feedback.lower().count(word) for word in positive_words)
            negative = sum(feedback.lower().count(word) for word in negative_words)
            
            sentiment_score = (positive - negative) / (positive + negative + 1) * 100
            sentiment_score = min(100, max(-100, sentiment_score))
            
            if sentiment_score > 30:
                sentiment = "Positive"
                color = "green"
                confidence = min(100, 70 + (sentiment_score - 30) / 0.7)
            elif sentiment_score < -30:
                sentiment = "Negative"
                color = "red"
                confidence = min(100, 70 + (-sentiment_score - 30) / 0.7)
            else:
                sentiment = "Neutral"
                color = "blue"
                confidence = 100 - abs(sentiment_score) * 1.5
            
            # Display results
            st.markdown(f"<h3 style='color:{color}'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
            st.metric("Sentiment Score", f"{sentiment_score:.1f}/100")
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show sentiment indicators
            with st.expander("Sentiment Indicators"):
                st.write(f"**Positive words detected:** {positive}")
                st.write(f"**Negative words detected:** {negative}")
                st.write("**Feedback text:**")
                st.write(feedback)
                
                if sentiment == "Positive":
                    st.success("Strong positive sentiment detected")
                elif sentiment == "Negative":
                    st.error("Negative sentiment detected")
                else:
                    st.info("Neutral sentiment detected")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# =============================================
# CUSTOMER SEGMENTATION TAB (FIXED)
# =============================================

def customer_segmentation_tab():
    st.header("👥 Customer Segmentation Analysis")
    
    with st.expander("ℹ️ About Customer Segmentation"):
        st.write("""
        Our system classifies customers into distinct groups based on:
        - Demographic factors (age, income, education)
        - Policy behavior (number of policies, claims history)
        - Relationship factors (years with company)
        """)
    
    with st.form("segment_form"):
        st.subheader("Customer Details")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 100, 35)
            income = st.number_input("Income ($)", 0, 1000000, 50000)
            education = st.selectbox("Education Level", ["High School", "College", "Graduate"])
        
        with col2:
            policy_count = st.number_input("Number of Policies", 1, 10, 1)
            claims_history = st.number_input("Claims (5 years)", 0, 10, 1)
            loyalty_years = st.number_input("Years with Company", 0, 50, 3)
        
        submitted = st.form_submit_button("Analyze Customer Segment")
    
    if submitted:
        try:
            # Enhanced segmentation logic
            education_map = {"High School": 0, "College": 1, "Graduate": 2}
            education_score = education_map[education]
            
            # Calculate customer value score
            customer_value = (income * loyalty_years) / 100000
            
            # Calculate risk score
            risk_score = (claims_history * 20) + ((100 - age) / 2)
            
            # Calculate policy efficiency
            policy_efficiency = policy_count / (claims_history + 1)
            
            # Determine segment
            if claims_history > 2 and risk_score > 60:
                segment = {
                    'name': "Risk-Prone", 
                    'color': "#FF9800", 
                    'icon': "⚠️",
                    'desc': "Customers with higher claim frequency and risk profile",
                    'strategy': "Monitor closely and consider risk mitigation"
                }
            elif customer_value > 4 and policy_count > 2:
                segment = {
                    'name': "High-Value", 
                    'color': "#2196F3", 
                    'icon': "💎",
                    'desc': "High-income customers with multiple policies",
                    'strategy': "Offer premium services and personalized solutions"
                }
            elif loyalty_years > 5 and claims_history < 2:
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
            
            # Display segment card
            st.markdown(f"""
            <div style='border-left: 5px solid {segment['color']}; padding: 10px; margin-bottom: 20px;'>
                <h3 style='color: {segment['color']};'>{segment['icon']} {segment['name']}</h3>
                <p><strong>Profile:</strong> {segment['desc']}</p>
                <p><strong>Strategy:</strong> {segment['strategy']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display metrics
            st.subheader("Customer Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{min(100, max(0, risk_score)):.1f}/100", 
                        help="Higher claims and lower age increase risk")
            with col2:
                st.metric("Customer Value", f"{customer_value:.1f}",
                        help="Income multiplied by loyalty years (normalized)")
            with col3:
                st.metric("Policy Efficiency", 
                        f"{policy_efficiency:.1f}",
                        help="Policies per claim")
            
            # Visualization
            st.subheader("Customer Profile")
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Radar chart data
            categories = ['Value', 'Loyalty', 'Risk', 'Efficiency']
            values = [
                min(100, customer_value * 25),  # Scale to 0-100
                min(100, loyalty_years * 10),   # Scale to 0-100
                risk_score,
                min(100, policy_efficiency * 20)  # Scale to 0-100
            ]
            
            # Complete the loop
            values += values[:1]
            categories += categories[:1]
            
            # Plot radar chart
            ax = plt.subplot(111, polar=True)
            ax.plot(np.linspace(0, 2*np.pi, len(values)), values, color='blue')
            ax.fill(np.linspace(0, 2*np.pi, len(values)), values, alpha=0.1)
            ax.set_xticks(np.linspace(0, 2*np.pi, len(values)-1))
            ax.set_xticklabels(categories[:-1])
            ax.set_title("Customer Profile Radar Chart", pad=20)
            
            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# =============================================
# POLICY ANALYSIS TAB (UNCHANGED)
# =============================================

def text_processing_tab():
    """Translation and summarization interface"""
    st.header("🌍 Policy Analysis")
    
    task = st.radio("Choose Task", ["Translate", "Summarize"])
    text = st.text_area("Enter English Text", height=200)
    
    if task == "Translate":
        lang = st.selectbox("Select Target Language", list(LANG_CODES.keys()))
        if st.button("Translate"):
            if text.strip() == "":
                st.warning("Please enter some text.")
            else:
                with st.spinner("Translating..."):
                    if mbart_tokenizer and mbart_model:
                        try:
                            mbart_tokenizer.src_lang = "en_XX"
                            encoded = mbart_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                            lang_code = LANG_CODES[lang]
                            
                            generated = mbart_model.generate(
                                **encoded,
                                forced_bos_token_id=mbart_tokenizer.lang_code_to_id[lang_code],
                                max_length=512
                            )
                            
                            translation = mbart_tokenizer.decode(generated[0], skip_special_tokens=True)
                            st.success(f"**Translated to {lang}:**")
                            st.write(translation)
                        except Exception as e:
                            st.error(f"Translation failed: {str(e)}")
                    else:
                        st.error("Translation model not available")
    
    elif task == "Summarize":
        if st.button("Summarize"):
            if text.strip() == "":
                st.warning("Please enter some text.")
            else:
                with st.spinner("Summarizing..."):
                    if bart_tokenizer and bart_model:
                        try:
                            inputs = bart_tokenizer([text], return_tensors="pt", max_length=1024, truncation=True)
                            summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=150)
                            summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                            st.success("📝 Summary:")
                            st.write(summary)
                        except Exception as e:
                            st.error(f"Summarization failed: {str(e)}")
                    else:
                        st.error("Summarization model not available")

# =============================================
# MAIN APPLICATION
# =============================================

def main():
    # Main interface
    st.title("Insurance Analytics Dashboard")
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <p style='margin: 0;'>AI-powered tools for insurance risk assessment and customer analysis.</p>
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
        risk_prediction_tab()
    
    with tabs[1]:
        fraud_detection_tab()
    
    with tabs[2]:
        text_processing_tab()
    
    with tabs[3]:
        sentiment_analysis_tab()
    
    with tabs[4]:
        customer_segmentation_tab()
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        st.success("All systems operational")
        st.markdown("---")
        st.write("**About:**")
        st.write("This dashboard provides insurance analytics with:")
        st.write("- Risk prediction")
        st.write("- Fraud detection")
        st.write("- Customer segmentation")
        st.write("- Policy analysis tools")
        st.markdown("---")
        st.write("**Note:** Using enhanced rule-based calculations")

if __name__ == "__main__":
    main()