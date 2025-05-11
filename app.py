import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Autism Spectrum Disorder Detection",
    page_icon="🧩",
    layout="wide"
)

# Custom CSS to improve UI/UX (must come after set_page_config)
st.markdown("""
<style>
    body {
        background-color: #ffc1cc;
        
        background-position: 0 0, 10px 10px;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: bold
        font-family: Times New Roman
        color: #4B6587;
        margin-bottom: 1rem;
        text-align: center;
        padding: 1rem;
        border-bottom: 2px solid #F7F6F2;
    }
    
    .subtitle {
        font-size: 2.5rem;
        color: #4B6587;
        margin-top: 1rem;
    }
    
    .section-header {
        font-size: 1.4rem;
        color: #4A6587;
        margin-top: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #F7F6F2;
    }
    
    .question-text {
        font-weight: 500;
        color: #2F2E41;
    }
    
    .info-box {
        background-color: #F7F9FC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4B6587;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .warning-box {
        background-color: #FFF4E6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF9F1C;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .result-positive {
        background-color: #FFE8E8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #FFCAD4;
        box-shadow: 0 2px 8px rgba(255,202,212,0.3);
    }
    
    .result-negative {
        background-color: #E3F8E9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #C1E7C5;
        box-shadow: 0 2px 8px rgba(193,231,197,0.3);
    }
    
    .footer {
        text-align: center;
        color: #888888;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #F7F6F2;
    }
    
    .stButton > button {
        background-color: #4B6587;
        color: white;
        font-weight: 500;
        padding: 0.5rem 2rem;
        border-radius: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #39516a;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }
    
    .st-eb {
        border-radius: 0.5rem;
    }
    
    div[data-testid="stExpander"] details summary p {
        font-weight: 600;
    }
    
    /* Custom card for horizontal navigation instead of sidebar */
    .horizontal-nav {
        background-color: white;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #4B6587;
    }
    
    .horizontal-nav-item {
        text-align: center;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        border-radius: 0.3rem;
        background-color: #f1f3f8;
        transition: all 0.2s ease;
    }
    
    .horizontal-nav-item:hover {
        background-color: #e1e5f0;
        transform: translateY(-2px);
    }
    
    .horizontal-nav-icon {
        margin-bottom: 0.3rem;
    }
    
    /* Improved visualization for the probability bar */
    .probability-container {
        width: 100%;
        height: 2.5rem;
        background: linear-gradient(to right, #5de88f, #f6e683, #ff8a8a);
        border-radius: 1rem;
        position: relative;
        margin: 1.5rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .probability-marker {
        position: absolute;
        height: 100%;
        width: 4px;
        background-color: white;
        border: 2px solid #333;
        border-radius: 4px;
        transform: translateX(-50%);
    }
    
    .probability-label {
        position: absolute;
        top: -25px;
        transform: translateX(-50%);
        background-color: white;
        border-radius: 0.3rem;
        padding: 0.2rem 0.5rem;
        font-weight: bold;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    
    /* App container with subtle shadow and border */
    .app-container {
        background-color: white;
        border-radius: 0.8rem;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #eaecef;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('notebooks/best_modelpkl')

@st.cache_resource
def load_encoders():
    with open("notebooks/encoders.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    encoders = load_encoders()
    model_loaded = True
except:
    model_loaded = False
    st.error("Model or encoders not found. Please train the model first by running the model_development.ipynb notebook.")

# Main app title with enhanced styling
st.markdown('<div class="main-title">Autism Spectrum Disorder Detection</div>', unsafe_allow_html=True)

# Create tabs with improved styling
tab1, tab2 = st.tabs(["📋 Detection Tool", "ℹ️ About ASD"])

with tab1:
    #st.markdown('<div class="app-container">', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Autism Detection Tool</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Model Not Loaded</h3>
            <p>Please train the model first by running the model_development.ipynb notebook.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.expander("📝 How to complete this assessment", expanded=True):
            st.write("""
            - Answer all questions as honestly as possible
            - Select 'Yes' or 'No' for each behavioral question
            - Complete all demographic information
            - Click 'Submit for Analysis' to see results
            - Remember: This is a screening tool, not a diagnosis
            """)
        
        st.markdown('<div class="section-header">Behavioral Questions</div>', unsafe_allow_html=True)
        st.write("Answer the following behavioral questions:")

        # Organize behavioral questions in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p class="question-text">1. I often notice small sounds when others do not</p>', unsafe_allow_html=True)
            a1 = st.selectbox("", ["Yes", "No"], key="a1", label_visibility="collapsed")
            
            st.markdown('<p class="question-text">3. I find it easy to do more than one thing at once</p>', unsafe_allow_html=True)
            a3 = st.selectbox("", ["Yes", "No"], key="a3", label_visibility="collapsed")
            
            st.markdown('<p class="question-text">5. I find it easy to \'read between the lines\'</p>', unsafe_allow_html=True)
            a5 = st.selectbox("", ["Yes", "No"], key="a5", label_visibility="collapsed")
            
            st.markdown('<p class="question-text">7. I find it difficult to work out characters\' intentions</p>', unsafe_allow_html=True)
            a7 = st.selectbox("", ["Yes", "No"], key="a7", label_visibility="collapsed")
            
            st.markdown('<p class="question-text">9. I find it easy to understand feelings</p>', unsafe_allow_html=True)
            a9 = st.selectbox("", ["Yes", "No"], key="a9", label_visibility="collapsed")

        with col2:
            st.markdown('<p class="question-text">2. I usually concentrate more on the whole picture</p>', unsafe_allow_html=True)
            a2 = st.selectbox("", ["Yes", "No"], key="a2", label_visibility="collapsed")
            
            st.markdown('<p class="question-text">4. I can switch back to what I was doing very quickly</p>', unsafe_allow_html=True)
            a4 = st.selectbox("", ["Yes", "No"], key="a4", label_visibility="collapsed")
            
            st.markdown('<p class="question-text">6. I know how to tell if someone is bored</p>', unsafe_allow_html=True)
            a6 = st.selectbox("", ["Yes", "No"], key="a6", label_visibility="collapsed")
            
            st.markdown('<p class="question-text">8. I like to collect information</p>', unsafe_allow_html=True)
            a8 = st.selectbox("", ["Yes", "No"], key="a8", label_visibility="collapsed")
            
            st.markdown('<p class="question-text">10. I find it difficult to work out intentions</p>', unsafe_allow_html=True)
            a10 = st.selectbox("", ["Yes", "No"], key="a10", label_visibility="collapsed")

        st.markdown('<div class="section-header">Personal Information</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)

        with col3:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            age = st.number_input("Age", min_value=1, max_value=100)
            jundice = st.selectbox("Born with jaundice?", ["Yes", "No"])
            autism = st.selectbox("Family history of autism?", ["Yes", "No"])

        with col4:
            country = st.selectbox("Country of residence", encoders['contry_of_res'].classes_)
            ethnicity = st.selectbox("Ethnicity", encoders['ethnicity'].classes_)
            used_app_before = st.selectbox("Used this app before?", ["Yes", "No"])
            relation = st.selectbox("Relation to individual", encoders['relation'].classes_)

        def process_binary(answer):
            return 1 if answer == "Yes" else 0

        # Center the submit button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            submit_button = st.button("Submit for Analysis", use_container_width=True)

        if submit_button:
            try:
                features = [
                    process_binary(a1), process_binary(a2), process_binary(a3),
                    process_binary(a4), process_binary(a5), process_binary(a6),
                    process_binary(a7), process_binary(a8), process_binary(a9),
                    process_binary(a10), 
                    age,
                    1 if gender == "Male" else 0, 
                    encoders['ethnicity'].transform([ethnicity])[0],
                    process_binary(jundice),
                    process_binary(autism),
                    encoders['contry_of_res'].transform([country])[0],
                    process_binary(used_app_before),
                    sum([  # Keep the original result calculation
                        process_binary(a1), process_binary(a2), process_binary(a3),
                        process_binary(a4), process_binary(a5), process_binary(a6),
                        process_binary(a7), process_binary(a8), process_binary(a9),
                        process_binary(a10)
                    ]),
                    encoders['relation'].transform([relation])[0]
                ]

                prediction = model.predict([features])[0]
                probability = model.predict_proba([features])[0][1]

                st.markdown('<div class="subtitle">Analysis Result</div>', unsafe_allow_html=True)
                
                # Add a divider
                st.markdown("<hr>", unsafe_allow_html=True)
                
                result_col1, result_col2 = st.columns([1, 2])

                with result_col1:
                    if prediction == 1:
                        st.markdown("""
                        <div class="result-positive">
                            <h3>Indicators of ASD detected</h3>
                            <p>Confidence: {:.1f}%</p>
                        </div>
                        """.format(probability*100), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-negative">
                            <h3>No indicators of ASD detected</h3>
                            <p>Confidence: {:.1f}%</p>
                        </div>
                        """.format((1-probability)*100), unsafe_allow_html=True)

                with result_col2:
                    # Improved visualization instead of matplotlib
                    st.markdown(f"""
                    <div class="probability-container">
                        <div class="probability-marker" style="left: {probability*100}%;"></div>
                        <div class="probability-label" style="left: {probability*100}%;">{probability*100:.1f}%</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                        <div>Low probability</div>
                        <div>High probability</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="warning-box">
                <h4>⚠️ Disclaimer</h4>
                <p>This assessment is based on a machine learning model and is not a clinical diagnosis. 
                Please consult with a healthcare professional for proper evaluation and diagnosis.</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Ensure all inputs match the expected format and the model is properly trained.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    #st.markdown('<div class="app-container">', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">About Autism Spectrum Disorder</div>', unsafe_allow_html=True)
    
    # Create tabs within the About section
    about_tab1, about_tab2, about_tab3 = st.tabs(["What is ASD?", "Signs & Symptoms", "Diagnosis & Support"])
    
    with about_tab1:
        st.markdown("""
        <div class="info-box">
        <h3>Understanding Autism Spectrum Disorder</h3>
        <p>Autism Spectrum Disorder (ASD) is a neurodevelopmental condition that affects communication,
        social interaction, and can include repetitive behaviors. It's a "spectrum" because it varies greatly between individuals.</p>
        
        <p>ASD is typically characterized by:</p>
        <ul>
            <li>Differences in social communication and interaction</li>
            <li>Restricted or repetitive patterns of behavior or interests</li>
            <li>Symptoms that appear in early childhood</li>
            <li>Symptoms that affect daily functioning</li>
        </ul>
        
        <p>Every person with autism has a unique set of strengths and challenges, and the ways in which
        these characteristics present can vary widely.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with about_tab2:
        col_signs1, col_signs2 = st.columns(2)
        
        with col_signs1:
            st.markdown("""
            <h3>Social Communication Signs</h3>
            <ul>
                <li>Difficulty with back-and-forth conversation</li>
                <li>Reduced sharing of interests or emotions</li>
                <li>Challenges in understanding or using gestures</li>
                <li>Difficulties with eye contact</li>
                <li>Troubles with developing and maintaining relationships</li>
            </ul>
            """, unsafe_allow_html=True)
    
    with about_tab3:
        st.markdown("""
        <h3>Diagnosis Process</h3>
        <p>Autism diagnosis typically involves:</p>
        <ol>
            <li>Developmental screening during regular checkups</li>
            <li>Comprehensive diagnostic evaluation</li>
            <li>Assessment by specialists (neurologists, psychologists, etc.)</li>
        </ol>
        
        <h3>Support & Resources</h3>
        <p>Early intervention services can significantly improve outcomes. These might include:</p>
        <ul>
            <li>Behavioral therapy</li>
            <li>Speech-language therapy</li>
            <li>Occupational therapy</li>
            <li>Educational support</li>
        </ul>
        
        <h3>Helpful Links</h3>
        <ul>
            <li><a href="https://www.autismspeaks.org/" target="_blank">Autism Speaks</a></li>
            <li><a href="https://www.autism-society.org/" target="_blank">Autism Society</a></li>
            <li><a href="https://www.cdc.gov/ncbddd/autism/" target="_blank">CDC Autism Information</a></li>
        </ul>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

