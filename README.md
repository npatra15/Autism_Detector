# Autism Spectrum Disorder Detection Tool
A web application for ASD screening using machine learning.
Overview
This project provides a Streamlit-based tool for early screening of Autism Spectrum Disorder (ASD) traits based on behavioral questions and demographic information. The app uses a trained machine learning model to identify potential indicators of ASD and provides users with immediate feedback.

##  Live Demo

Click the link below to try the app live on Streamlit Cloud:

🔗 [Live App: Autism Detection](https://share.streamlit.io/user/npatra15)

# Features

10-question behavioral assessment based on established ASD screening methods
Collection of relevant demographic information
Real-time prediction using machine learning
Visual representation of results with probability scores
Educational information about ASD
Modern, user-friendly interface

# Installation
bash# Clone the repository\\
git clone https://github.com/yourusername/asd-detection-tool.git\\
cd asd-detection-tool

Create and activate virtual environment (optional):\\
python -m venv venv\\
source venv/bin/activate  # On Windows: venv\Scripts\activate\\

Install dependencies:\\
pip install -r requirements.txt\\

Run the application:\\
streamlit run app.py

Tech Stack used and Requiremnets:\\
Python 3.8+,
Streamlit,
Pandas,
NumPy,
Matplotlib,
Seaborn,
Scikit-learn,
Joblib\\

How It Works:\\

Users answer 10 behavioral questions and provide demographic information
The app processes responses using the same preprocessing as during model training
A machine learning model evaluates the data to generate a prediction
Results are displayed with a probability score and appropriate guidance


Disclaimer
This application is for screening purposes only and not a diagnostic tool. Always consult healthcare professionals for proper diagnosis of Autism Spectrum Disorder.
