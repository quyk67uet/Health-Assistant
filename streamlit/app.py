import streamlit as st
import requests
import os
from streamlit_option_menu import option_menu
from streamlit_chat import message
import google.generativeai as genai

# Set base_url for FastAPI
base_url = os.getenv("BASE_URL", "http://localhost:8000")

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Custom CSS for styling
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-message.user {
        background-color: #DCF8C6;
        text-align: right;
    }
    .chat-message.bot {
        background-color: #E6E6E6;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Chatbot'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'chat'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    if st.button('Diabetes Test Result'):
        user_input = {
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "BloodPressure": BloodPressure,
            "SkinThickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
            "Age": Age
        }
        response = requests.post(f"{base_url}/predict-diabetes", json=user_input)
        result = response.json()
        st.success('The person is diabetic' if result["prediction"] == 1 else 'The person is not diabetic')

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    if st.button('Heart Disease Test Result'):
        user_input = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }
        response = requests.post(f"{base_url}/predict-heart-disease", json=user_input)
        result = response.json()
        st.success('The person is having heart disease' if result["prediction"] == 1 else 'The person does not have any heart disease')

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    if st.button("Parkinson's Test Result"):
        user_input = {
            'fo': fo, 'fhi': fhi, 'flo': flo, 'Jitter_percent': Jitter_percent,
            'Jitter_Abs': Jitter_Abs, 'RAP': RAP, 'PPQ': PPQ, 'DDP': DDP,
            'Shimmer': Shimmer, 'Shimmer_dB': Shimmer_dB, 'APQ3': APQ3,
            'APQ5': APQ5, 'APQ': APQ, 'DDA': DDA, 'NHR': NHR, 'HNR': HNR,
            'RPDE': RPDE, 'DFA': DFA, 'spread1': spread1, 'spread2': spread2,
            'D2': D2, 'PPE': PPE
        }
        response = requests.post(f"{base_url}/predict-parkinsons", json=user_input)
        result = response.json()
        st.success("The person has Parkinson's disease" if result["prediction"] == 1 else "The person does not have Parkinson's disease")

# Function to send query to chatbot API
def send_query_to_chatbot(query):
    try:
        response = requests.post(f"{base_url}/chat_with_knowledge_base", params={"query": query})
        response.raise_for_status()
        return response.json().get("response", "Sorry, there was an error with the chatbot API.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return "Sorry, there was an error with the chatbot API."

genai.configure(api_key='AIzaSyCwru2eJko9pYO5Iert88v0KWa6zBfFiUA')
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Chatbot Page
if selected == 'Chatbot':
    st.title("🤖 MediBot")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask MediBot...")
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        response = send_query_to_chatbot(user_prompt)

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(response)
        