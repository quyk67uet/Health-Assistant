from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load models
diabetes_model = pickle.load(open('Models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('Models/heart_model.sav', 'rb'))
parkinsons_model = pickle.load(open('Models/parkinson_model.sav', 'rb'))

# Request data models
class DiabetesData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

class HeartDiseaseData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

class ParkinsonsData(BaseModel):
    fo: float
    fhi: float
    flo: float
    Jitter_percent: float
    Jitter_Abs: float
    RAP: float
    PPQ: float
    DDP: float
    Shimmer: float
    Shimmer_dB: float
    APQ3: float
    APQ5: float
    APQ: float
    DDA: float
    NHR: float
    HNR: float
    RPDE: float
    DFA: float
    spread1: float
    spread2: float
    D2: float
    PPE: float

# Prediction endpoints

@app.post("/predict-diabetes")
async def predict_diabetes(data: DiabetesData):
    input_data = [[
        data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness,
        data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age
    ]]
    prediction = diabetes_model.predict(input_data)
    return {"prediction": int(prediction[0])}

@app.post("/predict-heart-disease")
async def predict_heart_disease(data: HeartDiseaseData):
    input_data = [[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak,
        data.slope, data.ca, data.thal
    ]]
    prediction = heart_disease_model.predict(input_data)
    return {"prediction": int(prediction[0])}

@app.post("/predict-parkinsons")
async def predict_parkinsons(data: ParkinsonsData):
    input_data = [[
        data.fo, data.fhi, data.flo, data.Jitter_percent, data.Jitter_Abs,
        data.RAP, data.PPQ, data.DDP, data.Shimmer, data.Shimmer_dB,
        data.APQ3, data.APQ5, data.APQ, data.DDA, data.NHR,
        data.HNR, data.RPDE, data.DFA, data.spread1, data.spread2,
        data.D2, data.PPE
    ]]
    prediction = parkinsons_model.predict(input_data)
    return {"prediction": int(prediction[0])}


# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
GEMINI_KEY = os.getenv("GEMINI_KEY")
DB_COLLECTION = os.getenv("DB_COLLECTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Initialize MongoDB Atlas client
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[DB_NAME]  # Use the database name from .env
collection = db[DB_COLLECTION]  # Use the collection name from .env

genai.configure(api_key=GEMINI_KEY)
llm = genai.GenerativeModel('gemini-1.5-flash')
# Initialize SentenceTransformer for embedding generation
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a given text using the SentenceTransformer model."""
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)
    return embedding.tolist()

def vector_search(user_query: str, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.
    Args:
        user_query (str): The user's query string.
        collection (MongoCollection): The MongoDB collection to search.
    Returns:
        list: A list of matching documents.
    """
    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)
    
    if not query_embedding:
        return {"error": "Invalid query or embedding generation failed."}
    
    # Define the vector search pipeline
    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "Embedding",
            "numCandidates": 150,  # Number of candidate matches to consider
            "limit": 4  # Return top 4 matches
        }
    }
    
    unset_stage = {
        "$unset": "Embedding"  # Exclude the 'embedding' field from the results
    }
    
    project_stage = {
        "$project": {
            "_id": 0,
            "Name": 1,  # Include the Name field
            "Symptoms": 1,  # Include the Symptoms field
            "Treatments": 1,  # Include the Treatments field
            "score": {
                "$meta": "vectorSearchScore"  # Include the search score
            }
        }
    }
    
    pipeline = [vector_search_stage, unset_stage, project_stage]
    
    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)

def get_search_result(query, collection):

    get_knowledge = vector_search(query, collection)

    search_result = ""
    for result in get_knowledge:
        print('---result', result)
        search_result += f"Name: {result.get('Name', 'N/A')}, Symptoms: {result.get('Symptoms', 'N/A')}, Treatments: {result.get('Treatments', 'N/A')}\n"

    return search_result

def generate_response(query, collection):
    context = get_search_result(query, collection)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = llm.generate_content(prompt)
    return response.text

@app.post("/chat_with_knowledge_base")
def chat_with_knowledge_base(query: str = Query(...)):
    """
    Endpoint to chat with the knowledge base using a vector search and language model.
    """
    try:

        # Use the LLM model to generate an answer based on the retrieved context
        llm_response = generate_response(query, collection)
        print(llm_response)
        return JSONResponse(content={"response": llm_response}, status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))