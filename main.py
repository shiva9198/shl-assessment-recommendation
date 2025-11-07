import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# --- 1. CONFIGURATION ---
logging.basicConfig()

# Load environment variables from .env file
load_dotenv()

if "OPENROUTER_API_KEY" not in os.environ:
    print("Error: OPENROUTER_API_KEY not found in environment variables.")
    print("Make sure you have a .env file with OPENROUTER_API_KEY set.")
    exit()

DB_PERSIST_DIRECTORY = "chroma_db_shl"

# --- 2. LOAD THE VECTOR DB ---
print("Loading vector database...")
try:
    embedding = OpenAIEmbeddings(
        model="openai/text-embedding-ada-002",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.environ["OPENROUTER_API_KEY"]
    )
    vectordb = Chroma(
        persist_directory=DB_PERSIST_DIRECTORY, 
        embedding_function=embedding
    )
    print("✅ Vector database loaded successfully.")
except Exception as e:
    print(f"Error loading vector database: {e}")
    exit()

# --- 3. INITIALIZE SIMPLE RETRIEVER (NO LLM NEEDED) ---
print("Initializing simple retriever...")
retriever = vectordb.as_retriever(search_kwargs={"k": 10}) # Get top 10 docs directly
print("✅ Retriever initialized successfully.")


# --- 4. DEFINE THE FASTAPI APP & API MODELS ---
app = FastAPI(title="SHL Assessment Recommendation API")

# Mount static files for web interface
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class Recommendation(BaseModel):
    name: str
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: list[str]

class RecommendResponse(BaseModel):
    recommended_assessments: list[Recommendation]


# --- 5. DEFINE API ENDPOINTS ---

@app.get("/")
def read_root():
    """Serve the web interface"""
    return FileResponse('static/index.html')

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "database_products": vectordb._collection.count()
    }

@app.post("/recommend", response_model=RecommendResponse)
def recommend_assessments(request: QueryRequest):
    """
    Receives a query, uses direct vector similarity search to get the top 10
    most relevant documents, and returns them directly.
    
    This implementation bypasses LLM calls entirely to avoid rate limits.
    """
    print(f"Received query: {request.query[:70]}...") # Truncate log
    
    try:
        # 1. Get the 10 most relevant documents using direct similarity search
        # This ONLY uses embeddings, no LLM calls needed
        retrieved_docs = retriever.invoke(request.query)
        
        print(f"Retrieved {len(retrieved_docs)} documents.")

        # 2. Build the final JSON response from the metadata
        recommendations = []
        for doc in retrieved_docs:
            meta = doc.metadata
            
            # Convert 'test_type' string from DB back to a list
            test_type_list = [t.strip() for t in meta.get("test_type", "Unknown").split(',')]

            rec = Recommendation(
                name=meta.get("name", "Unknown Name"),
                url=meta.get("url", ""),  # Use 'url' field from metadata
                adaptive_support=meta.get("adaptive_support", "No"),
                description=meta.get("description", "No description available."),
                duration=int(meta.get("duration", 0)),
                remote_support=meta.get("remote_support", "Yes"),
                test_type=test_type_list
            )
            recommendations.append(rec)
            
            # Ensure we only return a max of 10
            if len(recommendations) >= 10:
                break
        
        print(f"Returning {len(recommendations)} recommendations.")
        return RecommendResponse(recommended_assessments=recommendations)

    except Exception as e:
        print(f"CRITICAL ERROR during retrieval: {e}")
        # Return an empty list instead of crashing
        return RecommendResponse(recommended_assessments=[])

@app.post("/recommend-simple")
async def recommend_simple(request: QueryRequest):
    """
    Simplified endpoint for web interface that returns just URLs and scores
    """
    try:
        # Get similar documents using vector search with scores
        results = vectordb.similarity_search_with_score(request.query, k=10)
        
        recommendations = []
        for doc, score in results:
            recommendations.append({
                "assessment_url": doc.metadata.get("url", ""),
                "score": float(score)
            })
        
        return {"recommendations": recommendations}
        
    except Exception as e:
        print(f"Error in simple recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 6. RUN THE SERVER ---
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable for deployment compatibility
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting RATE-LIMIT-SAFE API server at http://{host}:{port}")
    print("(Using direct vector similarity search - no LLM calls)")
    uvicorn.run(app, host=host, port=port)