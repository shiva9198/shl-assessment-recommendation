import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

# --- LangChain Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# --- 1. CONFIGURATION ---
logging.basicConfig()

if "OPENROUTER_API_KEY" not in os.environ:
    print("Error: OPENROUTER_API_KEY not found in environment variables.")
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

@app.get("/health")
def health_check():
    return {"status": "healthy"}

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


# --- 6. RUN THE SERVER ---
if __name__ == "__main__":
    import uvicorn
    print("Starting RATE-LIMIT-SAFE API server at http://127.0.0.1:8000")
    print("(Using direct vector similarity search - no LLM calls)")
    uvicorn.run(app, host="127.0.0.1", port=8000)