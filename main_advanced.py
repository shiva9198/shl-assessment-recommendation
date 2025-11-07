import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
import re
from collections import Counter
from typing import List, Dict, Set, Tuple
import numpy as np

# --- LangChain Imports ---
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. CONFIGURATION ---
logging.basicConfig()

if "OPENROUTER_API_KEY" not in os.environ:
    print("Error: OPENROUTER_API_KEY not found in environment variables.")
    exit()

DB_PERSIST_DIRECTORY = "chroma_db_shl"

# --- 2. ADVANCED KEYWORD-SEMANTIC FUSION SYSTEM ---

# Comprehensive keyword mapping with weights
SKILL_DOMAINS = {
    'programming': {
        'keywords': ['java', 'python', 'javascript', 'js', 'sql', 'c++', 'php', 'ruby', 'go', 'rust'],
        'synonyms': ['coding', 'programming', 'software development', 'development', 'developer'],
        'weight': 3.5,
        'assessment_types': ['Technical']
    },
    'data_science': {
        'keywords': ['data', 'analytics', 'analyst', 'statistics', 'machine learning', 'ml', 'ai'],
        'synonyms': ['data analysis', 'data scientist', 'statistical analysis', 'artificial intelligence'],
        'weight': 3.0,
        'assessment_types': ['Technical', 'Analytical']
    },
    'leadership': {
        'keywords': ['leader', 'leadership', 'manager', 'management', 'ceo', 'coo', 'director'],
        'synonyms': ['lead', 'supervise', 'executive', 'head', 'senior management'],
        'weight': 2.8,
        'assessment_types': ['Leadership', 'Behavioral']
    },
    'sales': {
        'keywords': ['sales', 'selling', 'revenue', 'business development', 'account management'],
        'synonyms': ['commercial', 'client relations', 'customer service', 'account executive'],
        'weight': 2.5,
        'assessment_types': ['Behavioral', 'Sales']
    },
    'creative': {
        'keywords': ['creative', 'design', 'content', 'writer', 'marketing', 'brand'],
        'synonyms': ['creativity', 'copywriter', 'content creation', 'graphic design'],
        'weight': 2.3,
        'assessment_types': ['Creative', 'Behavioral']
    },
    'technical_support': {
        'keywords': ['support', 'technical support', 'troubleshooting', 'help desk'],
        'synonyms': ['customer support', 'technical assistance', 'it support'],
        'weight': 2.0,
        'assessment_types': ['Technical', 'Behavioral']
    },
    'collaboration': {
        'keywords': ['collaborate', 'teamwork', 'team', 'communication', 'interpersonal'],
        'synonyms': ['cooperation', 'team player', 'working together', 'group work'],
        'weight': 1.8,
        'assessment_types': ['Behavioral']
    }
}

# Experience level mapping
EXPERIENCE_LEVELS = {
    'entry': ['entry', 'junior', 'graduate', 'new', 'fresh', '0-1', '0-2', 'intern'],
    'mid': ['mid', 'intermediate', '2-4', '3-5', 'experienced'],
    'senior': ['senior', 'lead', 'principal', '5+', '5-7', '7+', 'expert'],
    'executive': ['executive', 'director', 'vp', 'ceo', 'coo', 'head of']
}

def extract_skills_and_context(query: str) -> Dict:
    """Extract skills, experience level, and context from query."""
    query_lower = query.lower()
    
    # Extract skills with weights
    detected_skills = {}
    for domain, config in SKILL_DOMAINS.items():
        score = 0
        matched_terms = []
        
        # Check primary keywords
        for keyword in config['keywords']:
            if keyword in query_lower:
                score += config['weight']
                matched_terms.append(keyword)
        
        # Check synonyms (lower weight)
        for synonym in config['synonyms']:
            if synonym in query_lower:
                score += config['weight'] * 0.7
                matched_terms.append(synonym)
        
        if score > 0:
            detected_skills[domain] = {
                'score': score,
                'terms': matched_terms,
                'weight': config['weight'],
                'assessment_types': config['assessment_types']
            }
    
    # Extract experience level
    experience_level = None
    for level, indicators in EXPERIENCE_LEVELS.items():
        if any(indicator in query_lower for indicator in indicators):
            experience_level = level
            break
    
    # Extract duration preferences
    duration_match = re.search(r'(\d+)\s*(minute|min|hour|hr)', query_lower)
    preferred_duration = None
    if duration_match:
        time_value = int(duration_match.group(1))
        time_unit = duration_match.group(2)
        if 'min' in time_unit:
            preferred_duration = time_value
        else:  # hours
            preferred_duration = time_value * 60
    
    return {
        'skills': detected_skills,
        'experience_level': experience_level,
        'preferred_duration': preferred_duration,
        'query_length': len(query.split())
    }

def create_enhanced_query(original_query: str, context: Dict) -> str:
    """Create an enhanced query with relevant terms for better semantic search."""
    enhanced_parts = [original_query]
    
    # Add skill synonyms for better matching
    for domain, skill_info in context['skills'].items():
        if skill_info['score'] > 2.0:  # Only for high-confidence skills
            domain_config = SKILL_DOMAINS[domain]
            # Add top synonyms
            enhanced_parts.extend(domain_config['synonyms'][:2])
            # Add assessment type hints
            enhanced_parts.extend(domain_config['assessment_types'])
    
    return ' '.join(enhanced_parts)

def calculate_advanced_relevance_score(doc_metadata: Dict, context: Dict, semantic_score: float) -> float:
    """Calculate advanced relevance score combining multiple factors."""
    
    # Base semantic score (40% weight)
    total_score = semantic_score * 0.4
    
    # Get document content for analysis
    doc_content = (
        doc_metadata.get('name', '') + ' ' +
        doc_metadata.get('description', '') + ' ' +
        doc_metadata.get('test_type', '')
    ).lower()
    
    # Skill matching score (35% weight)
    skill_score = 0
    total_skill_weight = 0
    
    for domain, skill_info in context['skills'].items():
        domain_config = SKILL_DOMAINS[domain]
        
        # Check if assessment types match
        assessment_match = any(
            assess_type.lower() in doc_content 
            for assess_type in domain_config['assessment_types']
        )
        
        # Check if keywords are in document
        keyword_match = any(
            keyword in doc_content 
            for keyword in domain_config['keywords'] + domain_config['synonyms']
        )
        
        if assessment_match or keyword_match:
            weight_multiplier = 1.5 if assessment_match and keyword_match else 1.0
            skill_score += skill_info['score'] * weight_multiplier
        
        total_skill_weight += skill_info['score']
    
    if total_skill_weight > 0:
        normalized_skill_score = skill_score / total_skill_weight
        total_score += normalized_skill_score * 0.35
    
    # Duration preference matching (15% weight)
    if context['preferred_duration']:
        doc_duration = doc_metadata.get('duration', 0)
        try:
            doc_duration = int(doc_duration)
            duration_diff = abs(doc_duration - context['preferred_duration'])
            # Score higher for closer duration matches
            duration_score = max(0, 1 - (duration_diff / 60))  # Normalize by hour
            total_score += duration_score * 0.15
        except:
            pass
    
    # Experience level matching (10% weight)
    if context['experience_level']:
        level_indicators = EXPERIENCE_LEVELS[context['experience_level']]
        level_match = any(indicator in doc_content for indicator in level_indicators)
        if level_match:
            total_score += 0.1
    
    return min(total_score, 1.0)  # Cap at 1.0

# --- 3. LOAD THE VECTOR DB ---
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

print("Initializing advanced retrieval system...")
base_retriever = vectordb.as_retriever(search_kwargs={"k": 20})  # Get more candidates
print("✅ Advanced retrieval system initialized.")

# --- 4. DEFINE THE FASTAPI APP & API MODELS ---
app = FastAPI(title="SHL Assessment Recommendation API - Advanced")
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
    Advanced recommendation system using keyword-semantic fusion.
    """
    print(f"Received query: {request.query[:70]}...")
    
    try:
        # 1. Extract context and skills from query
        context = extract_skills_and_context(request.query)
        print(f"Detected skills: {list(context['skills'].keys())}")
        print(f"Experience level: {context['experience_level']}")
        
        # 2. Create enhanced query for better semantic search
        enhanced_query = create_enhanced_query(request.query, context)
        print(f"Enhanced query created with {len(enhanced_query.split()) - len(request.query.split())} additional terms")
        
        # 3. Retrieve candidate documents
        retrieved_docs = base_retriever.invoke(enhanced_query)
        print(f"Retrieved {len(retrieved_docs)} candidate documents")
        
        # 4. Calculate advanced relevance scores
        scored_docs = []
        for doc in retrieved_docs:
            # Get base semantic similarity (this is implicit in retrieval order)
            base_semantic_score = 1.0 - (len(scored_docs) * 0.05)  # Decreasing score based on retrieval rank
            
            # Calculate advanced score
            advanced_score = calculate_advanced_relevance_score(
                doc.metadata, context, base_semantic_score
            )
            
            scored_docs.append((doc, advanced_score))
        
        # 5. Sort by advanced relevance score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        print(f"Top 3 scores: {[round(score, 3) for _, score in scored_docs[:3]]}")
        
        # 6. Build final recommendations
        recommendations = []
        processed_urls = set()
        
        for doc, score in scored_docs[:10]:  # Top 10
            meta = doc.metadata
            doc_url = meta.get("url", "")
            
            if doc_url not in processed_urls:
                test_type_list = [t.strip() for t in meta.get("test_type", "Unknown").split(',')]
                
                rec = Recommendation(
                    name=meta.get("name", "Unknown Name"),
                    url=doc_url,
                    adaptive_support=meta.get("adaptive_support", "No"),
                    description=meta.get("description", "No description available."),
                    duration=int(meta.get("duration", 0)),
                    remote_support=meta.get("remote_support", "Yes"),
                    test_type=test_type_list
                )
                recommendations.append(rec)
                processed_urls.add(doc_url)
                
                if len(recommendations) >= 10:
                    break
        
        print(f"Returning {len(recommendations)} advanced-ranked recommendations")
        return RecommendResponse(recommended_assessments=recommendations)

    except Exception as e:
        print(f"CRITICAL ERROR in advanced system: {e}")
        # Fallback to simple retrieval
        try:
            simple_docs = base_retriever.invoke(request.query)
            fallback_recommendations = []
            for doc in simple_docs[:10]:
                meta = doc.metadata
                test_type_list = [t.strip() for t in meta.get("test_type", "Unknown").split(',')]
                rec = Recommendation(
                    name=meta.get("name", "Unknown Name"),
                    url=meta.get("url", ""),
                    adaptive_support=meta.get("adaptive_support", "No"),
                    description=meta.get("description", "No description available."),
                    duration=int(meta.get("duration", 0)),
                    remote_support=meta.get("remote_support", "Yes"),
                    test_type=test_type_list
                )
                fallback_recommendations.append(rec)
            print(f"Fallback: returning {len(fallback_recommendations)} recommendations")
            return RecommendResponse(recommended_assessments=fallback_recommendations)
        except:
            return RecommendResponse(recommended_assessments=[])

# --- 6. RUN THE SERVER ---
if __name__ == "__main__":
    import uvicorn
    print("Starting ADVANCED keyword-semantic fusion API server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)