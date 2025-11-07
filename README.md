# SHL Assessment Recommendation System

A sophisticated AI-powered recommendation system that matches job requirements with appropriate SHL assessment tools using natural language processing and vector similarity search.

## 🏆 Performance Achievement

**Mean Recall@10: 0.5133**

Our system successfully achieved a strong performance score across diverse query types, demonstrating effective matching between job descriptions and assessment recommendations.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd SHL
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set API key**
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

5. **Start the API server**
```bash
python main.py
```

6. **Open web interface**
Open `index.html` in your browser or serve it locally:
```bash
python -m http.server 3000
# Then visit http://localhost:3000
```

## 📁 Project Structure

```
SHL/
├── main.py                     # FastAPI server with recommendation engine
├── build_database.py          # Vector database construction script
├── evaluate.py                # Evaluation script for Mean Recall@10
├── debug_urls.py              # URL matching debugging utility
├── index.html                 # Web frontend interface
├── predictions.csv            # Generated test set predictions
├── Technical_Report.md        # Comprehensive technical documentation
├── requirements.txt           # Python dependencies
├── chroma_db_shl/            # Vector database storage
│   ├── chroma.sqlite3
│   └── 34e024b7.../
└── data/                     # Training and test datasets
    ├── Train-Set.json
    └── Test-Set.json
```

## 🔧 Core Components

### 1. API Server (`main.py`)
- **Endpoint**: `POST /recommend`
- **Input**: Natural language job query
- **Output**: Top 10 ranked assessment recommendations
- **Features**: Rate-limit-safe, direct vector similarity search

### 2. Vector Database (`build_database.py`)
- **Products**: 54 SHL assessment tools
- **Embeddings**: OpenAI text-embedding-ada-002
- **Storage**: ChromaDB with persistent storage
- **Coverage**: Technical, behavioral, and executive assessments

### 3. Web Interface (`index.html`)
- Clean, responsive UI for query submission
- Real-time recommendation display
- Assessment details and links
- CORS-enabled API integration

### 4. Evaluation System (`evaluate.py`)
- Automated testing on labeled dataset
- Recall@10 metric calculation
- Test set prediction generation
- Performance analysis and reporting

## 📊 Performance Results

| Query Category | Recall@10 | Example |
|----------------|-----------|---------|
| Technical Roles | 0.60-0.80 | Java Developer, Data Analyst |
| Executive Positions | 0.83 | COO, Leadership roles |
| Creative Roles | 0.80 | Content Writer, Marketing |
| Sales/Graduate | 0.56 | Entry-level sales positions |
| Specialized Domains | 0.20-0.50 | Audio Producer, Banking |

**Overall Mean Recall@10: 0.5133**

## 🛠️ Technical Architecture

### Vector Similarity Pipeline
1. **Query Processing**: Natural language input preprocessing
2. **Embedding Generation**: OpenAI ada-002 embeddings via OpenRouter
3. **Similarity Search**: Cosine similarity against product database
4. **Ranking & Filtering**: Top-k retrieval with metadata extraction
5. **Response Generation**: Structured JSON with assessment details

### Rate Limit Mitigation
- Direct vector search (no LLM query generation)
- Embedding-only approach for consistent performance
- Graceful error handling with fallback responses

### Database Schema
Each assessment product includes:
- `name`: Product title
- `url`: SHL product page URL
- `description`: Detailed assessment description
- `duration`: Test duration in minutes
- `test_type`: Categories (Technical, Behavioral, etc.)
- `adaptive_support`: Adaptive testing capability
- `remote_support`: Remote proctoring availability

## 🚦 API Usage

### Request Format
```json
{
  "query": "I am hiring for Java developers who can also collaborate effectively with cross-functional teams."
}
```

### Response Format
```json
{
  "recommended_assessments": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "adaptive_support": "No",
      "description": "Java 8 assessment for experienced developers...",
      "duration": 60,
      "remote_support": "Yes",
      "test_type": ["Technical"]
    }
  ]
}
```

### Example cURL
```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"query": "Looking for Python developers with ML experience"}'
```

## 📈 Evaluation & Testing

### Run Full Evaluation
```bash
python evaluate.py
```

This will:
- Test all 10 labeled queries
- Calculate Mean Recall@10 score
- Generate `predictions.csv` for test set
- Display detailed performance metrics

### Debug URL Matching
```bash
python debug_urls.py
```

### Rebuild Database
```bash
python build_database.py
```

## 🏗️ Development

### Adding New Assessments
1. Update the product list in `build_database.py`
2. Run database rebuild: `python build_database.py`
3. Restart API server: `python main.py`

### Custom Evaluation
Modify `evaluate.py` to test with your own queries and ground truth data.

### Frontend Customization
Edit `index.html` to customize the web interface appearance and functionality.

## 📋 Requirements

See `requirements.txt` for full dependency list. Key packages:
- `fastapi`: Web API framework
- `langchain`: LLM and vector database integration
- `chromadb`: Vector database storage
- `openai`: Embedding generation
- `beautifulsoup4`: Web scraping utilities
- `requests`: HTTP client for API calls

## 🐛 Troubleshooting

### Rate Limit Errors
Our system uses direct vector search to avoid most rate limits. If you encounter issues:
- Verify your OpenRouter API key
- Check your usage limits at OpenRouter dashboard
- The system gracefully handles rate limits with empty responses

### Database Issues
```bash
# Rebuild the vector database
python build_database.py

# Check database contents
python -c "from langchain_community.vectorstores import Chroma; print(Chroma(persist_directory='chroma_db_shl')._collection.count())"
```

### API Connection Issues
- Ensure the API server is running: `python main.py`
- Check if port 8000 is available: `lsof -i :8000`
- Verify CORS settings for web interface access

## 📄 Documentation

- **Technical Report**: `Technical_Report.md` - Comprehensive system documentation
- **API Docs**: Visit `http://127.0.0.1:8000/docs` when server is running
- **Code Comments**: Detailed inline documentation throughout codebase

## 🎯 Key Features

- ✅ **High Performance**: 0.5133 Mean Recall@10 achievement
- ✅ **Rate Limit Safe**: Direct vector search approach
- ✅ **Comprehensive Coverage**: 54 SHL assessment products
- ✅ **User-Friendly**: Clean web interface
- ✅ **Production Ready**: FastAPI with proper error handling
- ✅ **Fully Documented**: Technical report and code documentation
- ✅ **Evaluation Ready**: Complete testing and submission pipeline

## 📊 Submission Files

- `predictions.csv`: Test set predictions for submission
- `Technical_Report.md`: 2-page technical documentation
- Complete codebase with web interface
- Evaluation results and performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:
- Review the Technical Report for detailed system information
- Check the troubleshooting section above
- Examine the evaluation results in `evaluate.py` output

---

**SHL Assessment Recommendation System - Achieving 0.5133 Mean Recall@10**