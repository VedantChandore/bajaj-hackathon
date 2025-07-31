# Bajaj Insurance Claim Evaluation System

## 🚀 Project Overview
This is an AI-powered insurance claim evaluation system that:
1. **Parses** user queries to extract claim details
2. **Retrieves** relevant policy clauses using semantic search  
3. **Evaluates** claims against policy terms using AI decision engine

## 📋 Requirements
- Python 3.8+
- Cohere API key
- Virtual environment (recommended)

## 🔧 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd bajaj-hackathon
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows
   # source venv/bin/activate    # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file with:
   ```
   COHERE_API_KEY=your_cohere_api_key_here
   ```

## 🎯 Usage

### Test with Sample Data
```bash
python main_test.py
```

### Run Multiple Test Cases
```bash
python test_multiple.py
```

### Run with Real PDF (when available)
```bash
# Place your PDF file in data/ directory
python main.py
```

## 📁 Project Structure
```
bajaj-hackathon/
├── main.py              # Main application (requires PDF)
├── main_test.py          # Test version with sample data
├── test_multiple.py      # Multiple test scenarios
├── config.py            # Configuration settings
├── document_ingestion.py # PDF processing
├── semantic_retriever.py # Embedding & similarity search
├── policy_parser.py     # Query parsing with AI
├── decision_engine.py   # Claim evaluation logic
├── requirements.txt     # Python dependencies
├── .env                # Environment variables
├── data/               # PDF documents
├── output/             # Generated results
└── embeddings/         # Vector embeddings storage
```

## 🔍 How It Works

1. **Document Ingestion**: Loads and chunks PDF policy documents
2. **Embedding**: Creates vector embeddings of policy text using Cohere
3. **Query Parsing**: Extracts structured data from natural language queries
4. **Semantic Retrieval**: Finds most relevant policy clauses using cosine similarity
5. **Decision Engine**: Evaluates claims against retrieved policy terms using AI

## 📊 Example Results

**Input Query**: "35-year-old female, arthroscopic knee surgery, 8-month-old insurance policy"

**Output**:
```json
{
  "decision": "approved",
  "amount": 50000,
  "justification": "Policy covers arthroscopic procedures for members above 30 years. Policy duration of 8 months meets minimum 6-month requirement."
}
```

## 🧪 Test Cases Included

- ✅ **Approved**: Valid age, covered procedure, sufficient policy duration
- ❌ **Rejected**: Insufficient policy duration (< 6 months)
- ❌ **Rejected**: Excluded procedures (cosmetic surgery)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Cohere API**: For embeddings and language model
- **scikit-learn**: For cosine similarity calculations
- **PyMuPDF**: For PDF text extraction
- **FAISS**: For efficient vector similarity search
- **NumPy**: For numerical operations

## 📝 Configuration

Edit `config.py` to modify:
- API models (Cohere)
- PDF file paths
- Embedding models
- Other settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is created for the Bajaj Hackathon.
