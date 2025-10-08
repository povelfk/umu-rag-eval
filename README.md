# RAG System Evaluation Framework

Evaluate your AI search system using synthetic test questions. This framework helps you understand **what's working** and **what needs improvement** in your RAG (Retrieval-Augmented Generation) system.

**This repository addresses the challenge of no human-labeled data through controlled synthetic data generation.**

## 🎯 Why Evaluate RAG Systems?

When your AI assistant gives wrong answers, where's the problem?

```
User Question → [Search] → [AI Response] → Answer
                   ↓            ↓
              Problem here? Or here?
```

This framework tests both parts separately:

1. **Search Evaluation** → Are we finding the right documents?
2. **AI Evaluation** → Is the AI using those documents correctly?

By testing each part independently, you can pinpoint exactly where improvements are needed.

> **Want to learn more?** Read the detailed [Evaluation Approach Guide](docs/evaluation-approach.md) for theory, methodology, and best practices.

## 📚 The 5-Step Process

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Synthetic Data Generation (k-NN approach)                │
│    • Generate questions from document chunks                │
│    • Find k=5 similar chunks using cosine similarity        │
│    • Create grounded & not-grounded question pairs          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Quality Assessment                                       │
│    • Validate length distributions                          │
│    • Check category balance                                 │
│    • Statistical validation                                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────┬──────────────────────────────────┐
│ 3A. Retrieval Eval       │ 3B. Generation Eval              │
│    • Test search system  │    • Test LLM with fixed context │
│    • Hit Rate @ k        │    • Groundedness scoring        │
│    • MRR metrics         │    • Both grounded/not-grounded  │
└──────────────────────────┴──────────────────────────────────┘
```

| Step | Notebook | What It Does |
|------|----------|--------------|
| **1** | `01_generate_data.ipynb` | Creates test questions from your documents |
| **2** | `02_evaluate_synthetic_data.ipynb` | Checks if test questions are good quality |
| **3** | `03_evaluate_ai_search.ipynb` | Tests if search finds correct documents |
| **4** | `04_run_ai_system.ipynb` | Generates AI answers using those documents |
| **5** | `05_evaluate_llm.ipynb` | Scores how well the AI answered |

## 🚀 Quick Start

### 1. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\venv\Scripts\Activate.ps1

# If you get an error about execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Azure Credentials

Copy the sample environment file and fill in your details:

Edit `.env` with your Azure credentials:

```bash
# Azure OpenAI
AOAI_ENDPOINT=https://your-resource.openai.azure.com/
AOAI_KEY=your-key-here
AOAI_API_VERSION=2025-04-01-preview

# Models (adjust to your deployed models)
chatModel=gpt-4o
chatModelMini=gpt-4o-mini
embeddingModel=text-embedding-ada-002

# Azure AI Search
SEARCH_ENDPOINT=https://your-search.search.windows.net
SEARCH_KEY=your-search-key
SEARCH_INDEX_NAME=your-index-name
```

### 4. Run the Notebooks

Open notebooks in VS Code or Jupyter and run them **in order** (01 → 02 → 03 → 04 → 05).

## Understanding Your Results

### Search Quality (Notebook 03)

**Hit Rate @ 5** tells you: "How often do we find the right document in the top 5 results?"
- **> 85%** = Good! Your search is working well ✅
- **< 70%** = Problem! You're missing too many documents ❌

**MRR (Mean Reciprocal Rank)** tells you: "How high do we rank the correct document?"
- **> 0.80** = Excellent! Right docs appear near the top ✅
- **< 0.60** = Needs work! Right docs buried too deep ❌

### AI Response Quality (Notebook 05)

**Groundedness Score** (1-5 scale) tells you if the AI is using documents correctly:

- **Grounded Questions** (answer IS in the documents)
  - **High score** = AI found and used the information correctly ✅
  - **Low score** = AI missed information or made up facts ❌

- **Not-Grounded Questions** (answer NOT in the documents)
  - **High score** = AI correctly said "I don't know" ✅
  - **Low score** = AI made up an answer (hallucination!) ❌

## How It Works

### Two Types of Test Questions

**Grounded Questions** → Answer exists in your documents
> "What are the safety features of Product X?"

**Not-Grounded Questions** → Answer is NOT in your documents  
> "What's the price of Product X in Canada?" (when only US prices are available)

This tests if your AI knows when to answer vs. when to say "I don't have that information."

### Question Generation

Each test question is generated with context from **6 similar chunks** (main + top 5 similar chunks), making the tests realistic and challenging

## 📁 Project Structure

```
01_rag_eval/
├── 01-05_*.ipynb          # Main evaluation notebooks (run in order)
├── data/                  # Generated test data
├── configs/
│   ├── prompts/           # System instructions for AI
│   └── settings/          # Test question parameters
└── utils/                 # Helper functions for search & generation
```

## 🎛️ Customization

Adjust test question generation in `configs/settings/`:
- `domains.jsonl` - Question domains (e.g., "technical", "customer_service")
- `difficulties.jsonl` - Easy, medium, hard questions
- `tones.jsonl` - Formal vs. casual questions
- `topics.jsonl` - Subject areas to cover
- `languages.jsonl` - Multi-language support

## 💡 What To Do With Results

**If Search is poor (low Hit Rate):**
- Improve document chunking strategy
- Add more metadata to documents
- Tune search parameters (hybrid search weights)

**If AI responses are poor (low Groundedness):**
- Improve system prompt instructions
- Adjust how context is presented to the AI
- Consider using a more capable model

**If both are good but users still complain:**
- Generate more diverse test questions
- Check for edge cases not covered in tests
- Review actual user query patterns

## 📖 Key Terms

- **RAG** = Retrieval-Augmented Generation (search + AI answer)
- **Hit Rate** = How often we find the right document
- **MRR** = How highly we rank the right document
- **Groundedness** = How well AI sticks to provided documents
- **Synthetic Data** = Computer-generated test questions

---

**Need Help?** Each notebook contains detailed explanations. Start with notebook 01 and read the markdown sections.
