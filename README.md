# IDB Governance & Compliance Auditor

Machine Learning + LLM system for multi-label classification, rule analysis, and risk assessment of IDB project documents.

## Installation

Updated for Python 3.11+

1. Clone the project
git clone <your_repo_url>
cd project_root

2. Create a local environment
python -m venv .venv

3. Activate the environment

Windows

.venv\Scripts\activate


Mac/Linux

source .venv/bin/activate

4. Install required packages
pip install -r requirements.txt

## Running the Project

All training and testing scripts are inside the src/ folder.

1. Run the Sentence Encoder Baseline

MiniLM embeddings → Logistic Regression

python src/train_sentence_baseline.py


This generates:

models/sentence_baseline/
  ├── classifier.joblib
  ├── label_binarizer.joblib
  └── encoder/

2. Run the Transformer (BERT) Fine-tuning
python src/train_transformer_simple.py


Outputs:

models/transformer_finetuned/

3. Compare Models (Baseline vs Transformer)
python src/compare_models.py


This prints model predictions side-by-side.

4. Set Up OpenAI API Key

Windows PowerShell

setx OPENAI_API_KEY "sk-..."


Restart PowerShell → activate venv again → verify:

python -c "import os; print(os.getenv('OPENAI_API_KEY'))"

5. Run the LLM Baseline

This generates risk assessment + explanations.

python src/llm_baseline.py


Expected output:

{
  "labels": [...],
  "risk_level": "Medium",
  "explanation": "..."
}

## Project Structure
project_root/
│
├── src/
│   ├── train_sentence_baseline.py
│   ├── train_transformer_simple.py
│   ├── compare_models.py
│   └── llm_baseline.py
│
├── data/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
│
├── models/
├── requirements.txt
└── README.md

## Quick Start (Everything at Once)
git clone <repo>
cd project_root

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/train_sentence_baseline.py
python src/train_transformer_simple.py
python src/compare_models.py

setx OPENAI_API_KEY "sk-..."
python src/llm_baseline.py
