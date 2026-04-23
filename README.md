# Expense Classification: AI/ML/NLP Pipeline

## ⚙️ Minimum Requirements

- **Python**: 3.8 or higher
- **Jupyter Notebook** or **JupyterLab**
- **pip** (Python package manager)

> Development verified on Python 3.12.1
---

## 🚀 Quick Setup & Installation

### Option 1: Using pip (Recommended for Quick Setup)

1. **Clone or download the repository:**
   ```bash
   cd /path/to/ct-ai-ml-nlp
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install --upgrade pip
   pip install jupyter pandas numpy matplotlib seaborn scikit-learn openpyxl python-dotenv sentence-transformers
   ```

### Option 2: Using requirements.txt

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
ct-ai-ml-nlp/
├── ct_ai_ml_nlp.ipynb       # Main Jupyter notebook
├── README.md                 # This file
├── EXECUTIVE_SUMMARY.md      # Project summary
└── content/
    ├── data.xlsx             # main dataset (XLSX format)
    ├── labelled_data.csv     # Labeled generated dataset (CSV format)
    └── se_labelled_data.csv  # Secondary generated dataset (CSV format)
```