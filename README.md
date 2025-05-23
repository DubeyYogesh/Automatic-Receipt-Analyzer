# 🧾 Automated Receipt Analyzer

## 📌 Project Overview

**Automated Receipt Analyzer** is a Python-based application that automates the extraction and categorization of data from retail receipts. It uses OCR and machine learning to convert receipt images into structured data, which can then be stored, analyzed, and visualized.

---

## 🧠 Features

- 📸 Upload receipt images via a web interface
- 🔍 Extract:
  - Store Name
  - City
  - Date of Purchase
  - Total Amount
  - List of Items with Categories
- 🧠 Categorize items using a KNN classifier with sentence embeddings
- 💾 Insert extracted data into an Oracle XE database
- 🌐 Streamlit web interface for user interaction

---

## ⚙️ Technology Stack

| Component            | Technology                          |
|---------------------|--------------------------------------|
| OCR                 | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| Item Embeddings     | SentenceTransformers (all-MiniLM-L6-v2) |
| Item Classification | Custom-trained KNN model             |
| Web Interface       | Streamlit                            |
| Database            | Oracle XE                            |
| Language            | Python 3.9+                          |

---

## 📥 Installation

### 🔧 Prerequisites
- Python 3.9+
- Oracle XE Database
- Oracle Client for Python (e.g., `oracledb`)

### 🔌 Clone the Repository
```bash
git clone https://github.com/DubeyYogesh/Automatic-Receipt-Analyzer.git
cd Automatic-Receipt-Analyzer 
```
---

📦 Install Dependencies
It's recommended to use a virtual environment:
```bash
- python -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

Now install the required packages:

```bash
pip install -r requirements.txt
```

---

If requirements.txt isn't present, use:

```bash
pip install paddleocr==2.6.1.3
pip install streamlit
pip install sentence-transformers
pip install opencv-python
pip install matplotlib
pip install oracledb
pip install transformers
```

---

🚀 Usage
1. Start the Streamlit App
```bash
streamlit run receipt_app.py
```

2. Web Interface
- Upload a receipt image (.jpg, .jpeg, .png)
- View extracted fields and categorized items
- Click "Insert to Database" to store results in Oracle

---

🗃️ Oracle DB Schema
You must have two tables created:
```SQL
CREATE TABLE receipts (
  id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  store_name VARCHAR2(255),
  city VARCHAR2(100),
  receipt_date DATE,
  total NUMBER
);

CREATE TABLE receipt_items (
  id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  receipt_id NUMBER REFERENCES receipts(id),
  item_name VARCHAR2(255),
  category VARCHAR2(100)
);
```

---

💡 Sample Output
```Json
{
  "store_name": "ABC Supermart",
  "city": "Bengaluru",
  "date": "03/05/24",
  "total": "978.00",
  "items": [
    {"item": "Milk 2L", "category": "dairy"},
    {"item": "Shampoo 1L", "category": "household"}
  ]
}
```

---

🧪 Challenges Faced
- Improving OCR accuracy for rotated/blurry receipts
- Extracting structured data using regex logic
- Creating a generalizable KNN-based classifier

---

🚧 Limitations
- Relies on receipt formats being fairly standard
- OCR may merge multiple fields into one
- Limited generalisation for uncommon items

---

🛠 Future Enhancements
- Image preprocessing (de-skewing, denoising)
- Use of Named Entity Recognition (NER) for smarter parsing
- Template-based layout detection
- Replace KNN with advanced ML models (SVM, Random Forest, Transformers)

---

📈 Use Cases
- Expense management tools
- Automated expense logging
- Retail analytics and purchase behaviour tracking

---

📬 Feedback & Contributions
Have feedback or want to contribute?
Feel free to open an issue, fork the repo, or suggest improvements via pull requests!

⭐ If you found this project helpful, give it a star!
```Yaml

Let me know if you’d like me to generate a `requirements.txt` file as well.
```
