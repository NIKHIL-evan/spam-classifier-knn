# 📩 SMS Spam Classifier using TF-IDF and K-Nearest Neighbors (KNN)

This project builds a machine learning model to classify SMS messages as **Spam** or **Ham** using **TF-IDF vectorization** and the **K-Nearest Neighbors (KNN)** algorithm.

---

## 🚀 Project Overview
The purpose of this project is to learn text classification using real-world SMS messages.  
The model takes a text message as input and predicts whether it is spam or legitimate.

### Key Objectives:
- Understand end-to-end ML workflow
- Learn TF-IDF vectorization for text
- Handle imbalanced data with **RandomOverSampler**
- Improve model performance through tuning

---

## 🧠 Technologies & Libraries Used
- Python
- Pandas, NumPy
- Scikit-Learn
- imbalanced-learn (RandomOverSampler)
- TF-IDF Vectorizer
- KNN Classifier (cosine distance + distance weights)

---

## 🧹 Data Preprocessing & Cleaning
- Lowercasing text
- Removing digits and unnecessary punctuation
- Removing extra spaces
- TF-IDF vectorization with max_features=2000
- Label encoding (ham → 0, spam → 1)

---

## ⚙ Model Details
**Selected model: KNN with cosine distance**
KNeighborsClassifier(
n_neighbors=5,
metric='cosine',
weights='distance'
)

---

## 📊 Final Performance
| Metric | Value |
|--------|-------|
| Accuracy | **95.5%** |
| Spam Recall | **91%** |
| Spam Precision | 79% |
| F1-Score (spam) | 0.84 |

### Confusion Matrix
Predicted   Ham     Spam
Actual
Ham (0)     930     36
Spam (1)    14      135

## 🧪 Example Predictions
| Message | Prediction |
|----------|-----------|
| “Congratulations! You won a $1000 Walmart gift card.” | Spam |
| “Your OTP is 48291. Do not share with anyone.” | Ham |
| “Urgent: unusual login activity detected. Verify now.” | Spam |

---

## 📂 Project Structure
spam-classifier-knn/
│
├── spam_detector.ipynb
├── requirements.txt
└── README.md

---

## 🎯 Next Steps / Improvements
- Add comparison with Naive Bayes / Logistic Regression
- Deploy UI using Streamlit
- Add ROC-Curve visualization

---

## 👤 Author
**Nikhil**  
Engineering student (AI/ML)

---

## ⭐ Support
If you found this project useful, please ⭐ the repository!
