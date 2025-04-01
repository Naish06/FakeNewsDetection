# FakeNewsDetection
📌 Overview
This project presents a comprehensive approach to fake news detection using machine learning, deep learning, and ensemble models. It evaluates various models, including traditional ML classifiers, deep learning architectures, and hybrid combinations, to achieve high accuracy in classifying news as real or fake.

📂 Dataset
The datasets used in this project were sourced from Kaggle and FakeNewsNet. Preprocessing techniques applied include:

Text Normalization: Lowercasing, removing punctuation, special characters, and URLs.

Tokenization & Lemmatization: Converting text into tokens and reducing words to their base form.

Stopword Removal: Eliminating common words that do not contribute to classification.

🔍 Feature Engineering
Feature extraction techniques used for better representation of textual data:

TF-IDF (Term Frequency-Inverse Document Frequency)

Word Embeddings (Word2Vec, GloVe, BERT)

Sentiment Analysis to assess linguistic cues in fake vs. real news

🛠 Model Development
A variety of models were implemented and compared:

1️⃣ Machine Learning Models
Logistic Regression

Support Vector Machines (SVM)

Random Forest

Naïve Bayes

2️⃣ Deep Learning Models
LSTMs

CNNs

Transformer-based models (BERT, RoBERTa, XLNet)

3️⃣ Ensemble & Hybrid Approaches
Stacking and Boosting Methods

Combination of ML + Deep Learning for improved accuracy

📊 Performance Metrics
The models were evaluated using:
✅ Accuracy
✅ Precision
✅ Recall
✅ F1-score
✅ ROC-AUC Score

📈 Results & Discussion
The top 5 models based on accuracy were identified, and a hybrid model was created to improve performance further.

A comparative analysis of individual, ensemble, and deep learning models was performed to highlight the best-performing techniques.

Accuracy and ROC-AUC graphs were generated to visualize model performance.

📌 Key Findings & Contributions
✔️ A hybrid model combining the top-performing ML and deep learning models achieved the best accuracy.
✔️ The study demonstrates that transformer-based models (BERT, RoBERTa, XLNet) significantly improve fake news detection.
✔️ Ensemble techniques further enhance performance by leveraging the strengths of multiple models.

👨‍💻 Contributors
