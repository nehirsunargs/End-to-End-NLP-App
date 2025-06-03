# News Headline Source Classifier

This project builds a **binary text classification model** that distinguishes between **Fox News** and **NBC** headlines using traditional machine learning methods. It covers the full ML pipeline: **data scraping, cleaning, feature engineering, model selection, evaluation, and deployment.**

## Summary

Starting with 3,805 URLs (2,010 Fox, 1,805 NBC), I scraped headline data using `BeautifulSoup`, preprocessed it through normalization and punctuation removal, and vectorized it using **TF-IDF**. Several models were tested (logistic regression, random forest, and Naive Bayes), with **multinomial Naive Bayes** achieving the best performance at **79.04% accuracy**. The final model and vectorizer are [hosted on Hugging Face](https://huggingface.co/nehirsunargs/nb-model) for inference and evaluation.

---

## Core Components

### Data Collection

* **Sources**: 3,805 URLs (public articles only).
* **Scraping Tools**: `requests`, `BeautifulSoup`.
* **Cleaning**: Lowercasing, removing special characters, filtering broken/missing headlines.
* **Ethics**: Scraping limited to public headlines; used for educational purposes only.

### Model Design

* **Baseline**: Logistic Regression with TF-IDF (100 features).
* **Final Model**: Naive Bayes with TF-IDF (5,000 features, unigrams + bigrams).
* **Other Models Tried**: Random Forest, tuned Logistic Regression.

### 📈 Performance

| Model                   | Accuracy   | Precision | Recall   | F1 Score |
| ----------------------- | ---------- | --------- | -------- | -------- |
| Logistic Regression     | 77.84%     | 0.78      | 0.78     | 0.78     |
| Random Forest           | 75.15%     | 0.75      | 0.75     | 0.75     |
| **Naive Bayes (Final)** | **79.04%** | **0.80**  | **0.79** | **0.79** |

🔗 **[Final Model on Hugging Face](https://huggingface.co/nehirsunargs/nb-model)**

---

## 🔍 Exploratory Questions

### 1. Does Model Choice Matter?

> **Yes.** Switching from baseline Logistic Regression to Naive Bayes + tuned TF-IDF significantly boosted accuracy.

### 2. Do Bigrams Help?

> **Yes.** Including bi-grams improved performance, especially for Fox News predictions.

### 3. Does the Model Generalize?

> The model performed consistently on a **20-headline hidden Sub-Testset**, showing good generalization to unseen samples.
