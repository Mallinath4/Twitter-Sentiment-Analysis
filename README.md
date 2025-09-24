# Twitter Sentiment Analysis

**Repository:** Twitter Sentiment Prediction

## Project Overview

This project builds a multi-class text classification pipeline to predict the sentiment of tweets as **positive**, **negative**, or **neutral**. It demonstrates data ingestion, exploratory data analysis (EDA), text preprocessing, feature extraction (TF-IDF and Bag-of-Words), and deep neural network modeling using TensorFlow / Keras. Visualizations (wordclouds, countplots) and evaluation metrics are included to inspect model behaviour and spot overfitting.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Key Steps](#key-steps)
* [Modeling](#modeling)
* [Results](#results)
* [Visualizations](#visualizations)
* [How to run](#how-to-run)
* [File structure](#file-structure)
* [Dependencies](#dependencies)
* [Notes & Tips](#notes--tips)
* [Future improvements](#future-improvements)
* [License & Contact](#license--contact)

---

## Dataset

* Source: Kaggle — Tweet Sentiment Extraction competition dataset.
* File used: `twitter_dataset.csv` (expected columns: `textID`, `text`, `selected_text`, `sentiment`).
* The notebook drops `selected_text` (not available in test set) and uses `text` as input.
* After cleaning NaNs, dataset size used in the notebook: **27,480 rows**.

---

## Key Steps

1. **Read data**: load CSV and inspect (`df.head()`, `df.info()`, `df.shape`).
2. **EDA**: sentiment countplot, text length and word counts, wordclouds for positive/negative texts.
3. **Preprocessing**:

   * Decontract common English contractions.
   * Remove non-alphanumeric characters.
   * Remove stopwords (custom list; retains negations like `no`, `not`).
   * Lowercase and strip.
4. **Train/validation split**: 70% train / 30% CV with `train_test_split(random_state=50)`.
5. **Encoding labels**: `LabelEncoder` + `to_categorical` for multi-class output.
6. **Feature extraction**:

   * TF-IDF (`TfidfVectorizer`) — used for some experiments.
   * Count / Bag-of-Words (`CountVectorizer`) — converted to dense arrays and concatenated with scaled numeric features (`text_length`, `text_words`).
7. **Scaling**: `MinMaxScaler` for numeric columns.
8. **Model training**: Keras Sequential networks (several architectures tried). Early results show overfitting when trained for many epochs.

---

## Modeling (Notebook highlights)

* Two types of NN setups appear in the notebook:

  1. An attempt with `input_shape=(n_features,)` and a final `sigmoid` (this was inconsistent with categorical loss).
  2. Final model: three hidden layers with Dropout and `softmax` output (correct for 3-class classification) and `categorical_crossentropy` loss.
* Training example: 10 epochs produced high training accuracy (\~96%) but low validation accuracy (\~66%) and rising validation loss — clear overfitting.

---

## Results

* Training accuracy increased rapidly while validation accuracy plateaued near \~66–70%.
* Validation loss rose with more epochs (overfitting).
* Wordclouds helped visualize common words per sentiment.

**Metrics tracked:**

* Accuracy (primary)
* Precision, Recall (recommended to compute per-class — not shown in notebook but suggested)

---

## Visualizations

### Wordclouds

* Generated separate wordclouds for **positive** and **negative** tweets.
* Helped identify common patterns and frequently used terms across sentiments.

*Example:*
![Positive Wordcloud](https://github.com/Mallinath4/Twitter-Sentiment-Analysis/blob/main/images/postive.png?raw=true)
![Negative Wordcloud](https://github.com/Mallinath4/Twitter-Sentiment-Analysis/blob/main/images/negative.png?raw=true)

### ROC Curves

* Multi-class ROC curves were plotted (one vs all approach).
* Provided insight into class separability and model confidence.

*Example:*
![ROC Curve](https://github.com/Mallinath4/Twitter-Sentiment-Analysis/blob/main/images/roc.png?raw=true
)


## Dependencies

* Python 3.8+ (notebook uses Python 3.12 features in Colab but 3.8+ is fine)
* pandas, numpy, matplotlib, seaborn
* scikit-learn
* tensorflow (2.x)
* wordcloud, tqdm

---

##  Conclusions
1. It would be a good idea to use some tools such as wordcloud when we are doing Natural Language Processing (NLP) to ensure that we are getting the best results for predictions respectively. We would be able to understand the frequently occurring words from the less frequently occurring words by the size of the words that are plotted in the wordcloud respectively.
2. Steps should be taken to ensure that the model does not overfit or underfit. This ensures that the best predictions are being generated and therefore, we are going to get the best outputs respectively.
3. Standarizing the text and ensuring that the values lie between 0 and 1 would be good as this would allow the machine learning models to generate weights that are quite small rather than having different weight range values.
---

## Future improvements

* Use pretrained language models (BERT, DistilBERT) for better text representations — fine-tuning will likely boost accuracy significantly over TF-IDF + dense NN.
* Use word embeddings + LSTM / CNN architectures for sequence-aware modeling.
* Implement cross-validation, hyperparameter tuning (GridSearchCV / RandomizedSearchCV or Keras Tuner).
* Address class imbalance (if present) using class weights or resampling.
* Add saving/loading of trained model and inference script (`predict.py`) with an example CLI or REST API.

---



*Generated README for the Twitter Sentiment Analysis notebook. Feel free to ask if you want me to:*

* *Produce a `requirements.txt` file;*
* *Create a short `predict.py` CLI script; or*
* *Convert the notebook workflow into a clean Python package structure.*
