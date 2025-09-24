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
![ROC Curve](<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/879aca3f-4368-4aa2-918b-da2b5de36656" />
)

---

## How to run

1. Create and activate a Python environment (recommended: conda or venv).
2. Install dependencies:

```bash
pip install -r requirements.txt
# or individually:
# pip install pandas numpy matplotlib seaborn scikit-learn tensorflow wordcloud tqdm
```

3. Place `twitter_dataset.csv` in the project root (or update the path in the notebook).
4. Open `Twitter Sentiment Analysis.ipynb` in Jupyter / Colab and run cells sequentially. Colab-specific cell magic (e.g., `!pip install`) will work in Colab.

---

## File structure (suggested)

```
/ (repo root)
├─ Twitter Sentiment Analysis.ipynb
├─ twitter_dataset.csv
├─ README.md
├─ requirements.txt
├─ notebooks/ (optional: extra experiments)
├─ models/ (saved trained models)
└─ assets/ (figures, wordcloud images, ROC curves)
```

---

## Dependencies

* Python 3.8+ (notebook uses Python 3.12 features in Colab but 3.8+ is fine)
* pandas, numpy, matplotlib, seaborn
* scikit-learn
* tensorflow (2.x)
* wordcloud, tqdm

---

## Notes & Tips

* **Label encoding / loss mismatch**: When doing multi-class classification, ensure final layer activation is `softmax` and use `categorical_crossentropy` with one-hot labels. If using integer labels, use `sparse_categorical_crossentropy`.
* **Sparse matrices**: Converting dense `toarray()` can be memory heavy for large vocabs. Consider using sparse-aware models or dimensionality reduction (e.g., TruncatedSVD) or using neural nets with embedding layers.
* **Overfitting mitigation**:

  * Reduce model capacity, add/dropout, use `EarlyStopping` callback based on `val_loss`.
  * Use regularization, tune epochs, or add more data/augmentation.
* **Evaluation**: Add confusion matrix, per-class precision/recall/F1, and classification report.

---

## Future improvements

* Use pretrained language models (BERT, DistilBERT) for better text representations — fine-tuning will likely boost accuracy significantly over TF-IDF + dense NN.
* Use word embeddings + LSTM / CNN architectures for sequence-aware modeling.
* Implement cross-validation, hyperparameter tuning (GridSearchCV / RandomizedSearchCV or Keras Tuner).
* Address class imbalance (if present) using class weights or resampling.
* Add saving/loading of trained model and inference script (`predict.py`) with an example CLI or REST API.

---

## License & Contact

* License: MIT (or choose your preferred license).
* Author / Contact: \[Your Name] — add email / LinkedIn if desired.

---

*Generated README for the Twitter Sentiment Analysis notebook. Feel free to ask if you want me to:*

* *Produce a `requirements.txt` file;*
* *Create a short `predict.py` CLI script; or*
* *Convert the notebook workflow into a clean Python package structure.*
