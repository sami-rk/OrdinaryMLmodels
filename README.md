# 🤖 Machine Learning Classification Project

**Course:** Artificial Intelligence, Spring 1405 (Computer Assignment 4)  
**Instructor:** Dr. Fada'i  
**Authors:** Parsa Daqiq, Yasaman Amoo-Jafari, Mostafa Kermani-Nia

---

## Overview

A narrative-driven machine learning project covering four real-world classification problems, implemented both from scratch and using scikit-learn. Each part tells the story of university students coping with internet outages, burnout, mountain trekking, and music preferences.

---

## Project Structure

```
.
├── Data/
│   ├── daneshchat_logs.csv        # Part 1: Message routing logs
│   ├── student_burnout.csv        # Part 2: Student mental health data
│   ├── trekking_expedition.csv    # Part 3: Mountain route safety data
│   └── student_music_taste.csv    # Part 4: Music preference data
└── models.ipynb                   # Full implementation notebook
```

---

## Part 1: DaneshChat Messenger

**Problem:** Predict (1) whether a message was **lost in transit** (`is_lost`) and (2) whether a message is **spam** (`is_spam`).

**Dataset features:** `message_size_kb`, `transfer_steps`, `connection_quality`, `has_urgent_keyword`, `has_link`

### Models Implemented

#### Custom Decision Tree (from scratch, NumPy/Pandas only)

- `Node` class with feature, threshold, left/right children, and leaf value
- `DecisionTreeClassifierScratch` with:
    - Entropy-based information gain for best-split selection
    - Configurable `max_depth`, `min_samples_split`, `min_samples_leaf`
    - Two prediction modes: majority-class binary and probability-ratio with tunable threshold
- Used for **lost message prediction** (`is_lost`)

#### Naive Bayes (scikit-learn `GaussianNB`)

- Used for **spam detection** (`is_spam`)

#### Random Forest + GridSearchCV (scikit-learn)

- `RandomForestClassifier` with hyperparameter tuning over `max_depth ∈ {3, 5, 7, 10, 12, 15, 20, None}`
- 5-fold cross-validation, scoring on F1

### Results on Spam Detection

|Model|Accuracy|Precision|Recall|F1 Score|
|---|---|---|---|---|
|Custom Decision Tree (binary)|~0.68|~0.69|~0.47|~0.56|
|Custom Decision Tree (ratio + threshold)|competitive|—|—|—|
|Naive Bayes|competitive|**highest**|—|competitive|
|Random Forest (best depth via GridSearchCV)|competitive|—|—|competitive|

**Key finding:** Naive Bayes achieves the highest precision among all models, which matters most for spam detection (minimizing false positives that would incorrectly suppress legitimate messages).

### Analytical Conclusions

- The independence assumption between `has_urgent_keyword` and `has_link` is violated in practice — spam messages commonly contain both simultaneously. Despite this, Naive Bayes performs well.
- A Decision Tree with `max_depth=1` suffers from **underfitting (high bias)** — it cannot represent the true decision boundary.
- For lost-message detection, **Recall is more important than Precision**: a False Negative (predicting delivered when lost) means the sender never resends, causing critical information loss.

---

## Part 2: Student Mental Health Radar

**Problem:** Classify each student's mental state into three levels — Healthy (0), Tired (1), Burned-out (2), based on daily behavioral metrics.

**Dataset features:** `sleep_hours_avg`, `count_read_news`, `attempts_connection_vpn`, `hw_pending`, `hours_time_screen`

### Models Implemented

#### Custom Multinomial Logistic Regression (from scratch, NumPy only)

- Softmax activation for multi-class output
- Cross-entropy loss
- Gradient Descent weight updates
- Configurable `learning_rate` and number of epochs

#### sklearn `LogisticRegression`

- Used for comparison of training time and accuracy

### Results

| Metric         | Custom Softmax | sklearn LogisticRegression |
| -------------- | -------------- | -------------------------- |
| Accuracy       | reported       | reported (comparable)      |
| Macro F1-Score | reported       | —                          |

**Key finding:** Accuracy alone is misleading on this imbalanced dataset (fewer burned-out students than healthy ones). Macro F1-Score reveals true per-class performance, especially the model's difficulty with the minority "burned-out" class. The gap between Macro F1 and Accuracy demonstrates the imbalance effect.

### Analytical Conclusions

- **Feature scaling is critical for gradient descent:** Without scaling, the loss surface is elongated and gradient descent converges slowly or diverges. StandardScaler makes the surface more spherical, enabling faster convergence and numerical stability in the Softmax exponentials.
- **Weight matrix analysis:** Positive weights on `hw_pending` for the "burned-out" class confirm that accumulated unfinished homework significantly increases the probability of burnout.
- **Softmax vs. One-vs-Rest:** Softmax is more appropriate here because the three mental states are mutually exclusive. OvR trains independent binary classifiers and may be preferable when adding new classes incrementally or when classes are naturally binary-separable.

---

## Part 3: Payamnouria Mountain Route Safety

**Problem:** Binary classification, predict whether a trekking route is **safe (1)** or **dangerous (0)**.

**Dataset features:** `slope_angle`, `wolf_prob`, `rain_mm`, `cold_resistance` (categorical: Low/Medium/High)

### Feature Engineering

A composite engineered feature was added:

```
environmental_danger = slope_angle × rain_mm × (1 + wolf_prob)
```

Adding this feature improved both Accuracy and F1-Score by combining three correlated signals into a single highly informative feature that the Decision Tree selected as the top split at depths 1–2.

### Preprocessing Pipeline (Data-Leak-Free)

1. Train/Validation split **before** any preprocessing
2. `OneHotEncoder` fit on `cold_resistance` from **train only**, then transform both sets
3. `StandardScaler` fit on numeric columns from **train only**, then transform both sets

### Model

`DecisionTreeClassifier` from scikit-learn (`max_depth=4`, `min_samples_split=25`, `min_samples_leaf=12`)

### Results

|Experiment|Accuracy|F1 Score|
|---|---|---|
|Baseline (no feature engineering)|lower|lower|
|With `environmental_danger`|**higher**|**higher**|

### Analytical Conclusions

- **Data leakage:** Fitting the scaler on the full dataset before splitting leaks validation statistics (mean, variance) into training. The correct approach — fit on train only, transform both — prevents inflated validation performance.
- **Extracted decision rules from the tree:**
    1. `environmental_danger ≤ 0.004` → likely Safe
    2. `cold_resistance_High ≥ 0.5` → likely Safe
- **OneHotEncoder vs. OrdinalEncoder:** `cold_resistance` has a natural ordering (Low < Medium < High), so OrdinalEncoder is arguably more appropriate — it lets the tree make threshold splits (e.g., "≤ Medium") that respect the ordering, requiring fewer splits than OneHotEncoder's independent binary columns.

---

## Part 4: Mountain Symphony Music Genre Recommender

**Problem:** Multi-class classification, predict a student's preferred music genre (Rock, Pop, Classical, Traditional) from demographic and behavioral features.

**Dataset features:** `age`, `daily_music_hours`, `favorite_instrument` (categorical), `personality_trait` (categorical)

### Preprocessing

- `OneHotEncoder` on `favorite_instrument` and `personality_trait` (fit on train only)
- `StandardScaler` on `age` and `daily_music_hours` (fit on train only)

### Model

`KNeighborsClassifier` from scikit-learn, with k=5 as the baseline and a sweep over k ∈ {1, 3, 5, 7, 11, 21, 51}.

### Results — k=5

|Metric|Value|
|---|---|
|Accuracy|reported|
|Macro F1-Score|reported|

### Bias–Variance vs. k

|k|Behavior|
|---|---|
|k=1|Train Acc=1.0, high variance — **Overfitting**|
|k=5|Best validation accuracy — **optimal trade-off**|
|k=51|Both train/val accuracy drop — high bias, **Underfitting**|

Increasing k reduces variance (less sensitivity to individual training points) but increases bias (over-smoothing).

### Analytical Conclusions

- **Scaling is essential for KNN:** Euclidean distance is dominated by features with larger ranges. Without scaling, small-range features are effectively ignored. Decision Trees don't need scaling because splits compare values within a single feature (no cross-feature distance computation).
- **LabelEncoder is wrong for nominal categories in KNN:** Assigning Guitar=1, Piano=2, Violin=3 implies Guitar and Piano are "closer" than Guitar and None — a false ordinal relationship that distorts distance calculations.
- **KNN prediction is much slower than Decision Trees at inference:** KNN requires O(n×p) distance computations per query (comparing against all training samples). A Decision Tree requires only O(depth) ≈ O(log n) comparisons, making it orders of magnitude faster for large n.

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

## How to install:

```bash
pip install numpy pandas matplotlob scikit-learn
```
---

## Running the Notebook

```bash
jupyter notebook models.ipynb
```

Datasets should be placed in a `Data/` subdirectory relative to the notebook. All random seeds are fixed for reproducibility.