# Cardiovascular Disease Detection Using Machine Learning

A machine learning project exploring cardiovascular disease indicators through exploratory data analysis, classification models, hyperparameter tuning, and decision-threshold optimization. The project also includes a Gradio interface that allows users to interact with the trained model using their own inputs.

## Project Overview

Cardiovascular disease is a major public health concern. This project explores how machine learning classification models can identify patterns associated with a recorded history of heart disease using health and lifestyle indicators.

The project focuses on comparing multiple classification algorithms, addressing class imbalance, evaluating model performance using appropriate metrics, and improving the detection of positive cases by prioritizing recall.

**Note:** This is an educational machine learning project, not a clinically validated diagnostic system.

## Dataset

- **Source:** [Personal Key Indicators of Heart Disease — Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
- **Dataset identifier:** `kamilpytlak/personal-key-indicators-of-heart-disease`
- **Number of records:** 319,795
- **Number of columns:** 18
- **Missing values:** The provided dataset summary reports no missing values in the listed columns.
- **Target variable:** `HeartDisease`

The dataset contains health-related, demographic, and lifestyle indicators, including BMI, smoking, alcohol consumption, physical and mental health, sleep duration, physical activity, and existing health conditions.

The target variable indicates whether a respondent reported a history of coronary heart disease or myocardial infarction. The model therefore predicts the recorded target label, not future cardiovascular disease risk.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Exploratory analysis was performed to understand the dataset, examine feature distributions, and investigate relationships between the input features and the target variable.

The analysis included:
- Dataset inspection and descriptive statistics.
- Target class distribution analysis.
- Distribution plots, box plots, violin plots, and Q–Q plots for numerical features.
- Count plots for categorical features.

### 2. Data Preprocessing and Class Imbalance

Categorical features were encoded to prepare the data for machine learning models. The dataset exhibited substantial class imbalance, with positive heart disease labels representing a minority of the observations.

SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features) was applied to the training data to address class imbalance while accounting for categorical features.

### 3. Model Training and Comparison

The following classification models were evaluated:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- XGBoost
- Extra Trees Classifier
- Random Forest
- LightGBM

Models were compared using accuracy, precision, recall, F1-score, and confusion matrices. Accuracy alone was not considered sufficient because of the imbalanced target distribution.

### 4. Model Selection

Logistic Regression was selected for further experimentation because it achieved the highest recall among the initial model comparisons, at approximately **75.30%**.

Recall was prioritized because false negatives—positive cases incorrectly classified as negative—are particularly important to monitor in a cardiovascular disease detection project. Higher recall can help identify more positive cases, although it may also increase false positives.

### 5. Hyperparameter Tuning

RandomizedSearchCV was used to search for promising Logistic Regression hyperparameters through three-fold cross-validation.

The best configuration found was:

| Parameter | Selected value |
|---|---|
| `C` | `0.001009` |
| `class_weight` | `balanced` |
| `solver` | `lbfgs` |

This configuration achieved a cross-validation recall of approximately **80.15%**. On the test set, recall was approximately **75.01%**, with precision of **20.44%** and an F1-score of **32.13%**.

### 6. Decision-Threshold Tuning

The default classification threshold of 0.5 was investigated to explore the trade-off between recall, precision, F1-score, and accuracy.

A threshold of **0.4** was selected for the Gradio prediction interface based on the project's preference for recall while considering overall accuracy.

At this threshold, a prediction is classified as positive when the estimated probability for class 1 reaches or exceeds 0.4. The final performance depends on the data and model used.

## Try the Project Yourself

The notebook includes a Gradio interface that allows users to provide health and lifestyle inputs and receive a model prediction.

### Requirements

- Python 3.10 or a compatible Python version for the installed dependencies.
- Jupyter Notebook or JupyterLab.
- The Python packages imported in the notebook, including the relevant machine learning libraries and Gradio.

### Steps to Run

1. Clone or download this repository.
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease).
3. Place the required CSV file in the location expected by the notebook, or update the dataset path in the notebook.
4. Install the required dependencies. A `requirements.txt` file is provided, run:

   ```bash
   pip install -r requirements.txt
   ```

   Otherwise, install the packages imported by the notebook.

5. Open `Heart_Disease_Exploratory.ipynb` in Jupyter Notebook or JupyterLab.
6. Run the notebook cells in order, including the cells that preprocess the data, train or load the model, and launch the Gradio interface.
7. Open the local Gradio URL printed by the notebook.
8. Enter the requested inputs and submit them to see the prediction.

**Important:** The notebook must load the expected dataset and execute the relevant training and interface cells successfully. If the model is not saved separately, rerunning the notebook may be necessary to train it before launching the interface.

## Evaluation Metrics

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** The proportion of predicted positive cases that are actually positive.
- **Recall:** The proportion of actual positive cases correctly identified by the model.
- **F1-score:** The harmonic mean of precision and recall.
- **Confusion matrix:** A breakdown of true positives, true negatives, false positives, and false negatives.

Recall is emphasized because missing positive cases is a key concern in this project's evaluation objective. However, a high recall does not by itself establish clinical usefulness.

## Limitations

- The dataset contains reported health indicators and historical heart disease labels; it is not a clinical diagnostic dataset.
- SMOTENC and threshold tuning can change the balance between false negatives and false positives.
- Predictions depend on the selected features, data preprocessing, training procedure, and classification threshold.
- The model has not been established as suitable for medical diagnosis or clinical decision-making.

## Disclaimer

This project is intended for educational and research purposes only. Its predictions must not be used as a substitute for professional medical advice, diagnosis, or treatment.
