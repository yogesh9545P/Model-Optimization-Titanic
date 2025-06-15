# Model-Optimization-Titanic
This project compares and tunes multiple ML models using GridSearchCV on the Titanic dataset.
# 🚢 Model Optimization - Titanic Dataset (Task 4)

This project is part of my Data Science Internship (Task 4), where I compare and optimize multiple machine learning models on the Titanic dataset to predict passenger survival. The goal is to identify the most accurate model using both baseline training and hyperparameter tuning.

---

## 📌 Objective

- Train baseline ML models:  
  `Logistic Regression`, `KNN`, `Decision Tree`, `Random Forest`, `SVM`
- Optimize the best model (`Random Forest`) using **GridSearchCV**
- Visualize model comparison using a **bar chart**

---

## 📁 Project Structure

| File                     | Description |
|--------------------------|-------------|
| `Titanic-Csv.csv`        | Cleaned Titanic dataset used for modeling |
| `model_training.py`      | Trains 5 ML models and compares accuracy |
| `model_optimization.py`  | Uses GridSearchCV to optimize Random Forest |
| `model_comparison.png`   | Output chart comparing model accuracies |
| `README.md`              | You’re reading it! |

---

## 🔍 Models Compared

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

---

## ⚙️ Optimized Parameters (Random Forest)

Example best result using GridSearchCV:

```python
{'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2,
 'min_samples_leaf': 1, 'bootstrap': True}
👨‍💻 Author
Yogesh Pacharne
🔗 LinkedIn
🔗 GitHub
