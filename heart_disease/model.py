import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

df = pd.read_csv('heart-disease.csv')

disease_scale = df['target'].value_counts()
sex = df['sex'].value_counts()

female_disease = 72 / 96
male_disease = 93 / 207
# print(f'Tỉ lệ mắc bệnh tim mạch:\nNam: {male_disease:.2f}\nNữ: {female_disease:.2f}')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'RandomForestClassifier': RandomForestClassifier()
}

def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_scores = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        model_scores[model_name] = score
        
    return model_scores

model_scores = fit_and_score(models, X_train, X_test, y_train, y_test)

model_compare = pd.DataFrame(model_scores, index=['accuracy'])

log_reg_grid = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear']
}

gs_log_reg = GridSearchCV(LogisticRegression(), 
                          log_reg_grid, 
                          cv=5, 
                          verbose=True)

gs_log_reg.fit(X_train, y_train)

best_params = gs_log_reg.best_params_
# print(f'Best parameters:\n{best_params}')

model = LogisticRegression(C=best_params['C'], solver=best_params['solver'])
model.fit(X_train, y_train)

def predict_Heart_Disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):    
    x = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                     columns=X.columns)
    return gs_log_reg.predict(x)[0]

prediction = predict_Heart_Disease(20, 1, 1, 100, 10, 0, 0, 250, 0, 3, 2, 1, 2)
print(f'Kết quả: {prediction}')
