# -----------------------------
# LOAD DATASET
# -----------------------------
import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/projects/project3/adult.csv")
df

# -----------------------------
# REMOVE UNUSED COLUMNS
# -----------------------------
df.drop(['fnlwgt'], axis=1, inplace=True)
df.drop(['education'], axis=1, inplace=True)

# -----------------------------
# HANDLE MISSING VALUES ('?')
# -----------------------------
import numpy as np
df['occupation'] = df['occupation'].replace('?', np.nan)
df['workclass'] = df['workclass'].replace('?', np.nan)

# Fill missing values with mode
df['occupation'] = df['occupation'].fillna(df['occupation'].mode()[0])
df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])

df.select_dtypes(include='object').columns

# -----------------------------
# LABEL ENCODING CATEGORICAL COLUMNS
# -----------------------------
from sklearn.preprocessing import LabelEncoder
cat_cols = ['workclass', 'marital-status', 'occupation',
            'relationship', 'race', 'gender', 'native-country', 'income']

encoder = [LabelEncoder() for i in cat_cols]

for i, col in enumerate(cat_cols):
    df[col] = encoder[i].fit_transform(df[col])
    print(encoder[i].classes_)

df

# -----------------------------
# SPLIT FEATURES AND TARGET
# -----------------------------
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# -----------------------------
# STANDARD SCALING
# -----------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=1, stratify=y)

# -----------------------------
# IMPORT MODELS
# -----------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# -----------------------------
# METRICS, WARNINGS, PLOTTING
# -----------------------------
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# MODEL TRAINING BEFORE SAMPLING (ORIGINAL DATA)
# -----------------------------
models = [
    KNeighborsClassifier(n_neighbors=5),
    GaussianNB(),
    DecisionTreeClassifier(class_weight='balanced', criterion='entropy', random_state=1),
    SVC(class_weight='balanced', random_state=1),
    RandomForestClassifier(class_weight='balanced', random_state=1),
    AdaBoostClassifier(random_state=1),
    GradientBoostingClassifier(random_state=1),
    XGBClassifier(random_state=1)
]

for model in models:
    print("=" * 70)
    print(model.__class__.__name__)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title(f"{model.__class__.__name__} - Confusion Matrix")
    plt.show()
    print("*" * 70)

# -----------------------------
# HANDLE IMBALANCE USING SMOTE-TOMEK
# -----------------------------
from imblearn.combine import SMOTETomek
smote = SMOTETomek(random_state=1)
X_os, y_os = smote.fit_resample(X_scaled, y)

# -----------------------------
# TRAIN-TEST SPLIT AFTER SAMPLING
# -----------------------------
X_train_os, X_test_os, y_train_os, y_test_os = train_test_split(
    X_os, y_os, test_size=0.3, random_state=1)

# -----------------------------
# MODEL TRAINING AFTER SAMPLING
# -----------------------------
models = [
    KNeighborsClassifier(n_neighbors=5),
    GaussianNB(),
    DecisionTreeClassifier(class_weight='balanced', criterion='entropy', random_state=1),
    SVC(class_weight='balanced', random_state=1),
    RandomForestClassifier(class_weight='balanced', random_state=1),
    AdaBoostClassifier(random_state=1),
    GradientBoostingClassifier(random_state=1),
    XGBClassifier(random_state=1)
]

for model in models:
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    model.fit(X_train_os, y_train_os)
    y_pred2 = model.predict(X_test_os)
    print(classification_report(y_test_os, y_pred2))
    ConfusionMatrixDisplay.from_predictions(y_test_os, y_pred2, cmap='Blues')
    plt.title(f"{model.__class__.__name__} - Confusion Matrix")
    plt.show()
    print("*" * 70)

# -----------------------------
# HYPERPARAMETER TUNING GRID SETUP
# -----------------------------
param_grids = {
    "KNeighborsClassifier": {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },

    "GaussianNB": {
        'var_smoothing': [1e-09, 1e-08, 1e-07]
    },

    "DecisionTreeClassifier": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5]
    },

    "SVC": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },

    "RandomForestClassifier": {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'max_features': ['sqrt']
    },

    "AdaBoostClassifier": {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1, 0.5]
    },

    "GradientBoostingClassifier": {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4]
    },

    "XGBClassifier": {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
}

# -----------------------------
# RUN GRIDSEARCH / RANDOMSEARCH
# -----------------------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

models = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "DecisionTreeClassifier": DecisionTreeClassifier(class_weight='balanced', random_state=1),
    "SVC": SVC(class_weight='balanced', random_state=1),
    "RandomForestClassifier": RandomForestClassifier(class_weight='balanced', random_state=1),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=1),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=1),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1)
}

for name, model in models.items():
    print("=" * 70)
    print(f"Model: {name}")

    if name in ["RandomForestClassifier", "AdaBoostClassifier", "GradientBoostingClassifier", "XGBClassifier"]:
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grids[name],
            n_iter=10,
            scoring='f1',
            cv=5,
            random_state=1)
    else:
        search = GridSearchCV(
            model,
            param_grid=param_grids[name],
            scoring='f1',
            cv=5)

    search.fit(X_train_os, y_train_os)
    y_pred = search.best_estimator_.predict(X_test_os)

    print("Best Params:", search.best_params_)
    print("CV F1 Score:", round(search.best_score_, 4))
    print("Test F1 Score:", round(f1_score(y_test_os, y_pred), 4))
    print(classification_report(y_test_os, y_pred))
    print("*" * 70)

# -----------------------------
# BASIC DATA INSPECTION
# -----------------------------
print("Shape of dataset:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isna().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

df.describe(include='all')

# -----------------------------
# EXPLORATORY DATA ANALYSIS (PLOTS)
# -----------------------------
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(x='income', data=df, palette='viridis')
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=30, kde=True, color='steelblue')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='income', y='educational-num', data=df, palette='Set2')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(y='workclass', hue='income', data=df, palette='coolwarm')
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='gender', hue='income', data=df, palette='Accent')
plt.show()

# -----------------------------
# RANDOM FOREST TUNING + RESULTS
# -----------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier(random_state=1, class_weight='balanced')

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'max_features': ['sqrt']
}

rf_search = RandomizedSearchCV(
    rf,
    param_distributions=param_grid,
    n_iter=5,
    scoring='f1',
    cv=5,
    random_state=1)

rf_search.fit(X_train_os, y_train_os)

best_rf = rf_search.best_estimator_
print("Best Parameters:", rf_search.best_params_)

y_pred_rf = best_rf.predict(X_test_os)
print(classification_report(y_test_os, y_pred_rf))

ConfusionMatrixDisplay.from_predictions(y_test_os, y_pred_rf, cmap='Blues')
plt.show()

# -----------------------------
# RANDOM FOREST WITHOUT TUNING
# -----------------------------
rf = RandomForestClassifier(class_weight='balanced', random_state=1)
rf.fit(X_train_os, y_train_os)
y_pred_rf = rf.predict(X_test_os)

print(classification_report(y_test_os, y_pred_rf))

ConfusionMatrixDisplay.from_predictions(y_test_os, y_pred_rf, cmap='Blues')
plt.show()

# -----------------------------
# SAVE MODEL, SCALER, LABEL ENCODER
# -----------------------------
import pickle

pickle.dump(rf, open('random_forest_model.sav', 'wb'))
pickle.dump(scaler, open('scaler.sav', 'wb'))

encoder_dict = {col: encoder[i] for i, col in enumerate(cat_cols)}
pickle.dump(encoder_dict, open('label_encoders.sav', 'wb'))

print("Files saved successfully!")
