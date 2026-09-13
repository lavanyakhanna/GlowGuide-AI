# =========================================================
# GLOWGUIDE AI
# Personalized Skincare Concern Predictor
# =========================================================

# Step 1: Import Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# =========================================================
# Step 2: Create Sample Dataset
# =========================================================

data = {

    'water_intake': [2, 1, 3, 1.5, 2.5, 1, 3, 2, 1.5, 2.5,
                     3, 1, 2, 2.5, 1.5, 3, 2, 1, 2.5, 3],

    'sleep_hours': [8, 5, 7, 5, 8, 4, 7, 8, 6, 7,
                    8, 5, 7, 8, 6, 9, 7, 5, 8, 7],

    'stress_level': [2, 8, 3, 7, 2, 9, 4, 2, 6, 3,
                     1, 8, 4, 2, 7, 1, 3, 9, 2, 4],

    'screen_time': [3, 8, 2, 7, 4, 9, 3, 2, 7, 4,
                    2, 8, 4, 3, 7, 2, 4, 9, 3, 5],

    'sun_exposure': [2, 7, 3, 8, 2, 9, 4, 2, 7, 3,
                     2, 8, 3, 2, 7, 1, 4, 9, 3, 5],

    'pollution_exposure': [2, 7, 3, 8, 2, 9, 4, 2, 7, 3,
                           2, 8, 3, 2, 7, 1, 4, 9, 3, 5],

    # 0 = Hydration
    # 1 = Oil Control
    # 2 = Sun Protection
    # 3 = Stress/Sleep focused care

    'skin_concern': [
        0, 1, 0, 2, 0,
        3, 0, 0, 1, 0,
        0, 1, 0, 0, 2,
        0, 0, 3, 0, 1
    ]
}


df = pd.DataFrame(data)

print("Dataset:")
print(df.head())


# =========================================================
# Step 3: Feature Engineering
# =========================================================

# Create a simple "lifestyle score"

df['lifestyle_score'] = (
    df['water_intake'] * 2
    + df['sleep_hours']
    - df['stress_level'] * 0.5
)


# =========================================================
# Step 4: Select Features and Target
# =========================================================

X = df[
    [
        'water_intake',
        'sleep_hours',
        'stress_level',
        'screen_time',
        'sun_exposure',
        'pollution_exposure',
        'lifestyle_score'
    ]
]

y = df['skin_concern']


# =========================================================
# Step 5: Split Data
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# Step 6: Scale Features
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =========================================================
# Step 7: Logistic Regression
# =========================================================

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_prediction = logistic_model.predict(
    X_test_scaled
)


# =========================================================
# Step 8: Random Forest
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_prediction = rf_model.predict(
    X_test
)


# =========================================================
# Step 9: Compare Model Accuracy
# =========================================================

logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)

rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    "Logistic Regression Accuracy:",
    logistic_accuracy
)

print(
    "Random Forest Accuracy:",
    rf_accuracy
)


# =========================================================
# Step 10: Confusion Matrix
# =========================================================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(
    confusion_matrix(
        y_test,
        rf_prediction
    )
)


# =========================================================
# Step 11: Classification Report
# =========================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        rf_prediction,
        zero_division=0
    )
)


# =========================================================
# Step 12: Predict for a New User
# =========================================================

new_user = pd.DataFrame([{

    'water_intake': 1.5,

    'sleep_hours': 5,

    'stress_level': 8,

    'screen_time': 8,

    'sun_exposure': 7,

    'pollution_exposure': 6,

    'lifestyle_score':
        1.5 * 2 + 5 - 8 * 0.5

}])


# Make prediction

prediction = rf_model.predict(
    new_user
)[0]


# =========================================================
# Step 13: Convert Prediction into Recommendation
# ========================================================
recommendations = {

    0: "💧 Hydration-focused routine",

    1: "🌿 Oil-control focused routine",

    2: "☀️ Sun-protection focused routine",

    3: "😴 Stress and sleep focused skincare routine"
}


print("\n==============================")
print("GLOWGUIDE RESULT")
print("==============================")

print(
    "Predicted Skin Concern Category:",
    prediction
)

print(
    "Recommended Routine:",
    recommendations[prediction]
)