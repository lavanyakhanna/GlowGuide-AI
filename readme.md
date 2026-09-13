# ✨ GlowGuide AI

### AI-Powered Personalized Skincare Recommendation System

GlowGuide AI is a machine learning-based web application that analyzes lifestyle factors and predicts the user's primary skincare concern. Based on the prediction, the system provides a personalized skincare focus and recommendation.

The project combines **Machine Learning, Data Analysis, and Web Development** to demonstrate how lifestyle data can be used to generate personalized recommendations.

---

## 🌸 Features

- Analyzes daily water intake
- Considers sleep patterns
- Evaluates stress levels
- Tracks screen time
- Considers sun exposure
- Analyzes pollution exposure
- Uses Machine Learning for prediction
- Generates personalized skincare recommendations
- Compares multiple Machine Learning models
- Interactive web interface using Streamlit

---

## 🧠 How It Works

The user provides information about their daily lifestyle, including:

- Water Intake
- Sleep Hours
- Stress Level
- Screen Time
- Sun Exposure
- Pollution Exposure

The system processes this information and calculates a **Lifestyle Score**.

The data is then analyzed using Machine Learning models to predict the user's primary skincare concern.

---

## 🤖 Machine Learning Models Used

### 1. Logistic Regression

Logistic Regression is used as a baseline classification model.

### 2. Random Forest Classifier

Random Forest is used to analyze multiple lifestyle factors and make the final skincare concern prediction.

The models are compared using accuracy metrics.

---

## 🎯 Skincare Concern Categories

The AI predicts one of the following categories:

| Category | Recommendation |
|----------|---------------|
| 💧 Hydration | Hydration-focused skincare routine |
| 🌿 Oil Control | Oil-control focused skincare routine |
| ☀️ Sun Protection | Sun-protection focused skincare routine |
| 😴 Stress & Sleep | Stress and sleep-focused skincare routine |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

---

## 📁 Project Structure

```text
GlowGuide-AI/
│
├── app.py
├── glowguide.py
├── requirements.txt
└── README.md

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/lavanyakhanna/GlowGuide-AI.git
```

### 2. Go to the project folder

```bash
cd GlowGuide-AI
```

### 3. Install the required libraries

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit web application

```bash
python -m streamlit run app.py
```

The application will open in your browser automatically.

---

## 🧠 Run the Machine Learning Model Separately

If you want to run the original machine learning program and view the model results:

```bash
python glowguide.py
```
