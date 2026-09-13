# =========================================================
# 🎀 GLOWGUIDE AI
# Interactive Lifestyle-Based Routine Recommender
# =========================================================

import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="GlowGuide AI",
    page_icon="🎀",
    layout="centered"
)


# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #FFF8FB;
}

h1 {
    text-align: center;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #FFF0F6;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.title("🎀 GlowGuide AI")
st.subheader("Your Lifestyle-Based Skincare Routine Guide ✨")

st.write(
    "Answer a few questions about your lifestyle and get a "
    "personalized skincare routine focus recommendation."
)

st.markdown("---")


# =========================================================
# CREATE TRAINING DATA
# =========================================================

data = {

    'water_intake': [
        2, 1, 3, 1.5, 2.5,
        1, 3, 2, 1.5, 2.5,
        3, 1, 2, 2.5, 1.5,
        3, 2, 1, 2.5, 3
    ],

    'sleep_hours': [
        8, 5, 7, 5, 8,
        4, 7, 8, 6, 7,
        8, 5, 7, 8, 6,
        9, 7, 5, 8, 7
    ],

    'stress_level': [
        2, 8, 3, 7, 2,
        9, 4, 2, 6, 3,
        1, 8, 4, 2, 7,
        1, 3, 9, 2, 4
    ],

    'screen_time': [
        3, 8, 2, 7, 4,
        9, 3, 2, 7, 4,
        2, 8, 4, 3, 7,
        2, 4, 9, 3, 5
    ],

    'sun_exposure': [
        2, 7, 3, 8, 2,
        9, 4, 2, 7, 3,
        2, 8, 3, 2, 7,
        1, 4, 9, 3, 5
    ],

    'pollution_exposure': [
        2, 7, 3, 8, 2,
        9, 4, 2, 7, 3,
        2, 8, 3, 2, 7,
        1, 4, 9, 3, 5
    ],

    'skin_concern': [
        0, 1, 0, 2, 0,
        3, 0, 0, 1, 0,
        0, 1, 0, 0, 2,
        0, 0, 3, 0, 1
    ]
}


df = pd.DataFrame(data)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["lifestyle_score"] = (
    df["water_intake"] * 2
    + df["sleep_hours"]
    - df["stress_level"] * 0.5
)


# =========================================================
# TRAIN THE MODEL
# =========================================================

features = [
    "water_intake",
    "sleep_hours",
    "stress_level",
    "screen_time",
    "sun_exposure",
    "pollution_exposure",
    "lifestyle_score"
]

X = df[features]

y = df["skin_concern"]


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# =========================================================
# USER INPUT SECTION
# =========================================================

st.header("🌸 Tell Us About Your Lifestyle")


water_intake = st.slider(
    "💧 Daily Water Intake (Litres)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5
)


sleep_hours = st.slider(
    "😴 Average Sleep (Hours)",
    min_value=3,
    max_value=12,
    value=7
)


stress_level = st.slider(
    "🧠 Stress Level",
    min_value=1,
    max_value=10,
    value=5
)


screen_time = st.slider(
    "📱 Daily Screen Time (Hours)",
    min_value=1,
    max_value=12,
    value=5
)


sun_exposure = st.slider(
    "☀️ Sun Exposure Level",
    min_value=1,
    max_value=10,
    value=5
)


pollution_exposure = st.slider(
    "🌫️ Pollution Exposure Level",
    min_value=1,
    max_value=10,
    value=5
)


# =========================================================
# CALCULATE LIFESTYLE SCORE
# =========================================================

lifestyle_score = (
    water_intake * 2
    + sleep_hours
    - stress_level * 0.5
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("---")


if st.button(
    "✨ Analyze My Routine",
    use_container_width=True
):

    new_user = pd.DataFrame([{

        "water_intake": water_intake,

        "sleep_hours": sleep_hours,

        "stress_level": stress_level,

        "screen_time": screen_time,

        "sun_exposure": sun_exposure,

        "pollution_exposure": pollution_exposure,

        "lifestyle_score": lifestyle_score

    }])


    prediction = model.predict(new_user)[0]


    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    recommendations = {

        0: {
            "title": "💧 Hydration Focus",
            "message": (
                "Your lifestyle pattern suggests that focusing "
                "on hydration and maintaining a consistent routine "
                "could be beneficial."
            )
        },

        1: {
            "title": "🌿 Balance & Oil-Control Focus",
            "message": (
                "Your lifestyle pattern suggests focusing on a "
                "balanced routine and gentle oil-control habits."
            )
        },

        2: {
            "title": "☀️ Sun-Care Focus",
            "message": (
                "Your lifestyle pattern suggests prioritizing "
                "sun protection and consistent daytime skincare habits."
            )
        },

        3: {
            "title": "😴 Lifestyle & Recovery Focus",
            "message": (
                "Your lifestyle pattern suggests focusing on rest, "
                "stress management, and maintaining a consistent routine."
            )
        }

    }


    result = recommendations[prediction]


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.markdown("## ✨ Your GlowGuide Result")

    st.markdown(
        f"""
        <div class="result-box">

        <h2>{result["title"]}</h2>

        <p>{result["message"]}</p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("### 📊 Your Lifestyle Score")

    st.progress(
        min(int(lifestyle_score * 5), 100)
    )

    st.write( f"Your calculated lifestyle score is: "
        f"**{lifestyle_score:.1f}**"
    )


    st.success("🎀 Analysis Complete!")


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("---")

st.caption(
    "⚠️ GlowGuide AI is an educational machine learning project "
    "and does not provide medical or dermatological diagnoses."
)