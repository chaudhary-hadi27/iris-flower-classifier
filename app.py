import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree, DecisionTreeClassifier
import matplotlib.pyplot as plt

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("iris_model.pkl")
    except FileNotFoundError:
        iris = load_iris()
        dtc = DecisionTreeClassifier(random_state=42)
        dtc.fit(iris.data, iris.target)
        return dtc

iris = load_iris()
model = load_model()
class_names = iris.target_names

flower_emoji = {"setosa": "🌼", "versicolor": "🌺", "virginica": "🌸"}
flower_desc = {
    "setosa": "Small & hardy. Usually found in Arctic regions. Easily identifiable by its small petals.",
    "versicolor": "Medium-sized. Common in eastern North America. Has mixed characteristics.",
    "virginica": "Large & striking. Found in eastern US wetlands. Known for its big, beautiful petals."
}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📖 About")
    st.info(
        "This app uses a **Decision Tree Classifier** "
        "to predict the species of an Iris flower.\n\n"
        "**3 Species:**\n"
        "- 🌼 Setosa\n"
        "- 🌺 Versicolor\n"
        "- 🌸 Virginica"
    )

    st.markdown("---")
    st.subheader("🌿 Feature Ranges")
    st.dataframe(
        {
            "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
            "Min (cm)": [4.3, 2.0, 1.0, 0.1],
            "Max (cm)": [7.9, 4.4, 6.9, 2.5],
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-learn 🌿")

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🌸 Iris Flower Classifier")
st.write("Adjust the sliders below and click **Predict** to identify the Iris species.")
st.divider()

# ─── Input Sliders ─────────────────────────────────────────────────────────────
st.subheader("📏 Flower Measurements (cm)")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length", 4.3, 7.9, 5.8, 0.1)
    petal_length = st.slider("Petal Length", 1.0, 6.9, 3.7, 0.1)

with col2:
    sepal_width  = st.slider("Sepal Width",  2.0, 4.4, 3.0, 0.1)
    petal_width  = st.slider("Petal Width",  0.1, 2.5, 1.2, 0.1)

st.divider()

# ─── Predict ──────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Species", type="primary", use_container_width=True):

    input_data    = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction    = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    species    = class_names[prediction]
    confidence = probabilities[prediction] * 100
    emoji      = flower_emoji[species]
    desc       = flower_desc[species]

    # ── Result ──
    st.success(f"{emoji}  **Predicted Species: Iris {species.capitalize()}**")
    st.write(desc)

    st.divider()

    # ── Confidence Metrics ──
    st.subheader("📊 Prediction Confidence")
    col1, col2, col3 = st.columns(3)

    for col, name, prob, emo in zip(
        [col1, col2, col3],
        class_names,
        probabilities,
        ["🌼", "🌺", "🌸"]
    ):
        with col:
            st.metric(
                label=f"{emo} {name.capitalize()}",
                value=f"{prob * 100:.1f}%"
            )

    # ── Progress Bars ──
    st.write("")
    for name, prob, emo in zip(class_names, probabilities, ["🌼", "🌺", "🌸"]):
        st.write(f"{emo} **{name.capitalize()}**")
        st.progress(float(prob))

    st.divider()

    # ── Decision Tree Visualization ──
    with st.expander("🌳 View Decision Tree Diagram"):
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            filled=True,
            rounded=True,
            fontsize=9,
            ax=ax
        )
        plt.title("Iris Decision Tree", fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Input Summary ──
    with st.expander("📋 View Your Input Summary"):
        st.dataframe(
            {
                "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
                "Value (cm)": [sepal_length, sepal_width, petal_length, petal_width],
            },
            hide_index=True,
            use_container_width=True
        )