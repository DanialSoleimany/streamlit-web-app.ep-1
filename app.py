import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from sklearn.preprocessing import LabelEncoder

st.write("""
# Iris Flower Web Application
This is a **Web Application** to Predict **Species of Iris Flower**
""")

image_path = "iris.png"
image = Image.open(image_path)

with st.expander("Iris Species Image"):
    st.image(image)

data = pd.read_csv("Iris.csv")
df = pd.DataFrame(data)

with st.expander("Dataset"):
    st.write(df)

def user_inputs():
    st.sidebar.subheader("Features of Iris Flower:")
    sepal_length = st.sidebar.slider("Sepal Length (Cm)", df["SepalLengthCm"].min(), df["SepalLengthCm"].max())
    sepal_width = st.sidebar.slider("Sepal Width (Cm)", df["SepalWidthCm"].min(), df["SepalWidthCm"].max())
    petal_length = st.sidebar.slider("Petal Length (Cm)", df["PetalLengthCm"].min(), df["PetalLengthCm"].max())
    petal_width = st.sidebar.slider("Petal Width (Cm)", df["PetalWidthCm"].min(), df["PetalWidthCm"].max())

    features = {
        "SepalLengthCm": sepal_length,
        "SepalWidthCm": sepal_width,
        "PetalLengthCm": petal_length,
        "PetalWidthCm": petal_width
    }

    features_df = pd.DataFrame(features, index=[0])

    return features_df

features_df = user_inputs()
with st.expander("Iris Features"):
    st.write(features_df)

X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

clf = RandomForestClassifier()
clf.fit(X, y_encoded)

y_pred_proba = clf.predict_proba(features_df)
y_pred = clf.predict(features_df)

with st.expander("Class Probabilities"):
    species_names = label_encoder.classes_
    for i, species in enumerate(species_names):
        st.write(f"{species}: {y_pred_proba[0][i]}")

with st.expander("Predicted Class"):
    st.write(label_encoder.inverse_transform(y_pred))
