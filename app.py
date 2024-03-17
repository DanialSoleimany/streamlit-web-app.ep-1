import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import requests
from io import StringIO

st.write("""
# Iris Flower Web Application
This is a **Web Application** to Predict **Species of Iris Flower**
""")

image_path = "iris.png"
image = Image.open(image_path)

with st.expander("Iris Species Image"):
    st.image(image)

# Kaggle dataset URL
url = "https://www.kaggle.com/uciml/iris/download"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Read the content of the response into a DataFrame
    data = pd.read_csv(StringIO(response.text))
    df = pd.DataFrame(data)
    
    with st.expander("Dataset"):
        st.write(df)
else:
    st.error("Failed to fetch data from Kaggle.")

def user_inputs():
    st.sidebar.subheader("Features of Iris Flower:")
    sepal_length = st.sidebar.slider("Sepal Length (Cm)", df["sepal_length"].min(), df["sepal_length"].max())
    sepal_width = st.sidebar.slider("Sepal Width (Cm)", df["sepal_width"].min(), df["sepal_width"].max())
    petal_length = st.sidebar.slider("Petal Length (Cm)", df["petal_length"].min(), df["petal_length"].max())
    petal_width = st.sidebar.slider("Petal Width (Cm)", df["petal_width"].min(), df["petal_width"].max())

    features = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    features_df = pd.DataFrame(features, index=[0])

    return features_df

features_df = user_inputs()
with st.expander("Iris Features"):
    st.write(features_df)

X = df.drop(["species", "Id"], axis=1)
y = df["species"]

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
