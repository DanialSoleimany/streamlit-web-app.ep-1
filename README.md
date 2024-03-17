# Streamlit Web App: Iris Flower Prediction

Welcome to the Iris Flower Prediction web application! This simple application predicts the species of an Iris flower based on user-provided features using a Random Forest Classifier.

## Features

- **Prediction of Iris Species**: Users can input the features of an Iris flower, such as Sepal Length, Sepal Width, Petal Length, and Petal Width, through sliders in the sidebar. The application then predicts the species of the Iris flower.

- **Visualization**: The web application displays the Iris species image, the dataset used for training the model, and the features inputted by the user for prediction.

## Usage

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/DanialSoleimany/streamlit-web-app.ep-1.git
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**:

    ```bash
    streamlit run app.py
    ```

4. **Interact with the Web Application**:
   
   - Open the provided URL in your web browser.
   - Use the sliders in the sidebar to input features of an Iris flower.
   - Explore the predictions and probabilities displayed in the expandable sections.

## Repository Structure

- **app.py**: Contains the main code for the Streamlit web application.
- **Iris.csv**: Dataset used for training the model.
- **iris.png**: Image displayed in the Iris Species Image section.
- **requirements.txt**: List of Python dependencies.

## Technologies Used

- **Streamlit**: For building the web application interface.
- **Pandas**: For data manipulation and handling the dataset.
- **scikit-learn**: For training the RandomForestClassifier model and preprocessing.
- **PIL (Python Imaging Library)**: For working with images.

## Contributors

- [Danial Soleimany](https://github.com/DanialSoleimany)

Feel free to contribute to this project by opening issues or submitting pull requests!

---
