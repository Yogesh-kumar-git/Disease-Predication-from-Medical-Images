import base64
import streamlit as st
from streamlit_option_menu import option_menu # type: ignore
import io
import PIL.Image
from PIL import Image
import numpy as np
from tensorflow import keras
from keras.models import load_model  # type: ignore

with open('pic2.jpg', "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction WebApp',

                           ['HOME',
                            'Pneumonia Detection',
                            'Brain Tumor Detection'],

                           icons=['house','lungs','stethoscope',],
                           default_index=0)
if (selected == 'HOME'):
    st.title("Multiple Disease Prediction WebApp")
    st.write("")
    image = Image.open('ChatBot.jpg')
    new_image = image.resize((600, 500))
    st.image(new_image)
    st.title("**Welcome to our Disease Prediction Web App**")
    st.write("Harnessing the power of Deep learning and advanced algorithms, our web application allows you to predict and assess various diseases with ease. Input your information and receive accurate predictions for conditions such as pneumonia, and Brain Tumour.")
    st.write("Our cutting-edge models, trained on extensive datasets, provide reliable insights into your health. Whether you're concerned about your health, our app has you covered. Simply enter your details, and our intelligent algorithms will analyze the data to generate personalized predictions.")
    st.write("With our web application, you can take charge of your well-being, make informed decisions, and seek timely medical intervention when necessary. Experience the future of healthcare with our comprehensive disease prediction app and embark on a journey towards a healthier life.")
    st.write("Connect Me at:")
    st.write("LinkedIn Link : ")
    st.write("Email : yk8459711@gmail.com") 

# Pneumonia Detector Prediction Page
if (selected == 'Pneumonia Detection'):
    st.title("Pneumonia Detector")
    st.button("About", help="The Pneumonia Detector Predictor App is an innovative tool designed to assist in the detection of pneumonia based on chest X-ray images. By uploading the patient's chest X-ray, the app utilizes advanced image recognition algorithms to analyze the presence of abnormalities indicative of pneumonia. Dataset used : https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images")
    image = Image.open('Pneumonia-p.jpg')
    new_image = image.resize((800, 500))
    st.image(new_image)
    st.header("Load X-Ray Chest image")
    img = st.file_uploader(label="Click Browse Files and Upload", type=['jpeg', 'jpg', 'png'], key="xray")

    if img is not None:
        try:
            # Preprocessing Image
            i11 = Image.open(img).convert("RGB")
            p_img = i11.resize((128, 128))  # Resize to match the model input size (128x128)
            p_img = np.array(p_img) / 255.0  # Normalize the image to [0, 1]
            p_img = np.reshape(p_img, (1, 128, 128, 3)) 
            # Loading model
            MODEL = "vgg_unfrozen.h5"
            loading_msg = st.empty()
            loading_msg.text("Predicting...")
            model = keras.models.load_model(f"{MODEL}", compile=True)

            # Predicting result
            prob = model.predict(p_img)
            prob = prob.reshape(1, -1)[0]

            if prob[0] > 0.5:
                st.warning("Pneumonia Detected! :slightly_frowning_face")
            else:
                st.success("No Pneumonia Detected, Healthy! :smile")

            st.text(f"Probability of Pneumonia is {round(prob[0] * 100, 2)}%")
            loading_msg.text('')

        except Exception as e:
            st.error(f"Error: {e}")

# Brain Tumor Detection Page
if (selected == 'Brain Tumor Detection'):
    st.title("Brain Tumor Detector")
    st.button("About", help="The Brain Tumor Detection Predictor App is designed to assist in identifying the presence of brain tumors based on MRI images. By uploading MRI scans, the app uses deep learning models to analyze and detect abnormalities. This app aids healthcare professionals in making accurate and timely diagnoses. Dataset used: https://www.kaggle.com/datasets/praneet0327/brain-tumor-dataset")
    image = Image.open('brain-tumour.jpg')
    new_image = image.resize((800, 500))
    st.image(new_image)
    st.header("Load Brain MRI image")
    img = st.file_uploader(label="Click Browse Files and Upload", type=['jpeg', 'jpg', 'png'], key="brain_mri")

    if img is not None:
        # Preprocessing Image
        i11 = Image.open(img).convert("RGB")
        p_img = i11.resize((224, 224))  # Resize to match the model's input size (224x224)
        p_img = np.array(p_img) / 255.0  # Normalize the image to [0, 1]
        p_img = np.reshape(p_img, (1, 224, 224, 3))  # Reshape to match model input shape

        # Loading Brain Tumor Model
        MODEL = "updated_model.h5"  # Replace with your brain tumor model's filename
        loading_msg = st.empty()
        loading_msg.text("Predicting...")
        model = keras.models.load_model(f"{MODEL}", compile=True)

        # Predicting result
        prob = model.predict(p_img)  # Directly pass preprocessed image
        prob = prob.reshape(1, -1)[0]  # Flatten probability for easier processing

        if prob[0] > 0.5:
            st.warning("Brain Tumor Detected! :slightly_frowning_face")
        else:
            st.success("No Brain Tumor Detected, Healthy! :smile")

        st.text(f"Probability of Brain Tumor is {round(prob[0] * 100, 2)}%")
