import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Define the prediction function
def model_prediction(test_image):
    model = tf.keras.models.load_model("Models/trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Agri.ai")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Home Page
if app_mode == "HOME":
    # Display banner image
    banner_image = Image.open("Diseases.png")  # Replace with your image path
    st.image(banner_image, use_column_width=True)

    # Title and description
    st.markdown(
        """
        <h1 style='text-align: center; color: green;'>Agri.ai: Smart Disease Detection</h1>
        <p style='text-align: center; font-size: 18px;'>
            Empowering Farmers with AI-Powered Plant Disease Recognition.<br>
            Upload plant images to detect diseases accurately and access actionable insights.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Features Section
    st.markdown(
        """
        <h2 style='text-align: center; color: green;'>Features</h2>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        feature1_image = Image.open("pic1.png")  # Replace with your image path
        st.image(feature1_image, use_column_width=True)
        st.markdown(
            "<p style='text-align: center;'><b>Disease Detection</b><br>Identify plant diseases with AI.</p>",
            unsafe_allow_html=True,
        )

    with col2:
        feature2_image = Image.open("pic2.png")  # Replace with your image path
        st.image(feature2_image, use_column_width=True)
        st.markdown(
            "<p style='text-align: center;'><b>Actionable Insights</b><br>Get disease details and remedies.</p>",
            unsafe_allow_html=True,
        )

    with col3:
        feature3_image = Image.open("pic3.png")  # Replace with your image path
        st.image(feature3_image, use_column_width=True)
        st.markdown(
            "<p style='text-align: center;'><b>Real-Time Results</b><br>Receive instant predictions.</p>",
            unsafe_allow_html=True,
        )

    # How It Works Section
    st.markdown(
        """
        <h2 style='text-align: center; color: green;'>How It Works</h2>
        <ol style='text-align: center; font-size: 18px;'>
            <li>Navigate to the "Disease Recognition" page.</li>
            <li>Upload an image of the affected plant.</li>
            <li>Get instant results along with disease information.</li>
        </ol>
        """,
        unsafe_allow_html=True,
    )

# Disease Recognition Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")

    # File uploader for test image
    test_image = st.file_uploader("Choose an Image:", type=["png", "jpg", "jpeg"])

    # Show uploaded image
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Predict"):
            st.snow()  # Display animation
            st.write("Analyzing the Image... Please wait.")
            
            try:
                # Call the prediction function
                result_index = model_prediction(test_image)

                # Class names for prediction
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]

                # Display the prediction result
                st.success(f"Model Prediction: {class_name[result_index]}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
