import streamlit as st
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import pipeline

# Custom CSS for a professional look
st.markdown("""
    <style>
    body { font-family: 'Arial', sans-serif; background-color: #121212; }
    .main { background-color: #1e1e1e; padding: 20px; border-radius: 10px; }
    .stButton>button { background-color: #444; color: white; border-radius: 5px; padding: 8px 16px; font-size: 16px; }
    .stFileUploader { border: 2px dashed #666; padding: 10px; border-radius: 5px; background-color: #222; color: white; }
    .menu-bar { background-color: #222; padding: 12px; border-radius: 10px; text-align: center; }
    .menu-button { background-color: #444; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 16px; }
    .menu-button:hover { background-color: #555; }
    h1, h2, h3 { color: #e0e0e0; }
    .treatment-box { background-color: #333; padding: 10px; border-left: 5px solid #4CAF50; margin-top: 10px; border-radius: 5px; color: white; }
    </style>
""", unsafe_allow_html=True)

# Load model efficiently
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/content/drive/MyDrive/trained_plant_disease_model.keras")

model = load_model()
#disease info
disease_info = {
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot causes dark spots on leaves and fruits, leading to poor growth.",
        "treatment": [
            "Use copper-based fungicides to control bacterial spread.",
            "Avoid overhead watering to minimize moisture on leaves.",
            "Remove infected leaves and plants to prevent further infection."
        ]
    },
    "Tomato_Early_blight": {
        "description": "Early blight leads to dark, concentric rings on leaves, weakening the plant.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or copper.",
            "Ensure proper crop rotation to prevent recurring infections.",
            "Remove infected leaves and avoid excessive nitrogen fertilizer."
        ]
    },
    "Tomato_Late_blight": {
        "description": "Late blight causes water-soaked lesions and can quickly kill plants.",
        "treatment": [
            "Use fungicides containing mancozeb or chlorothalonil.",
            "Destroy infected plants and avoid overhead irrigation.",
            "Plant resistant tomato varieties to prevent infections."
        ]
    },
    "Tomato_Leaf_Mold": {
        "description": "Leaf mold appears as yellow spots on leaves and fuzzy mold on the underside.",
        "treatment": [
            "Improve air circulation and reduce humidity around plants.",
            "Use fungicides containing copper or potassium bicarbonate.",
            "Remove and destroy infected leaves to control spread."
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Septoria leaf spot causes small, brown circular spots on leaves.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or copper-based sprays.",
            "Avoid watering leaves directly to reduce spread.",
            "Rotate crops and remove infected plant debris from soil."
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Spider mites cause yellow stippling on leaves and fine webbing on plants.",
        "treatment": [
            "Spray plants with neem oil or insecticidal soap.",
            "Encourage natural predators like ladybugs.",
            "Increase humidity and wash plants with water to remove mites."
        ]
    },
    "Tomato_Target_Spot": {
        "description": "Target spot causes dark, circular spots on leaves and stems.",
        "treatment": [
            "Apply fungicides like azoxystrobin or chlorothalonil.",
            "Avoid overhead irrigation and ensure good spacing between plants.",
            "Remove affected leaves and destroy infected debris."
        ]
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "This virus leads to curled, yellow leaves and stunted growth.",
        "treatment": [
            "Use insecticides to control whiteflies, which spread the virus.",
            "Remove infected plants immediately to prevent further spread.",
            "Grow virus-resistant tomato varieties for better protection."
        ]
    },
    "Tomato_mosaic_virus": {
        "description": "Mosaic virus causes mottled, wrinkled leaves and poor fruit development.",
        "treatment": [
            "Remove and destroy infected plants as there is no cure.",
            "Wash hands and tools thoroughly to avoid spreading the virus.",
            "Use resistant seed varieties to minimize future infections."
        ]
    },
    "Tomato_healthy": {
        "description": "Your tomato plant appears to be healthy! 🌱",
        "treatment": [
            "Continue regular watering and fertilization.",
            "Monitor for any early signs of disease.",
            "Keep plants well-spaced to ensure good air circulation."
        ]
    }
}


# Load chatbot model
@st.cache_resource
def load_chatbot():
    return pipeline("text-generation", model="facebook/blenderbot-400M-distill")

chatbot = load_chatbot()

# Sidebar dropdown for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a Page:", ["🏠 Home", "ℹ About", "🩺 Disease Recognition", "🤖 Chatbot", "🌐 Extra Services"])

# Home Page
if page == "🏠 Home":
    st.header("🌱 Welcome!")
    st.title("🍅 Tomato Leaf Disease Recognition System")
    st.image("/content/drive/MyDrive/home_page.jpg", use_container_width=True)
    st.markdown("""
    ## 🔍 How It Works:
    1. **Upload an image** of a tomato leaf.
    2. **Model analyzes it** and detects possible diseases.
    3. **View the result & suggested treatments**.
    
    👉 Click on **Disease Detection** to get started!
    """)

# About Page
elif page == "ℹ About":
    st.title("📜 About the Project")
    st.markdown("""
### 🌱 Tomato Disease Detection System 🍅  

This system is powered by **state-of-the-art Deep Learning technology**, trained to detect **10 different conditions affecting tomato plants**, including both **diseases and nutrient deficiencies**. It helps **farmers, researchers, and agricultural enthusiasts** take timely action to improve crop health and yield.  

### 🔬 How It Works  
✅ **Deep Learning Model:** Built using **Convolutional Neural Networks (CNNs)**, trained on thousands of labeled images.  
✅ **Dataset:** Trained on the **Tomato Leaf Disease Dataset** from Kaggle, ensuring high accuracy.  
✅ **Fast and Accurate Detection:** The model processes images in seconds, providing **real-time results**.  
✅ **Comprehensive Disease Information:** Detailed insights on each detected condition, including symptoms and treatment recommendations.  
✅ **Chatbot Assistance:** Get AI-powered responses to any plant health-related queries.  

### 📌 Why Use This System?  
🔹 **Early Detection:** Prevent crop loss by identifying diseases at an early stage.  
🔹 **Actionable Insights:** Get **expert-recommended treatments** tailored for each disease.  
🔹 **User-Friendly Interface:** Simple, mobile-friendly design for easy access in the field.  

This **Deep Learning-powered** system helps you **protect and optimize** your tomato crops efficiently! 🚀  
""")


# Disease Recognition Page
elif page == "🩺 Disease Recognition":
    st.title("🩺 Tomato Disease Recognition")
    
    test_image = st.file_uploader("📤 Upload a Tomato Leaf Image:", type=["jpg", "png", "jpeg"])
    
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Predict Disease"):
            with st.spinner("Analyzing... Please wait ⏳"):
                time.sleep(2)  # Simulating processing time
                
                image = Image.open(test_image).convert("RGB")  
                image = image.resize((128, 128))  
                input_arr = np.array(image) / 255.0  
                input_arr = np.expand_dims(input_arr, axis=0)  
                predictions = model.predict(input_arr)
                result_index = np.argmax(predictions)
                
                predicted_class = list(disease_info.keys())[result_index]
                confidence = np.max(predictions) * 100  
                
                st.success(f"✅ Model Prediction: **{predicted_class}**")
                st.info(f"📊 Confidence Level: **{confidence:.2f}%**")
                
                st.subheader("📊 Prediction Confidence Distribution")
                fig, ax = plt.subplots()
                sns.barplot(x=list(disease_info.keys()), y=predictions[0], ax=ax, palette="coolwarm")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                st.pyplot(fig)
                
                st.subheader("📝 Disease Information")
                st.write(disease_info[predicted_class]["description"])
                
                st.subheader("💊 Suggested Treatments")
                for treatment in disease_info[predicted_class]["treatment"]:
                    st.markdown(f'<div class="treatment-box">{treatment}</div>', unsafe_allow_html=True)
                
                st.subheader("📢 Feedback")
                feedback = st.radio("Was this prediction helpful?", ["👍 Yes", "👎 No"])
                if feedback:
                    st.write("Thank you for your feedback!")

# Chatbot Page
elif page == "🤖 Chatbot":
    st.title("🤖 Tomato Disease Chatbot")
    st.write("Ask me anything about tomato plant diseases!")
    
    user_input = st.text_input("💬 Type your question:")
    if st.button("Send") and user_input:
        with st.spinner("Thinking..."):
            response = chatbot(user_input, max_length=100)[0]['generated_text']
            st.write("🗨️ Chatbot:", response)

# Extra Services Page
elif page == "🌐 Extra Services":
    st.title("🌐 Extra Services")
    st.markdown("""
    ### Useful Agricultural Resources
    Explore these websites for more information on farming, plant care, and disease management:
    """)

    st.subheader("🔗 Helpful Links")
    st.markdown("""
    - [TNAU Agritech Crop Protection](https://agritech.tnau.ac.in/crop_protection/crop_prot.html)
    - [FAO Plant Protection](http://www.fao.org/agriculture/crops/thematic-sitemap/theme/pests/en/)
    - [Gardeners' World](https://www.gardenersworld.com/how-to/grow-plants/)
    - [USDA Plant Health](https://www.usda.gov/topics/plants)
    - [Kaggle Datasets](https://www.kaggle.com/datasets)
    """)
