import streamlit as st
import requests as req

st.set_page_config(page_title="Potato Blight Detection", layout="centered")

st.title("🔍 Potato Blight Detection")
st.markdown("Upload an image and click **Predict** to get results from the model.")

api_url = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("📂 Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="📷 Preview", width=250)

    file = {
        'file': (uploaded_file.name, uploaded_file, uploaded_file.type)
    }

    if st.button("🚀 Predict"):
        with st.spinner("Sending image to model..."):
            response = req.post(api_url, files=file)

        if response.status_code == 200:
            st.success("✅ Prediction received!")
            st.subheader("📊 Result:")
            st.write(response.json())
        else:
            st.error(f"❌ Error {response.status_code}")
            st.code(response.text, language="json")
