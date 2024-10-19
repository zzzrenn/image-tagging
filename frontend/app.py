import streamlit as st
import requests
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv('BACKEND_URL', 'http://image_analyzer_backend:6000')

def main():
    st.title("📸 Scene Context Analyzer")
    st.write("Upload an image to analyze its context, location, and environmental attributes!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with st.spinner('Analyzing image context...'):
            # Prepare image for API request
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_byte_arr = img_byte_arr.getvalue()

            # Send image to backend
            files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
            response = requests.post(f"{BACKEND_URL}/analyze", files=files)
            
            if response.status_code == 200:
                results = response.json()
                
                # Display results by category
                st.subheader("Scene Analysis:")
                
                # Create tabs for different categories
                tabs = st.tabs(list(results.keys()))
                
                for tab, category in zip(tabs, results.keys()):
                    with tab:
                        predictions = results[category]
                        cols = st.columns(len(predictions))
                        for idx, pred in enumerate(predictions):
                            with cols[idx]:
                                st.metric(
                                    label=pred['label'].title(),
                                    value=f"{pred['probability']:.1f}%"
                                )
                
                # Display comprehensive results
                st.subheader("Detailed Analysis:")
                
                # Create a DataFrame for all predictions
                all_results = []
                for category, predictions in results.items():
                    for pred in predictions:
                        all_results.append({
                            'Category': category.replace('_', ' ').title(),
                            'Label': pred['label'].title(),
                            'Confidence': f"{pred['probability']:.1f}%"
                        })
                
                st.dataframe(all_results)
            else:
                st.error("Error analyzing image. Please try again.")

if __name__ == "__main__":
    main()