# app.py
import streamlit as st
import os
import base64
import logging
from helper_utils import (
  save_image,
  delete_image,
  get_recommendation,
  query_pinecone,
  process_image_display,
  configure_genai,
)
from dotenv import load_dotenv
from PIL import Image
# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Read the image file
logo = Image.open("static-assets/groq-logo.png")
st.logo(
    logo,
)
# Set Streamlit page configuration
st.set_page_config(
  page_title="OutfitAdvisor",
  layout="wide",
  initial_sidebar_state="expanded",
)

st.markdown("<h1 style='text-align: center;'>OutfitAdvisor</h1>", unsafe_allow_html=True)  
st.markdown("<p style='text-align: center; color: gray;'>Smart Fashion Recommendations Powered by Llama 3.2 11B Vision </p>", unsafe_allow_html=True)

# 1. Create a sidebar for image upload
with st.sidebar:
  st.subheader("Upload a Fashion Image")
  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
  
  if uploaded_file is not None:
      try:
          # Save the uploaded image temporarily to process and display
          image_path_temp = save_image(uploaded_file)
          display_image = process_image_display(image_path_temp, max_width=400, max_height=400)
          st.image(display_image, caption="Uploaded Image", use_column_width=True)
      except Exception as e:
          logging.error(f"Error displaying uploaded image: {e}")
          st.error("Failed to display the uploaded image. Please try again.")

# 5. Graceful Error Handling for Missing Keys
# Load environment variables from .env
env_path = '.env'
if os.path.exists(env_path):
  load_dotenv(dotenv_path=env_path)
  logging.info(".env file found and loaded.")
else:
  logging.error(".env file not found.")
  st.error(".env file not found. Please ensure it exists and contains all necessary API keys.")
  st.stop()

# Define required environment variables
required_env_vars = {
  'PINECONE_API_KEY': 'Pinecone API Key',
  'INDEX_NAME': 'Pinecone Index Name',
  'GOOGLE_API_KEY': 'Google Gemini Key (for embeddings)',
  'GROQ_API_KEY': 'Groq API Key'
}

# Validate the presence of all required API keys
missing_vars = [
  var for var in required_env_vars if not os.getenv(var)
]

if missing_vars:
  missing_keys_str = ", ".join(missing_vars)
  logging.error(f"Missing required API keys: {missing_keys_str}")
  st.error(f"Missing required API keys: {missing_keys_str}. Please update the .env file.")
  st.stop()

# Extract API keys from environment variables
api_keys = {var: os.getenv(var) for var in required_env_vars}

# Configure Google Gemini once
try:
  configure_genai(api_keys['GOOGLE_API_KEY'])
except Exception as e:
  logging.error(f"Failed to configure Google Gemini API: {e}")
  st.error(f"Failed to configure Google Gemini API: {e}")
  st.stop()


with st.container():
  
    user_prompt = st.text_input(
        "Ask a Question",
        placeholder="Enter your question here...",
        key="user_prompt",
        max_chars=500  # Limit the width of the input to 500px
    )
  
    get_rec_button = st.button("Get Recommendations", type="primary")

# 4. Display Results Incrementally
if get_rec_button:
  if not uploaded_file:
      st.error("Please upload an image to get recommendations.")
  elif not user_prompt.strip():
      st.error("Please enter a question to proceed.")
  else:
      with st.spinner("Processing your request..."):
          try:
              # Save the uploaded image
              image_path = save_image(uploaded_file)

              # Encode image in base64
              with open(image_path, "rb") as img_file:
                  base64_image = base64.b64encode(img_file.read()).decode("utf-8")

              # Get recommendation from AI model
              recommendation = get_recommendation(base64_image, user_prompt)

              # Validate recommendation
              required_keys = [
                  "Option_1",
                  "Option_2",
                  "Option_3"
              ]
              if not all(key in recommendation for key in required_keys):
                  st.error("No fashion recommendation could be made. Please upload a valid fashion item image.")
              else:
                  st.markdown("### Recommendations:")
                  
                  # Display each option with its corresponding images
                  for option in required_keys:
                      st.markdown(f"**{recommendation[option]}:**")

                      # Perform Pinecone vector search for each option
                      results = query_pinecone(
                          recommendation[option],
                          pinecone_api_key=api_keys['PINECONE_API_KEY'],
                          index_name=api_keys['INDEX_NAME']
                      )

                      # Display search results in a grid (3 columns per row) with specified width and height
                      if results and 'matches' in results and len(results['matches']) > 0:
                          cols = st.columns(3)
                          for col, match in zip(cols, results['matches']):
                              with col:
                                  metadata = match['metadata']
                                  image_url = metadata.get('image_url', "https://via.placeholder.com/150")
                                  product_name = metadata.get('productDisplayName', "Product Name Not Available")

                                  col.image(image_url, use_column_width=True, clamp=True)

                                  # Display product details with limited width
                                  product_info = f"""
                                  **ID:** {match['id']}  
                                  **Name:** {product_name}  
                                  """
                                  col.markdown(product_info)
                      else:
                          st.info("No products found for this option.")

              # Delete the uploaded image after processing
              delete_image(image_path)

          except Exception as e:
              logging.error(f"An error occurred: {e}")
              st.error("An unexpected error occurred. Please try again.")

# 6. Footer with Groq Logo and Additional Messages
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 2])

with footer_col2:
  with open("static-assets/PBG mark2 color.svg", "r") as f:
    svg_content = f.read()

  # Display the SVG
  st.image(svg_content, use_column_width="always")
  