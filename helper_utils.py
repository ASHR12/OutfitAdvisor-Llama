# helper_utils.py

import os
import json
import logging
from PIL import Image
import google.generativeai as genai
from groq import Groq
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_genai(api_key):
  """Configure the Google Gemini API."""
  try:
      genai.configure(api_key=api_key)
      logging.info("Google Gemini API configured successfully.")
  except Exception as e:
      logging.error(f"Failed to configure Google Gemini API: {e}")
      raise

def get_embedding(text, model="models/text-embedding-004"):
  """Get embedding for a given text."""
  try:
      result = genai.embed_content(
          model=model,
          content=text,
          task_type="retrieval_document"
      )
      return result['embedding']
  except Exception as e:
      logging.error(f"Error getting embedding: {e}")
      raise

def save_image(uploaded_file, folder="user_images"):
  """Save uploaded image to the specified folder."""
  try:
      if not os.path.exists(folder):
          os.makedirs(folder)
          logging.info(f"Created folder: {folder}")

      image_path = os.path.join(folder, uploaded_file.name)
      
      # Verify the file is indeed an image
      with Image.open(uploaded_file) as img:
          img.verify()  # Verifies that it is, in fact, an image

      # Save the image
      with open(image_path, "wb") as f:
          f.write(uploaded_file.getbuffer())
      logging.info(f"Image saved to {image_path}")
      return image_path
  except Exception as e:
      logging.error(f"Error saving image: {e}")
      raise

def delete_image(image_path):
  """Delete the processed image from the local folder."""
  try:
      if os.path.exists(image_path):
          os.remove(image_path)
          logging.info(f"Deleted image at {image_path}")
  except Exception as e:
      logging.error(f"Error deleting image: {e}")

def get_recommendation(base64_image, user_prompt):
  """Call the Groq AI API to get a recommendation."""
  groq_api_key = os.getenv('GROQ_API_KEY')
  if not groq_api_key:
      logging.error("GROQ_API_KEY is not set in environment variables.")
      raise ValueError("GROQ_API_KEY not found.")

  client = Groq(api_key=groq_api_key)
  master_prompt = f"""
  You are an AI Fashion Assistant specializing in analyzing fashion images and providing personalized clothing recommendations based on user question: ```{user_prompt}``` You will provide minimum 3 options as given in JSON, Analyze the given image along with the user question and suggest item. Return your response in 10 words in the below JSON Format.
  {{
    "Option_1": "string",
    "Option_2": "string",
    "Option_3": "string"
  }}
  """


  try:
      response = client.chat.completions.create(
          messages=[
              {"role": "user", "content": [
                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                  {"type": "text", "text": master_prompt}
              ]}
          ],
          model='llama-3.2-11b-vision-preview',
          response_format={"type": "json_object"}
      )
      content = response.choices[0].message.content
      recommendation = json.loads(content)
      logging.info(f"Recommendation received: {recommendation}")
      return recommendation
  except json.JSONDecodeError:
      logging.error("Failed to decode JSON response from the AI model.")
      raise ValueError("Invalid response format from the AI model.")
  except Exception as e:
      logging.error(f"Error getting recommendation: {e}")
      raise

def query_pinecone(query_text, pinecone_api_key, index_name, top_k=3):
  """Perform a vector search in Pinecone."""
  try:
      pc = Pinecone(pinecone_api_key)
      while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
      index = pc.Index(index_name)
  except Exception as e:
      logging.error(f"Error connecting to Pinecone: {e}")
      raise

  try:
      # Get embedding for the query text
      embedding = get_embedding(query_text)

      # Query Pinecone
      results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
      logging.info(f"Pinecone query results: {results}")
      return results
  except Exception as e:
      logging.error(f"Error querying Pinecone: {e}")
      raise

def process_image_display(image_path, max_width=400, max_height=400):
  """Process image to display with specified width and height."""
  try:
      with Image.open(image_path) as img:
          img = img.convert("RGB")  # Ensure image is in RGB format
          img = img.resize((max_width, max_height), Image.LANCZOS)
          logging.info(f"Processed image to {max_width}x{max_height}")
          return img
  except Exception as e:
      logging.error(f"Error processing image for display: {e}")
      raise


def load_data(file_path):
  """Load JSON data from the specified file."""
  try:
      with open(file_path, 'r') as file:
          data = json.load(file)
      logging.info(f"Loaded data from {file_path}")
      return data
  except Exception as e:
      logging.error(f"Error loading data from {file_path}: {e}")
      raise

def search_by_id(data, search_id):
  """Search for an item by ID in the loaded data."""
  try:
      for item in data:
          if str(item['id']) == str(search_id):
              return item
      logging.warning(f"Product with ID {search_id} not found.")
      return None
  except Exception as e:
      logging.error(f"Error searching for ID {search_id}: {e}")
      raise