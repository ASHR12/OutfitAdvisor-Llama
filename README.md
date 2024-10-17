# OutfitAdvisor

OutfitAdvisor is a smart fashion recommendation application that leverages advanced AI models to provide personalized clothing suggestions based on user-uploaded images and queries. The application uses Google Gemini for text embeddings, and Llama 3.2 11B Vision (Preview) via Groq AI for image analysis, and Pinecone for vector search.

## Features

- **Image Upload**: Users can upload fashion images to receive recommendations.
- **AI-Powered Recommendations**: Utilizes Llama 3.2 11B Vision (Preview) to analyze images and provide fashion suggestions.
- **Vector Search**: Uses Pinecone to find similar fashion items based on AI recommendations.
- **User-Friendly Interface**: Built with Streamlit for an intuitive and interactive user experience.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/OutfitAdvisor.git
cd OutfitAdvisor
```

### Create a Virtual Environment

#### Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

#### macOS and Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory of the project and add the following environment variables with your API keys:

```plaintext
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=your_pinecone_index_name
GOOGLE_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

### Run the Application

Start the Streamlit application by running:

```bash
streamlit run app.py
```

This will launch the application in your default web browser.

## Usage

1. **Upload an Image**: Use the sidebar to upload a fashion image.
2. **Ask a Question**: Enter a question related to the fashion image.
3. **Get Recommendations**: Click the "Get Recommendations" button to receive personalized fashion suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

This README provides a comprehensive guide to setting up and using the OutfitAdvisor application. Ensure you have all necessary API keys and dependencies installed to fully utilize the application's features.
