# Face Grouping Professional 👥

A Streamlit web application for grouping similar faces from multiple images using face recognition technology.

## Features

- 📤 Upload multiple images simultaneously
- 🔍 Automatic face detection and encoding
- 🤝 Group similar faces based on similarity threshold
- 📊 Interactive statistics and visualizations
- 📥 Download Excel report with embedded images
- ⚡ Fast processing with multiprocessing support

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://face-grouping-pro.streamlit.app)

## How to Use

1. **Upload Images**: Click on "Browse files" and select multiple images (JPG, JPEG, PNG)
2. **Adjust Settings**: Use the similarity threshold slider to control grouping sensitivity
3. **Process**: Click "Start Processing" to begin face detection and grouping
4. **View Results**: See grouped faces in the interactive tabs
5. **Download**: Get an Excel report with embedded images and group information

## Installation (Local)

```bash
# Clone the repository
git clone https://github.com/yourusername/face-grouping-pro.git
cd face-grouping-pro

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py