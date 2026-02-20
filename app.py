import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import tempfile
from PIL import Image
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from io import BytesIO
import plotly.graph_objects as go
import shutil
import os
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Face Grouping Pro",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #4b8bec 0%, #2a5f9e 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #5cb85c;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #4b8bec;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #dff0d8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #5cb85c;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #4b8bec 0%, #2a5f9e 100%);
        color: white;
        border-radius: 5px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>👥 Face Grouping Professional</h1>
    <p>Upload images and group similar faces automatically</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'grouped_faces' not in st.session_state:
    st.session_state.grouped_faces = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location and return path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def extract_face_embedding(image_path, image_name):
    """Extract face embedding using DeepFace"""
    try:
        # Use DeepFace to get face embeddings
        embeddings = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",  # Lightweight and accurate
            enforce_detection=False,
            detector_backend='opencv'  # Use OpenCV for detection (lighter)
        )
        
        if embeddings and len(embeddings) > 0:
            # Return the embedding and image name
            return [(np.array(embeddings[0]['embedding']), image_name)]
        return []
    except Exception as e:
        st.warning(f"Could not process {image_name}: {str(e)}")
        return []

def group_similar_faces(descriptors, threshold=0.6):
    """Group similar faces based on embedding similarity"""
    if not descriptors:
        return []
    
    groups = []
    used = [False] * len(descriptors)
    
    for i in range(len(descriptors)):
        if used[i]:
            continue
        group = [descriptors[i][1]]  # Store image name
        used[i] = True
        
        for j in range(i + 1, len(descriptors)):
            if not used[j]:
                # Calculate cosine distance
                from scipy.spatial.distance import cosine
                distance = cosine(descriptors[i][0], descriptors[j][0])
                if distance < threshold:  # Lower distance means more similar
                    group.append(descriptors[j][1])
                    used[j] = True
        
        if len(group) > 1:  # Only groups with multiple faces
            groups.append(group)
    
    return groups

def create_excel_with_images(groups, image_data, image_paths):
    """Create Excel file with grouped faces and embedded images"""
    wb = Workbook()
    ws = wb.active
    ws.title = "Grouped Faces"
    
    # Add headers
    ws['A1'] = "Image"
    ws['B1'] = "Filename"
    ws['C1'] = "Group Number"
    
    row_num = 2
    temp_dir = tempfile.mkdtemp()
    
    try:
        for group_num, group in enumerate(groups, start=1):
            for image_name in group:
                if image_name in image_paths:
                    temp_path = image_paths[image_name]
                    
                    # Add to Excel
                    try:
                        img = ExcelImage(temp_path)
                        img.width = 100
                        img.height = 100
                        ws.add_image(img, f'A{row_num}')
                    except Exception as e:
                        ws[f'A{row_num}'] = "[Image Error]"
                    
                    ws[f'B{row_num}'] = image_name
                    ws[f'C{row_num}'] = f"Group {group_num}"
                    row_num += 1
        
        # Save to BytesIO
        excel_bytes = BytesIO()
        wb.save(excel_bytes)
        excel_bytes.seek(0)
        
        return excel_bytes
    
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

# Main app layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Images")
    st.markdown('<div class="info-box">Upload multiple images containing faces. Supported formats: JPG, JPEG, PNG</div>', 
                unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose images...",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"✅ {len(uploaded_files)} images uploaded successfully!")
        
        # Display uploaded images in a grid
        st.markdown("### 👁️ Preview")
        cols = st.columns(4)
        for idx, file in enumerate(uploaded_files[:8]):
            with cols[idx % 4]:
                try:
                    image = Image.open(file)
                    st.image(image, caption=file.name, use_column_width=True)
                except:
                    pass
        
        if len(uploaded_files) > 8:
            st.info(f"... and {len(uploaded_files) - 8} more images")

with col2:
    st.markdown("### ⚙️ Processing Settings")
    
    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Lower values = stricter matching, Higher values = looser matching"
    )
    
    st.markdown("### 📊 Statistics")
    
    if st.session_state.processed and st.session_state.grouped_faces:
        total_faces = sum(len(group) for group in st.session_state.grouped_faces)
        total_groups = len(st.session_state.grouped_faces)
        
        # Create metrics
        col2_1, col2_2, col2_3 = st.columns(3)
        col2_1.metric("Total Groups", total_groups)
        col2_2.metric("Total Faces", total_faces)
        col2_3.metric("Avg Faces/Group", f"{(total_faces/total_groups):.1f}")
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=[f"Group {i+1}" for i in range(total_groups)],
            values=[len(group) for group in st.session_state.grouped_faces],
            hole=.3
        )])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# Process button
if st.button("🚀 Start Processing", type="primary", use_container_width=True):
    if not uploaded_files:
        st.error("Please upload at least one image first!")
    else:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Save uploaded files temporarily
            status_text.text("💾 Preparing images...")
            progress_bar.progress(5)
            
            all_embeddings = []
            image_paths = {}
            temp_files = []
            
            for idx, file in enumerate(uploaded_files):
                # Update progress
                progress = 5 + int((idx / len(uploaded_files)) * 45)
                progress_bar.progress(progress)
                status_text.text(f"Processing {file.name}...")
                
                # Save file temporarily
                temp_path = save_uploaded_file(file)
                if temp_path:
                    temp_files.append(temp_path)
                    image_paths[file.name] = temp_path
                    
                    # Extract face embedding
                    embeddings = extract_face_embedding(temp_path, file.name)
                    all_embeddings.extend(embeddings)
            
            if not all_embeddings:
                st.warning("No faces found in the uploaded images!")
            else:
                # Step 2: Group similar faces
                status_text.text("🔄 Grouping similar faces...")
                progress_bar.progress(70)
                
                grouped_faces = group_similar_faces(all_embeddings, threshold)
                
                # Step 3: Create Excel file
                status_text.text("📊 Creating Excel report...")
                progress_bar.progress(85)
                
                excel_file = create_excel_with_images(grouped_faces, image_paths, image_paths)
                
                # Step 4: Complete
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                # Save to session state
                st.session_state.processed = True
                st.session_state.grouped_faces = grouped_faces
                
                # Success message
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ Processing Complete!</h4>
                    <p>Found {len(grouped_faces)} groups of similar faces</p>
                    <p>Total faces detected: {len(all_embeddings)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_file,
                    file_name="face_groups_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            progress_bar.empty()
            status_text.empty()
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

# Results display
if st.session_state.processed and st.session_state.grouped_faces:
    st.markdown("---")
    st.markdown("### 📋 Grouped Faces Preview")
    
    groups = st.session_state.grouped_faces
    
    if groups:
        # Create tabs for each group
        tabs = st.tabs([f"Group {i+1} ({len(group)} faces)" for i, group in enumerate(groups)])
        
        for idx, (tab, group) in enumerate(zip(tabs, groups)):
            with tab:
                cols = st.columns(3)
                for img_idx, image_name in enumerate(group):
                    with cols[img_idx % 3]:
                        # Find the image in uploaded files
                        for file in uploaded_files:
                            if file.name == image_name:
                                image = Image.open(file)
                                st.image(image, caption=image_name, use_column_width=True)
                                file.seek(0)
                                break
    else:
        st.info("No face groups were found. Try adjusting the similarity threshold.")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Designed by Bijay Paswan | v1.0 | Face Grouping Professional</p>
</div>
""", unsafe_allow_html=True)
