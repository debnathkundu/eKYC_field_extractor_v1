import streamlit as st
import json
import re
from io import BytesIO
from PIL import Image
from transformers import pipeline
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from ultralytics import YOLO

from v1_raw import *

# -------------------- Initialize Global Models --------------------
ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq', pretrained=True).cuda()
yolo_model = YOLO('/workspaces/eKYC_field_extractor_v1/app.py')

# -------------------- Utility Functions --------------------
def filter_text(res):
    """Keep only words with high confidence to reduce OCR noise."""
    text = ""
    for page in res.pages:
        for block in page.blocks:
            for line in block.lines:
                line_str = " ".join([word.value for word in line.words if word.confidence > 0.97])
                if line_str.strip():
                    text += line_str.strip() + "\n"
    return text

def infer(image):
    """Perform OCR on the image and return both the rendered text and filtered text."""
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    img = DocumentFile.from_images([image_bytes.getvalue()])
    result = ocr_model(img)
    rendered_text = result.render()
    filtered_text = filter_text(result)
    return rendered_text, filtered_text

def extract_pan_details(text):
    return extract_pan_front(text)

def extract_aadhar_front_details(text):
    return extract_aadhar_front(text)

def extract_aadhar_rear_details(image, yolo_model, ocr_model):
    return extract_aadhar_rear(image, yolo_model, ocr_model)

# -------------------- Streamlit Interface --------------------
st.title("KYC Document Field Extractor")
st.markdown("""
Hello! This is a tiny web-app to help in data extraction for eKYC purposes. Please assist us in evaluating its functionalities. \n
Choose the corresponding option for eKYC & upload the appropriate document.
- **Option 1 - through PAN Front**: Name, Father's Name, DOB, PAN number will be extracted.
- **Option 2 - through Aadhaar**:
  - **Aadhaar Front**: Aadhaar number, name, DOB, gender, and Aadhaar number will be extracted.
  - **Aadhaar Rear**: Aadhaar number and Address will be extracted.
  
  

**Please correlate these fields according to your image.**
""")

col1, col2 = st.columns([1, 3])
with col1:
    doc_type = st.selectbox("Select Document Type", ["PAN Card", "Aadhaar Card"])

if doc_type == "PAN Card":
    pan_image = st.file_uploader("Upload PAN Card Front Image", type=["png", "jpg", "jpeg"], key="pan_front")

    if pan_image:
        st.image(pan_image, caption="Uploaded PAN Card Front", use_container_width=True)
        pan_img = Image.open(pan_image)
        _, pan_front_text = infer(pan_img)
        print("\n*** PAN Front Filtered Text: ***\n", pan_front_text)
        
        pan_details = extract_pan_details(pan_front_text)
        st.subheader("Extracted PAN Card Details:")
        st.json(pan_details)

elif doc_type == "Aadhaar Card":
    aadhar_front_image = st.file_uploader("Upload Aadhaar Card Front Image", type=["png", "jpg", "jpeg"], key="aadhaar_front")
    aadhar_rear_image = st.file_uploader("Upload Aadhaar Card Rear Image", type=["png", "jpg", "jpeg"], key="aadhaar_rear")

    if aadhar_front_image and aadhar_rear_image:
        st.image([aadhar_front_image, aadhar_rear_image], caption=["Uploaded Aadhaar Card Front", "Uploaded Aadhaar Card Rear"], use_container_width=True)
        aadhar_front_img = Image.open(aadhar_front_image)
        aadhar_rear_img = Image.open(aadhar_rear_image)
        
        _ , aadhar_front_text = infer(aadhar_front_img)
        print("\n*** Aadhaar Front Filtered Text: ***\n", aadhar_front_text)
        
        rear_details = extract_aadhar_rear_details(aadhar_rear_img, yolo_model, ocr_model)
        front_details = extract_aadhar_front_details(aadhar_front_text)
        
        st.subheader("Extracted Aadhaar Card Details:")
        st.json({
            "Front Side Details": front_details,
            "Rear Side Details": rear_details
        })
