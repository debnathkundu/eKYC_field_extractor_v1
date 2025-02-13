import re
import json
import numpy as np
import torch
# from transformers import pipeline
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from ultralytics import YOLO
from PIL import Image, ImageDraw
from io import BytesIO
import cv2

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


# ---------------------- Helper: Convert PIL Image to Bytes ----------------------
def pil_to_bytes(image, fmt="PNG"):
    """
    Converts a PIL Image to bytes.
    """
    buf = BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()

# ---------------------- Heuristic Helper ----------------------
def is_candidate_name(line):
    """
    Returns True if the line is a plausible candidate for a name.
    Heuristics used:
      - The line is not empty.
      - The line does not contain digits or date separators.
      - The line does not contain common words that are NOT names.
      - The line has at least two words.
    """
    if not line.strip():
        return False
    # Reject lines that have digits (dates, numbers)
    if re.search(r'\d', line):
        return False
    # Reject lines with common keywords found in header/footer or non-name parts.
    ignore_keywords = [
        "DEPARTMENT", "GOVT", "GOVT.", "INCOME", "TAX", "PERMANENT", "ACCOUNT", "PermanentAccount", "AccountNumber", "NumberCard", "AADHAAR", "DOWNLOAD", "DATE", "DOWNLOADDATE",
        "NUMBER", "SIGNATURE", "ADDRESS", "OF INDIA", "MINISTRY", "OFFICE", "PEHACHAN", "AAM", "ADMI", "KA", "ADHIKAR", "issue", "print", "date",
        "MERA", "MERI", "AUTHORITYOFINDIA", "UNIQUE", "IDENTIFICATION", "AUTHORITY", "Name", "Father", "Father's", "Father's Name",
        "GOVERNMENTOFT", "GOVERNMENTOF", "OFINDIA", "GENDER", "HELP", "UIDAI", "GOV", "OF", "DOB", "Male", "Female", "DOB:", "/Male", "/Female"
    ]
    for kw in ignore_keywords:
        if kw.lower() in line.lower():
            return False
    # # Require at least two words to consider as a full name.
    # if len(line.strip().split()) < 2:
    #     return False
    return True



# ---------------------- Extraction Functions ----------------------
# def extract_pan_front(text):
#     """
#     Extracts fields from PAN front:
#       - name, father's name, dob, pan_number.
#     Assumes that candidate names are the first two “name-like” lines.
#     """
#     details = {}
#     lines = [line.strip() for line in text.splitlines() if line.strip()]
    
#     # Heuristic: among the lines that look like names, assume the first is the card holder’s name,
#     # and the second is the father’s name.
#     candidate_names = [line for line in lines if is_candidate_name(line)]
#     details["name"] = candidate_names[0] if candidate_names else None
#     details["fathers_name"] = candidate_names[1] if len(candidate_names) > 1 else None

#     # DOB: look for dd/mm/yyyy or dd-mm-yyyy
#     dob_pattern = re.compile(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b')
#     m = dob_pattern.search(text)
#     details["dob"] = m.group(0) if m else None

#     # PAN number: pattern of 5 letters, 4 digits, and 1 letter (all uppercase)
#     pan_pattern = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')
#     m = pan_pattern.search(text)
#     details["pan_number"] = m.group(0) if m else None

#     print("\n*** PAN Front Details ***")
#     print(json.dumps(details, indent=4))
    
#     return details

def extract_aadhar_front(text):
    """
    Extracts fields from Aadhaar front:
      - name, dob, gender, aadhar_number.
    """
    details = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Use the same heuristic for name
    candidate_names = [line for line in lines if is_candidate_name(line)]
    details["name"] = candidate_names[0] if candidate_names else None

    # DOB extraction
    dob_pattern = re.compile(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b')
    m = dob_pattern.search(text)
    details["dob"] = m.group(0) if m else None

    # Gender extraction: look for "Male" or "Female" (case-insensitive)
    gender_pattern = re.compile(r'\b(Male|Female)\b', re.IGNORECASE)
    m = gender_pattern.search(text)
    details["gender"] = m.group(0).capitalize() if m else None

    # Aadhaar number: 12 consecutive digits or in groups (e.g., "1234 5678 9123")
    aadhaar_pattern = re.compile(r'(\d{4}\s\d{4}\s\d{4}|\d{12})(?=\n)')
    m = aadhaar_pattern.search(text)
    details["aadhar_number"] = m.group(0) if m else None

    return details

def extract_aadhar_rear(image, yolo_model, model):
    """
    For Aadhaar rear, first run the image through a YOLO model to detect the address region.
    Then:
      1. Crop the detected address region and run OCR on it to extract the "address" field.
      2. Black out that region in a copy of the image, run OCR on the modified image, and
         apply regex to extract the "aadhar_number" field.
    """
    # Load the image with PIL
    # image = Image.open(image_path).convert("RGB")
    
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    np_image = np.array(image)
    
    # ---------------- YOLO Address Detection ----------------
    # Run the YOLO model (assumed to be trained to detect the address region)
    results = yolo_model(np_image)
    
    # Get bounding boxes from the first result
    boxes = results[0].boxes
    # print (boxes)
    if len(boxes) == 0:
        address_box = None
    else:
        # Choose the bounding box with the maximum confidence
        confidences = boxes.conf
        max_idx = int(confidences.argmax())
        box = boxes.xyxy[max_idx].tolist()  # format: [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box)
        address_box = (x1, y1, x2, y2)
    
    if address_box is None:
        address_text = None
        aadhar_number = None
    else:
        # ---------------- Extract Address ----------------
        # Crop the detected address region and run OCR on it
        address_crop = image.crop(address_box)
        address_bytes = pil_to_bytes(address_crop)
        address_doc = DocumentFile.from_images([address_bytes])
        result_address = model(address_doc)
        address_text = filter_text(result_address)
        print ("\n*** Aadhaar Rear Filtered Address: ***\n", address_text)
        
        # ---------------- Extract Aadhaar Number ----------------
        # Create a copy of the image and black out the address region
        image_modified = image.copy()
        draw = ImageDraw.Draw(image_modified)
        draw.rectangle(address_box, fill="black")
        
        # # Display the modified image (for debugging/visualization)
        # import matplotlib.pyplot as plt
        # plt.imshow(image_modified)
        # plt.title("Modified Aadhaar Rear Image with Address Blacked Out")
        # plt.axis("off")
        # plt.show()
        
        
        modified_bytes = pil_to_bytes(image_modified)
        modified_doc = DocumentFile.from_images([modified_bytes])
        result_modified = model(modified_doc)
        ocr_text_modified = filter_text(result_modified)
        print("\n*** Aadhaar Rear Aadhar Number Filtered: ***\n", ocr_text_modified)
        # ocr_text_modified = result_modified.render()
        # print("Raw OCR text (modified):", ocr_text_modified)    
        
        # Use regex to extract the Aadhaar number from the OCR text.
        # aadhaar_pattern = re.compile(r'(\d{4}\s\d{4}\s\d{4}|\d{12})')
        aadhaar_pattern = re.compile(r'(\d{4}\s\d{4}\s\d{4}|\d{12})(?=\n)')
        m = aadhaar_pattern.search(ocr_text_modified)
        aadhar_number = m.group(0) if m else None
        
    return {"address": address_text, "aadhar_number": aadhar_number}


# ---------------------- Heuristic Helper ----------------------
def is_candidate_name_pan(line):
    """
    Returns True if the line is a plausible candidate for a name.
    Heuristics used:
      - The line is not empty.
      - The line does not contain digits.
      - The line does not contain common keywords (e.g. INCOME, TAX, SIGNATURE, etc).
      - The line has at least one alphabetical character.
    (Note: we do not require two words so that single–word names can pass.)
    """
    line = line.strip()
    if not line:
        return False
    # Reject lines that have digits (dates, numbers)
    if re.search(r'\d', line):
        return False

    # List of words/phrases to ignore
    ignore_keywords = [
        "DEPARTMENT", "GOVT", "GOVT.", "INCOME", "TAX", "PERMANENT", "ACCOUNT", "AADHAAR",
        "DOWNLOAD", "DATE", "DOWNLOADDATE", "NUMBER", "SIGNATURE", "ADDRESS", "OF INDIA",
        "MINISTRY", "OFFICE", "PEHACHAN", #"AAM", "ADMI", "KA", "ADHIKAR",
        "MERA", "MERI", "AUTHORITYOFINDIA", "UNIQUE", "IDENTIFICATION", "AUTHORITY", "OF",
        "FATHER", "FATHER'S", "FATHER'S NAME", "GOVERNMENTOFT", "GOVERNMENTOF",
        "OFINDIA", "GENDER", "HELP", "UIDAI", "GOV", "DOB", "/MALE", "/FEMALE", "DOB:"
    ]
    # If the entire line exactly matches an ignore keyword, reject it.
    for kw in ignore_keywords:
        if kw.lower() == line.lower():
            return False
    # Also, if the line contains any of these keywords (as part of a longer phrase), reject it.
    for kw in ignore_keywords:
        if kw.lower() in line.lower():
            return False

    # Reject lines that don't contain any alphabets.
    if not re.search(r'[A-Za-z]', line):
        return False
    return True

# ---------------------- Extraction Function ----------------------
def extract_pan_front(text):
    """
    Extracts PAN card front details: cardholder name, father's name, dob, pan_number.
    
    Two modes are used:
      1. **Explicit Label Mode:** If a line exactly matching “Name” or
         (a variant of) “Father’s Name” is found, then the candidate lines
         immediately following that label are used.
      2. **Fallback Mode:** If explicit labels are not found, then candidate lines 
         that appear before the date/PAN data are collected. In fallback mode, if 
         exactly two candidate lines are detected then they are assumed to be the cardholder’s
         and father’s names respectively. If only one is found, father's name is set to None.
         For more than two candidate lines, consecutive lines are grouped—with a heuristic 
         to avoid merging when the first candidate is a single word and the next has multiple words.
    """
    details = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # # Define patterns to detect explicit labels.
    # name_label_pattern   = re.compile(r'^\s*Name\s*$', re.IGNORECASE)
    # father_label_pattern = re.compile(r'^\s*/?\s*father(\'?s)?\s*name\s*$', re.IGNORECASE)
    
    # For the "Name" label, allow either an exact match "Name" or any extra text provided that there is a slash immediately before "Name".
    name_label_pattern = re.compile(
        r'^(?:Name|.*\/\s*Name)\s*$',
        re.IGNORECASE
    )

    # For the "Father's Name" label, allow either an exact match (with or without the apostrophe)
    # or extra text (with a slash) immediately preceding the keyword.
    father_label_pattern = re.compile(
        r'^(?:father(?:\'?s)?\s*name|.*\/\s*father(?:\'?s)?\s*name)\s*$',
        re.IGNORECASE
    )

    explicit_name_idx = None
    explicit_father_idx = None
    for i, line in enumerate(lines):
        if name_label_pattern.match(line):
            explicit_name_idx = i
        if father_label_pattern.match(line):
            explicit_father_idx = i

    # A helper that gathers candidate lines following a label.
    def extract_field(start_index):
        field_lines = []
        for j in range(start_index + 1, len(lines)):
            # Stop if we run into another label
            if name_label_pattern.match(lines[j]) or father_label_pattern.match(lines[j]):
                break
            # Stop if a line contains a date or PAN number.
            if re.search(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', lines[j]) or re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', lines[j]):
                break
            if is_candidate_name_pan(lines[j]):
                field_lines.append(lines[j])
            else:
                break
        return " ".join(field_lines).strip() if field_lines else None

    # --- Mode 1: Explicit Label Mode ---
    if explicit_name_idx is not None:
        details["name"] = extract_field(explicit_name_idx)
    if explicit_father_idx is not None:
        details["fathers_name"] = extract_field(explicit_father_idx)

    # --- Mode 2: Fallback Mode if explicit extraction did not yield both names ---
    if details.get("name") is None or details.get("fathers_name") is None:
        # Determine the earliest index where a date or PAN number appears.
        earliest_index = len(lines)
        date_pattern = re.compile(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b')
        pan_pattern  = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')
        for i, line in enumerate(lines):
            if date_pattern.search(line) or pan_pattern.search(line):
                earliest_index = min(earliest_index, i)
        
        print("earliest_index:", earliest_index)
        
        # Collect candidate lines (with their indices) before that point.
        fallback_candidates = []
        for i, line in enumerate(lines):
            if i >= earliest_index:
                break
            if is_candidate_name_pan(line):
                fallback_candidates.append((i, line))
        
        print("fallback_candidates:", fallback_candidates)
        
        # Define a set of label-like strings (in lowercase) that we want to ignore.
        explicit_labels = {
            "name",
            "/name",
            "father",
            "father's name",
            "/father",
            "/father's name",
            "father'",
            "/father'",
            "far",
            "hra"
        }

        # Filter out any fallback candidate that is just one of the explicit labels.
        fallback_candidates = [
            (i, line) for (i, line) in fallback_candidates
            if line.strip().lower() not in explicit_labels
        ]
        
        # Print the fallback candidates after filtering.
        print("fallback_candidates after filtering:", fallback_candidates)
        
        # Use the number of fallback candidates to decide:
        if len(fallback_candidates) == 0:
            # No candidates found.
            if details.get("name") is None:
                details["name"] = None
            if details.get("fathers_name") is None:
                details["fathers_name"] = None
        elif len(fallback_candidates) == 1:
            # Only one candidate; assume it's the cardholder's name.
            details["name"] = fallback_candidates[0][1]
            details["fathers_name"] = None
        elif len(fallback_candidates) == 2:
            # Exactly two candidate lines; assign them separately.
            details["name"] = fallback_candidates[0][1]
            details["fathers_name"] = fallback_candidates[1][1]
        else:
            # More than two candidates: group them based on index gaps.
            groups = []
            current_group = [fallback_candidates[0][1]]
            prev_idx = fallback_candidates[0][0]
            for idx, txt in fallback_candidates[1:]:
                if idx == prev_idx + 1:
                    # Heuristic: if the current group has exactly one candidate that is a single word
                    # and the new candidate has multiple words, treat them as separate fields.
                    if len(current_group) == 1 and len(current_group[0].split()) == 1 and len(txt.split()) >= 2:
                        groups.append(" ".join(current_group))
                        current_group = [txt]
                    else:
                        current_group.append(txt)
                else:
                    groups.append(" ".join(current_group))
                    current_group = [txt]
                prev_idx = idx
            groups.append(" ".join(current_group))
            # Assign the groups.
            details["name"] = groups[0] if groups else None
            details["fathers_name"] = groups[1] if len(groups) > 1 else None

    # --- Extract DOB and PAN number from the whole text ---
    m = re.search(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', text)
    details["dob"] = m.group(0) if m else None

    m = re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', text)
    details["pan_number"] = m.group(0) if m else None

    # Ensure that if father's name is still not detected, we explicitly set it to None.
    if "fathers_name" not in details or details["fathers_name"] is None:
        details["fathers_name"] = None

    print("\n*** PAN Front Details ***")
    print(json.dumps(details, indent=4))
    return details
