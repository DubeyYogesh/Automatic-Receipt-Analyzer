#!/usr/bin/env python
# coding: utf-8

# In[15]:

from paddleocr import PaddleOCR, draw_ocr
import cv2
from matplotlib import pyplot as plt
import re
from transformers import pipeline
import json

# Load the zero-shot-classification pipeline using a pretrained model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def extract_image(image_path):
    """
    This function performs OCR on a given image path and extracts structured receipt data.
    Arguments:
        image_path: str - Path to the receipt image file.
    Returns:
        A dictionary with extracted information: store name, city, date, total, and list of items with predicted categories.
    """

    # Initialize PaddleOCR with tuned parameters for more accurate detection
    ocr = PaddleOCR(
        use_angle_cls=True,     # Use angle classification to fix rotated text
        lang='en',              # Use English OCR model
        use_space_char=True,    # Preserve space characters in output
        show_log=False,         # Suppress PaddleOCR logs

        # Detection tuning parameters for better box quality
        det_db_box_thresh=0.7,       # Threshold to filter low-score detection boxes (default ~0.5)
        det_db_thresh=0.7,           # Binary thresholding step in box detection; higher = stricter
        det_db_unclip_ratio=1.2,     # Shrinks boxes slightly to reduce overlap (default ~2.0)
        det_db_score_mode='slow',    # Improves quality at the cost of speed; can use 'fast' or 'slow'
    )
    results = ocr.ocr(image_path, cls=True)

    # Helper to check if a line is a price (like "29.99" or "100")
    def is_price(line):
        return bool(re.fullmatch(r'\d+(\.\d{1,2})?', line) or re.fullmatch(r'^(?:\d+(?:\.\d+)?\s*)+$', line))

    # Helper to check if a line contains letters (likely item name)
    def is_item_name(line):
        return bool(re.search(r'[A-Za-z]', line))

    result = results[0]  # Take first OCR result set (usually sufficient for single images)

    # Detect address-like lines using keywords
    def is_address_line(text):
        address_keywords = ['road', 'street', 'lane', 'block', 'no.', 'opp', 'near', 'layout', 'nagar', 'market', 'circle', 'complex', 'building', 
                            'sector', 'phase', 'floor', 'main', 'hotels', 'pvt', 'ltd']
        text_lower = text.lower()
        return any(kw in text_lower for kw in address_keywords) or bool(re.search(r'\d{2,}', text))

    # Heuristics to detect store name: short, alphabetic, no numbers, not an address
    def is_possible_store_line(text):
        return (
            bool(re.search(r'[A-Za-z]', text)) and
            not bool(re.search(r'\d{2,}', text)) and
            len(text.split()) <= 4 and
            not is_address_line(text)
        )

    # Detect city using known city names and zip-like numbers
    def is_city_name(text):
        return bool(re.search(r'([A-Za-z ]+)[\s\-]+(\d{2,6})', text))

    # Try to extract store name from top lines
    store_name_lines = []
    for i in range(min(5, len(result))):
        line = result[i][1][0].strip()
        if is_possible_store_line(line):
            store_name_lines.append(line)
        elif is_address_line(line):
            break
    store_name = ' '.join(store_name_lines)

    # Try to match known city names in the first few lines
    cities = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata", "Surat", "Pune", "Jaipur"]
    city = ''
    for i in range(min(10, len(result))):
        line = result[i][1][0].strip()
        if is_city_name(line):
            for city in cities:
                if city.lower() in line.lower():
                    city = city
                    break

    # Try to find the total amount by looking for known keywords followed by a price
    total_keywords = ['grand total', 'total', 'total mrp', 'total amount', 'amount payable', 'amount due', 'food total']
    total = ""
    for i in range(len(result) - 1, 0, -1):
        current_line = result[i][1][0].strip().lower()
        prev_line = result[i - 1][1][0].strip().lower()
        if re.match(r'^\d+(\.\d{2})?$', current_line):
            if any(k in prev_line for k in total_keywords):
                total = current_line
                break
        else:
            if re.match(r'^\d+(\.\d{2})?$', prev_line):
                # Check if the previous line has a total-related keyword
                if any(k in current_line for k in total_keywords):
                    total = prev_line
                    break

    # Extract receipt date if available
    date = ""
    items = []

    # Keywords to start/stop item parsing
    item_header_keywords = ['description', 'item', 'qty', 'quantity', 'price', 'amount', 'amt', 'mrp', 'rate']
    item_stop_keywords = ['total', 'subtotal', 'sub total', 'amount due', 'balance', 'total inclusive gst',
                          'total sales (inclusive of gst)', 'sub', 'cgst', 'sgst']

    parsing_items = False
    i = 0
    while i < len(result):
        text = result[i][1][0].strip()
        text_lower = text.lower()

        # Extract date from format like dd/mm/yyyy or similar
        if not date:
            match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
            if match:
                date = match.group(0)

        # Begin parsing item lines when keywords found
        if any(k in text_lower for k in item_header_keywords):
            parsing_items = True
            i += 1
            continue

        # Stop parsing when we reach totals or tax-related lines
        if any(k in text_lower for k in item_stop_keywords):
            parsing_items = False

        # If currently parsing items, extract them intelligently
        if parsing_items:
            current = result[i][1][0].strip()
            next_line = result[i + 1][1][0].strip() if i + 1 < len(result) else ""

            if not is_price(current):
                if is_price(next_line):
                    items.append(current)
                else:
                    if items:
                        items[-1] += ' ' + current  # Handle multiline item names
                    else:
                        items.append(current)
            i += 1
            continue
        i += 1

    # Predefined categories for classification
    categories = [
        "food", "medicine", "electronics", "household", "clothing", "beverage", "juice", "vegetable",
        "dairy", "sweets", "sauce", "fruits", "meat", "seafood", "salad", "soup", "herb"
    ]

    # Classify each item using zero-shot classification
    classified_items = []
    for item in items:
        result = classifier(item, candidate_labels=categories)
        top_category = result["labels"][0]
        classified_items.append({
            "item": item,
            "category": top_category
        })

    # Return final JSON structure
    receipt_json = {
        "store_name": store_name,
        "city": city,
        "date": date,
        "total": total,
        "items": classified_items
    }

    return receipt_json


# In[17]:

import oracledb
from datetime import datetime

def insert_receipt_to_oracle(receipt_data):
    """
        Inserts extracted receipt data into an Oracle DB.
        Arguments:
            receipt_data: dict - Structured receipt data containing metadata and items
        """
    try:
        #Oracle Instant Client is installed and configured if using thick mode)
        connection = oracledb.connect(
            user="system",
            password="user",
            dsn="localhost:1521/xe"  # Use service name instead of SID for oracledb
        )
        cursor = connection.cursor()
        
        # Parse the date to datetime object
        date_obj = datetime.strptime(receipt_data["date"], "%d/%m/%y")
        total = float(receipt_data["total"])

        # Insert main receipt info and get generated receipt ID
        receipt_id = cursor.var(int)
        cursor.execute("""
            INSERT INTO receipts (store_name, city, receipt_date, total)
            VALUES (:store_name, :city, :receipt_date, :total)
            RETURNING id INTO :receipt_id
        """, {
            "store_name": receipt_data["store_name"],
            "city": receipt_data.get("city", "Unknown"),
            "receipt_date": date_obj,
            "total": total,
            "receipt_id": receipt_id
        })

        receipt_id_value = int(receipt_id.getvalue()[0])

        # Insert item rows
        for item in receipt_data["items"]:
            cursor.execute("""
                INSERT INTO receipt_items (receipt_id, item_name, category)
                VALUES (:receipt_id, :item_name, :category)
            """, {
                "receipt_id": receipt_id_value,
                "item_name": item["item"],
                "category": item["category"]
            })

        connection.commit()
        print("Data inserted successfully.")

    except Exception as e:
        print("Error inserting data:", e)
        connection.rollback()
    finally:
        cursor.close()
        connection.close()


# In[18]:

import streamlit as st
import tempfile
import os

st.title("Receipt OCR & Categorizer")

# Upload image using Streamlit UI
uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Call the extract_image function to parse receipt
    with st.spinner("Processing image..."):
        result_json = extract_image(tmp_path)

    # Display the parsed results
    st.subheader("Parsed Receipt:")
    st.json(result_json)

    # Add button to insert results into the database
    if st.button("Insert to Database"):
        with st.spinner("Inserting into database..."):
            try:
                insert_receipt_to_oracle(result_json)
                st.success("Inserted into Oracle DB successfully!")
            except Exception as e:
                st.error(f"Failed to insert into database: {e}")
                
    # Delete the temporary file
    os.remove(tmp_path)
