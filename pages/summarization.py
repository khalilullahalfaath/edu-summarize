import streamlit as st
from io import StringIO
import easyocr
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os

# load env varibles
load_dotenv()

# get credentials
endpoint = os.getenv("AZURE_ENDPOINT")
key = os.getenv("AZURE_API_KEY")

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

text_analytics_client = TextAnalyticsClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)


# Initialize the OCR reader
reader = easyocr.Reader(["id", "en"], gpu=False)  # 'id' : indonesian


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return gray


def read_text(image):
    preprocessed_image = preprocess_image(image)
    result = reader.readtext(preprocessed_image)
    return result


def extract_text_from_result(result):
    return " ".join([text for (bbox, text, prob) in result])


def summarize_text(text, subject):
    try:
        document = [text]
        poller = text_analytics_client.begin_extract_summary(document)
        extract_summary_results = poller.result()
        for result in extract_summary_results:
            if result.kind == "ExtractiveSummarization":
                summary = " ".join([sentence.text for sentence in result.sentences])
                return f"Summary of your {subject} notes: {summary}"
            elif result.is_error is True:
                return f"Error: {result.error.message}"
    except Exception as err:
        return f"An error occurred: {str(err)}"


def show_summarization():
    st.title("NoteSum - Note Summarization")

    input_method = st.radio(
        "Choose input method", ("Upload image for OCR", "Input text directly")
    )

    subject = st.selectbox(
        "Select your subject", ["Math", "Science", "History", "Literature", "Other"]
    )

    notes_text = ""

    if input_method == "Upload image for OCR":
        uploaded_file = st.file_uploader(
            "Choose your notes file", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Processing image..."):
                result = read_text(image_np)
                notes_text = extract_text_from_result(result)

            st.write("Extracted Text:")
            st.text_area("", notes_text, height=200)

    else:  # Input text directly
        notes_text = st.text_area("Enter your notes or article text here:", height=300)

    if notes_text:
        if st.button("Summarize"):
            summary = summarize_text(notes_text, subject)

            st.write("Summary:")
            st.text_area("", summary, height=200)

            st.download_button(
                label="Download summary",
                data=summary,
                file_name=f"{subject}_summary.txt",
                mime="text/plain",
            )

    st.sidebar.title("Tips for Better Summaries")
    st.sidebar.write(
        """
    1. Use clear handwriting or typed notes
    2. Include key terms and definitions
    3. Structure your notes with headings
    4. Use bullet points for main ideas
    5. Review and edit your notes before uploading
    """
    )


if __name__ == "__main__":
    show_summarization()
