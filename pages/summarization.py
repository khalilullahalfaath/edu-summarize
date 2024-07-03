import streamlit as st
from io import StringIO
import easyocr
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os
import requests
import time
from requests.exceptions import RequestException
from openai import OpenAI


# load env varibles
load_dotenv()

# get credentials
endpoint = os.getenv("AZURE_ENDPOINT")
key = os.getenv("AZURE_API_KEY")

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)


# API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
# headers = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE_API_KEY')}"}

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# # hugging face
# Initialize the Inference API
# inference = InferenceApi(
#     "meta-llama/Llama-2-7b-chat-hf", token=os.getenv("HUGGING_FACE_API_KEY")
# )

# HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# API_URL = "https://api-inference.huggingface.co/models/gpt2-large"
# headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

# initializers
# initialize azure
text_analytics_client = TextAnalyticsClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)

# Initialize the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


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


def query_openai(prompt, max_retries=5, delay=10):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.7,
                top_p=0.9,
                n=1,
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print(f"An unexpected error occurred: {e}")
                return generate_quiz_locally(prompt)


def generate_quiz_locally(text):
    questions = [
        "Who are the main characters in the story?",
        "What was the Grasshopper doing during the summer?",
        "What was the Ant doing during the summer?",
        "What did the Grasshopper say to the Ant?",
        "What happened to the Grasshopper when winter came?",
    ]
    return "\n\n".join(
        [
            f"Q{i+1}: {q}\nA{i+1}: [Write your answer here]"
            for i, q in enumerate(questions)
        ]
    )


def generate_quiz(text, num_questions=5):
    if len(text.split()) < 20:
        return "The provided text is too short to generate a meaningful quiz. Please provide more content."

    prompt = f"""Based on the following text, create a quiz with {num_questions} questions.
    Format each question and answer pair as follows:
    Q1: [Question]
    A1: [Answer]
    Q2: [Question]
    A2: [Answer]
    Q3: [Question]
    A3: [Answer]
    Q4: [Question]
    A4: [Answer]
    Q5: [Question]
    A5: [Answer]

    Text: {text}

    Now, generate the quiz:
    """

    response = query_openai(prompt)

    # Extract questions and answers from the response
    lines = response.split("\n")
    quiz = ""
    for i in range(0, len(lines), 2):
        if (
            i + 1 < len(lines)
            and lines[i].startswith("Q")
            and lines[i + 1].startswith("A")
        ):
            quiz += f"{lines[i]}\n{lines[i+1]}\n\n"

    if not quiz:
        return (
            "Sorry, I couldn't generate a proper quiz. Here's a template:\n"
            + generate_quiz_locally(text)
        )

    return quiz


def generate_flashcards(text, num_cards=5):
    prompt = f"Generate {num_cards} flashcards based on the following text:\n\n{text}\n\nFormat: Front: Term or question\nBack: Definition or answer"
    return query_openai(prompt)


def show_summarization():
    st.title("NoteSum - Note Summarization and Quiz Generator")

    input_method = st.radio(
        "Choose input method", ("Upload image for OCR", "Input text directly")
    )

    subject = st.selectbox(
        "Select your subject", ["Math", "Science", "History", "Literature", "Other"]
    )

    notes_text = ""  # Initialize notes_text

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
            with st.spinner("Generating summary..."):
                summary = summarize_text(notes_text, subject)

            st.write("Summary:")
            st.text_area("Summary", summary, height=200)

            st.download_button(
                label="Download summary",
                data=summary,
                file_name=f"{subject}_summary.txt",
                mime="text/plain",
            )

        if st.button("Generate Quiz"):
            if not notes_text.strip():
                st.warning(
                    "Please enter some text or upload an image with text content before generating a quiz."
                )
            else:
                with st.spinner("Generating quiz..."):
                    quiz = generate_quiz(notes_text)
                    st.write("Generated Quiz:")
                    st.text_area("Quiz", quiz, height=300)
                    if "[Write your answer here]" in quiz:
                        st.warning(
                            "The AI couldn't generate a proper quiz. A basic quiz template has been provided. You may need to adjust the questions based on the text."
                        )

        if st.button("Generate Flashcards"):
            with st.spinner("Generating flashcards..."):
                flashcards = generate_flashcards(notes_text)
            st.write("Generated Flashcards:")
            st.text_area("Flashcards", flashcards, height=300)

    st.sidebar.title("Tips for Better Summaries and Quizzes")
    st.sidebar.write(
        """
    1. Use clear handwriting or typed notes
    2. Include key terms and definitions
    3. Structure your notes with headings
    4. Use bullet points for main ideas
    5. Review and edit your notes before uploading
    """
    )


# def show_summarization():
#     st.title("NoteSum - Note Summarization")

#     input_method = st.radio(
#         "Choose input method", ("Upload image for OCR", "Input text directly")
#     )

#     subject = st.selectbox(
#         "Select your subject", ["Math", "Science", "History", "Literature", "Other"]
#     )

#     notes_text = ""

# if input_method == "Upload image for OCR":
#     uploaded_file = st.file_uploader(
#         "Choose your notes file", type=["jpg", "jpeg", "png"]
#     )

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         image_np = np.array(image)

#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         with st.spinner("Processing image..."):
#             result = read_text(image_np)
#             notes_text = extract_text_from_result(result)

#         st.write("Extracted Text:")
#         st.text_area("", notes_text, height=200)

#     else:  # Input text directly
#         notes_text = st.text_area("Enter your notes or article text here:", height=300)

#     if notes_text:
#         if st.button("Summarize"):
#             summary = summarize_text(notes_text, subject)

#             st.write("Summary:")
#             st.text_area("", summary, height=200)

#             st.download_button(
#                 label="Download summary",
#                 data=summary,
#                 file_name=f"{subject}_summary.txt",
#                 mime="text/plain",
#             )

#     st.sidebar.title("Tips for Better Summaries")
#     st.sidebar.write(
#         """
#     1. Use clear handwriting or typed notes
#     2. Include key terms and definitions
#     3. Structure your notes with headings
#     4. Use bullet points for main ideas
#     5. Review and edit your notes before uploading
#     """
#     )


if __name__ == "__main__":
    show_summarization()
