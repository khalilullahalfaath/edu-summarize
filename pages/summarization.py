import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import cv2
from dotenv import load_dotenv
import os
import time
import random
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import openai
import pandas as pd
import plotly.express as px

# Load environment variables
load_dotenv(override=True)

# Configuration
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize services
text_analytics_client = TextAnalyticsClient(
    endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_API_KEY)
)
reader = easyocr.Reader(["id", "en"], gpu=False)

# quiz state
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "reset" not in st.session_state:
    st.session_state.reset = False


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result

    return wrapper


@timer_decorator
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


@timer_decorator
def extract_text_from_image(image):
    preprocessed_image = preprocess_image(image)
    result = reader.readtext(preprocessed_image)
    return " ".join([text for (_, text, _) in result])


@timer_decorator
def summarize_text(text, subject):
    try:
        poller = text_analytics_client.begin_extract_summary([text])
        extract_summary_results = poller.result()
        for result in extract_summary_results:
            if result.kind == "ExtractiveSummarization":
                summary = " ".join([sentence.text for sentence in result.sentences])
                return f"Summary of your {subject} notes: {summary}"
            elif result.is_error:
                return f"Error: {result.error.message}"
    except Exception as err:
        return f"An error occurred: {str(err)}"


@timer_decorator
def generate_quiz(notes, num_questions=10, language="English"):
    if language == "English":
        prompt = f"""
        Given the following text, generate {num_questions} questions in English that evaluate reading comprehension and language skills. 
        Include a balanced mix of the following question types:
        1. Main idea (MI)
        2. Detail (D)
        3. Vocabulary in context (V)
        4. Inference (I)
        5. Grammar (G)

        For each question, provide 4 options in English, with the first option being the correct answer. 
        Also, include the question type abbreviation at the end of each question.
        Format the output as follows:

        Question 1: [Question text] (Question type abbreviation)
        A. [Correct answer]
        B. [Incorrect option]
        C. [Incorrect option]
        D. [Incorrect option]

        Question 2: [Question text] (Question type abbreviation)
        ...

        Text:
        {notes}
        """
    else:
        prompt = f"""
        Berdasarkan teks berikut, buatlah {num_questions} pertanyaan dalam Bahasa Indonesia yang mengevaluasi pemahaman bacaan dan keterampilan bahasa. 
        Sertakan campuran seimbang dari jenis pertanyaan berikut:
        1. Ide Pokok (IP)
        2. Detail (D)
        3. Kosakata dalam Konteks (K)
        4. Inferensi (I)
        5. Tata Bahasa (TB)

        Untuk setiap pertanyaan, berikan 4 pilihan dalam Bahasa Indonesia, dengan pilihan pertama sebagai jawaban yang benar. 
        Juga, sertakan singkatan jenis pertanyaan di akhir setiap pertanyaan.
        Format keluarannya sebagai berikut:

        Pertanyaan 1: [Teks pertanyaan] (Singkatan jenis pertanyaan)
        A. [Jawaban benar]
        B. [Pilihan salah]
        C. [Pilihan salah]
        D. [Pilihan salah]

        Pertanyaan 2: [Teks pertanyaan] (Singkatan jenis pertanyaan)
        ...

        Teks:
        {notes}
        """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that generates reading comprehension and language skill questions in {language} based on given text.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.7,
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates reading comprehension and language skill questions based on given text.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    generated_quiz = response.choices[0].message.content.strip()

    # Parse the generated quiz
    questions = generated_quiz.split("\n\n")
    structured_quiz = []
    for i, question in enumerate(questions, 1):
        lines = question.split("\n")
        if len(lines) >= 5:  # Ensure we have a question and 4 options
            question_text, question_type = lines[0].split(": ", 1)[1].rsplit(" ", 1)
            question_type = question_type.strip("()")
            options = [line.split(". ", 1)[1] for line in lines[1:5]]
            correct_answer = options[0]
            random.shuffle(options)

            structured_quiz.append(
                {
                    "id": i,
                    "question": question_text,
                    "options": options,
                    "correct_answer": correct_answer,
                    "user_answer": None,
                    "question_type": question_type,
                }
            )

    return structured_quiz


def display_quiz_and_collect_answers():
    for q in st.session_state.quiz:
        st.write(f"Q{q['id']}: {q['question']}")
        options = q["options"]
        user_answer = st.radio("Your answer:", options, key=f"q{q['id']}")
        q["user_answer"] = user_answer
        st.write("---")


def calculate_score(quiz):
    correct_answers = sum(1 for q in quiz if q["user_answer"] == q["correct_answer"])
    total_questions = len(quiz)
    score = (correct_answers / total_questions) * 100
    return score, correct_answers, total_questions


def get_rating(percentage, language):
    if language == "English":
        ratings = [
            "Excellent (5/5)",
            "Strong (4/5)",
            "Good (3/5)",
            "Fair (2/5)",
            "Weak (1/5)",
            "Needs improvement (0/5)",
        ]
    else:
        ratings = [
            "Sangat Baik (5/5)",
            "Kuat (4/5)",
            "Baik (3/5)",
            "Cukup (2/5)",
            "Lemah (1/5)",
            "Perlu peningkatan (0/5)",
        ]

    if percentage == 100:
        return ratings[0]
    elif percentage >= 80:
        return ratings[1]
    elif percentage >= 60:
        return ratings[2]
    elif percentage >= 40:
        return ratings[3]
    elif percentage >= 20:
        return ratings[4]
    else:
        return ratings[5]


def generate_evaluation(quiz, language="English"):
    if language == "English":
        skill_categories = {
            "MI": "Main Idea Comprehension",
            "D": "Attention to Detail",
            "V": "Vocabulary",
            "I": "Inference Skills",
            "G": "Grammar",
        }
        evaluation_header = "Detailed Skill Evaluation:\n\n"
        overall_assessment = "\nOverall Assessment:\n"
        strengths_label = "Strengths: "
        weaknesses_label = "Areas for Improvement: "
    else:
        skill_categories = {
            "IP": "Pemahaman Ide Pokok",
            "D": "Perhatian terhadap Detail",
            "K": "Kosakata",
            "I": "Kemampuan Inferensi",
            "TB": "Tata Bahasa",
        }
        evaluation_header = "Evaluasi Keterampilan Terperinci:\n\n"
        overall_assessment = "\nPenilaian Keseluruhan:\n"
        strengths_label = "Kekuatan: "
        weaknesses_label = "Area yang Perlu Ditingkatkan: "

    skill_scores = {
        category: {"correct": 0, "total": 0} for category in skill_categories.values()
    }

    for q in quiz:
        category = skill_categories.get(q["question_type"], "Unknown")
        skill_scores[category]["total"] += 1
        if q.get("user_answer") == q.get("correct_answer"):
            skill_scores[category]["correct"] += 1

    evaluation = evaluation_header

    for category, scores in skill_scores.items():
        if scores["total"] > 0:
            percentage = (scores["correct"] / scores["total"]) * 100
            evaluation += (
                f"{category}: {percentage:.2f}% - {get_rating(percentage, language)}\n"
            )
        else:
            evaluation += f"{category}: N/A (no questions in this category)\n"

    evaluation += overall_assessment
    strengths = [
        category
        for category, scores in skill_scores.items()
        if scores["total"] > 0 and (scores["correct"] / scores["total"]) * 100 >= 80
    ]
    weaknesses = [
        category
        for category, scores in skill_scores.items()
        if scores["total"] > 0 and (scores["correct"] / scores["total"]) * 100 < 40
    ]

    if strengths:
        evaluation += strengths_label + ", ".join(strengths) + "\n"
    else:
        evaluation += strengths_label + "None identified\n"
    if weaknesses:
        evaluation += weaknesses_label + ", ".join(weaknesses) + "\n"
    else:
        evaluation += weaknesses_label + "None identified\n"

    # Calculate overall quantitative score
    total_correct = sum(scores["correct"] for scores in skill_scores.values())
    total_questions = sum(scores["total"] for scores in skill_scores.values())
    overall_score = (
        (total_correct / total_questions) * 100 if total_questions > 0 else 0
    )

    if language == "English":
        evaluation += f"\nOverall Quantitative Score: {overall_score:.2f}%\n"
    else:
        evaluation += f"\nSkor Kuantitatif Keseluruhan: {overall_score:.2f}%\n"

    return evaluation


def generate_flashcards(notes, language="English"):
    prompt = f"""
    Based on the following notes, create 10 flashcards. Each flashcard should have a question on one side and the answer on the other.
    Format the output as follows:

    1. Q: [Question]
       A: [Answer]

    2. Q: [Question]
       A: [Answer]

    ...

    Notes:
    {notes}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that creates educational flashcards in {language}.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
    )

    flashcards_text = response.choices[0].message.content.strip()
    flashcards = []

    for card in flashcards_text.split("\n\n"):
        parts = card.split("\n")
        if len(parts) >= 2:
            question = parts[0].split("Q: ", 1)[1] if "Q: " in parts[0] else parts[0]
            answer = parts[1].split("A: ", 1)[1] if "A: " in parts[1] else parts[1]
            flashcards.append({"question": question, "answer": answer})

    return flashcards


def show_summarization():
    st.title("NoteSum - Note Summarization and Quiz Generator")

    # Check if reset is requested
    if st.session_state.reset:
        # Clear all session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.reset = False
        st.experimental_rerun()

    if "quiz" not in st.session_state:
        st.session_state.quiz = None

    language = st.selectbox(
        "Select language / Pilih bahasa", ["English", "Bahasa Indonesia"]
    )

    input_method = st.radio(
        "Choose input method / Pilih metode input",
        (
            "Upload image for OCR / Unggah gambar untuk OCR",
            "Input text directly / Masukkan teks langsung",
        ),
    )
    subject = st.selectbox(
        "Select your subject / Pilih mata pelajaran",
        [
            "Math / Matematika",
            "Science / IPA",
            "History / Sejarah",
            "Literature / Sastra",
            "Other / Lainnya",
        ],
    )

    notes_text = ""
    if input_method == "Upload image for OCR / Unggah gambar untuk OCR":
        uploaded_file = st.file_uploader(
            "Choose your notes file / Pilih file catatan Anda",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption="Uploaded Image / Gambar yang Diunggah",
                use_column_width=True,
            )
            with st.spinner("Processing image... / Memproses gambar..."):
                notes_text = extract_text_from_image(np.array(image))
            st.write("Extracted Text / Teks yang Diekstrak:")
            st.text_area("", notes_text, height=200)
    else:
        notes_text = st.text_area(
            "Enter your notes or article text here / Masukkan catatan atau teks artikel Anda di sini:",
            height=300,
        )

    if notes_text:
        col1, col2, col3 = st.columns(3)

        summarize_button = col1.button("Summarize / Ringkas")
        quiz_button = col2.button("Generate Quiz / Hasilkan Kuis")
        flashcard_button = col3.button("Generate Flashcards / Hasilkan Kartu Flash")

        # Move summary generation and display outside of columns
        if summarize_button:
            with st.spinner("Generating summary... / Menghasilkan ringkasan..."):
                summary = summarize_text(notes_text, subject)
            st.write("Summary / Ringkasan:")
            st.text_area("", summary, height=200)
            st.download_button(
                label="Download summary / Unduh ringkasan",
                data=summary,
                file_name=f"{subject}_summary.txt",
                mime="text/plain",
            )

        if quiz_button:
            if notes_text.strip():
                with st.spinner("Generating quiz... / Menghasilkan kuis..."):
                    st.session_state.quiz = generate_quiz(notes_text, language=language)
                st.experimental_rerun()
            else:
                if language == "English":
                    st.warning(
                        "Please enter some text or upload an image with text content before generating a quiz."
                    )
                else:
                    st.warning(
                        "Silakan masukkan beberapa teks atau unggah gambar dengan konten teks sebelum menghasilkan kuis."
                    )

        if flashcard_button:
            if notes_text.strip():
                with st.spinner(
                    "Generating flashcards... / Menghasilkan kartu flash..."
                ):
                    st.session_state.flashcards = generate_flashcards(
                        notes_text, language
                    )
                st.success(
                    "Flashcards generated successfully! / Kartu flash berhasil dibuat!"
                )
            else:
                if language == "English":
                    st.warning(
                        "Please enter some text or upload an image with text content before generating flashcards."
                    )
                else:
                    st.warning(
                        "Silakan masukkan beberapa teks atau unggah gambar dengan konten teks sebelum menghasilkan kartu flash."
                    )

    if st.session_state.quiz:
        if language == "English":
            st.write("Please answer the following questions:")
        else:
            st.write("Silakan jawab pertanyaan berikut:")

        for i, q in enumerate(st.session_state.quiz):
            st.write(f"Q{i+1}: {q['question']}")
            options = q.get("options", [])
            if not options:
                st.error(
                    f"No options available for question {i+1}. Please regenerate the quiz."
                )
                continue

            key = f"question_{i}"
            user_answer = st.radio("Your answer:", options, key=key, index=0)
            q["user_answer"] = user_answer
            st.write("---")

        if st.button("Submit Answers / Kirim Jawaban"):
            st.session_state.quiz_submitted = True
            st.experimental_rerun()

    # Display flashcards if they exist
    if "flashcards" in st.session_state and st.session_state.flashcards:
        st.subheader("Flashcards / Kartu Flash")
        for i, card in enumerate(st.session_state.flashcards, 1):
            with st.expander(f"Flashcard {i}: {card['question']}"):
                st.write(f"Answer / Jawaban: {card['answer']}")

    if st.session_state.get("quiz_submitted", False):
        st.write("Quiz Results:")
        for i, q in enumerate(st.session_state.quiz):
            st.write(f"Q{i+1}: {q['question']}")
            st.write(f"Your answer: {q['user_answer']}")
            st.write(f"Correct answer: {q['correct_answer']}")
            if q["user_answer"] == q["correct_answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect")
            st.write("---")

        score, correct_answers, total_questions = calculate_score(st.session_state.quiz)
        if language == "English":
            st.write(f"Your overall score: {score:.2f}%")
            st.write(
                f"You got {correct_answers} out of {total_questions} questions correct."
            )
        else:
            st.write(f"Skor keseluruhan Anda: {score:.2f}%")
            st.write(
                f"Anda menjawab benar {correct_answers} dari {total_questions} pertanyaan."
            )

        evaluation = generate_evaluation(st.session_state.quiz, language)
        st.write(evaluation)

        # Create a dataframe for the skill scores
        skill_categories = (
            {
                "MI": "Main Idea Comprehension",
                "D": "Attention to Detail",
                "V": "Vocabulary",
                "I": "Inference Skills",
                "G": "Grammar",
            }
            if language == "English"
            else {
                "IP": "Pemahaman Ide Pokok",
                "D": "Perhatian terhadap Detail",
                "K": "Kosakata",
                "I": "Kemampuan Inferensi",
                "TB": "Tata Bahasa",
            }
        )

        skill_scores = {}
        for q in st.session_state.quiz:
            category = skill_categories[q["question_type"]]
            if category not in skill_scores:
                skill_scores[category] = {"correct": 0, "total": 0}
            skill_scores[category]["total"] += 1
            if q["user_answer"] == q["correct_answer"]:
                skill_scores[category]["correct"] += 1

        df_scores = pd.DataFrame(
            [
                {
                    "Skill": category,
                    "Score (%)": (
                        (scores["correct"] / scores["total"]) * 100
                        if scores["total"] > 0
                        else 0
                    ),
                }
                for category, scores in skill_scores.items()
            ]
        )

        # Create a bar chart of skill scores
        fig = px.bar(df_scores, x="Skill", y="Score (%)", title="Skill Performance")
        st.plotly_chart(fig)

        # Display qualitative evaluation
        st.subheader(
            "Qualitative Evaluation" if language == "English" else "Evaluasi Kualitatif"
        )
        strengths = [
            category
            for category, scores in skill_scores.items()
            if (scores["correct"] / scores["total"]) * 100 >= 80
        ]
        weaknesses = [
            category
            for category, scores in skill_scores.items()
            if (scores["correct"] / scores["total"]) * 100 < 40
        ]

        if language == "English":
            st.write(
                "Strengths:", ", ".join(strengths) if strengths else "None identified"
            )
            st.write(
                "Areas for Improvement:",
                ", ".join(weaknesses) if weaknesses else "None identified",
            )
        else:
            st.write(
                "Kekuatan:",
                ", ".join(strengths) if strengths else "Tidak teridentifikasi",
            )
            st.write(
                "Area yang Perlu Ditingkatkan:",
                ", ".join(weaknesses) if weaknesses else "Tidak teridentifikasi",
            )

        # Display quantitative evaluation
        st.subheader(
            "Quantitative Evaluation"
            if language == "English"
            else "Evaluasi Kuantitatif"
        )
        st.write(
            f"Overall Score: {score:.2f}%"
            if language == "English"
            else f"Skor Keseluruhan: {score:.2f}%"
        )

        # Provide recommendations based on the score
        if language == "English":
            if score >= 80:
                st.write("Excellent performance! Keep up the good work.")
            elif score >= 60:
                st.write(
                    "Good performance. Focus on improving your weaker areas to boost your score."
                )
            else:
                st.write(
                    "There's room for improvement. Consider reviewing the material and practicing more."
                )
        else:
            if score >= 80:
                st.write("Kinerja yang sangat baik! Pertahankan pekerjaan yang bagus.")
            elif score >= 60:
                st.write(
                    "Kinerja yang baik. Fokus pada peningkatan area yang lebih lemah untuk meningkatkan skor Anda."
                )
            else:
                st.write(
                    "Masih ada ruang untuk perbaikan. Pertimbangkan untuk meninjau materi dan berlatih lebih banyak."
                )

    # reset button
    if st.button("Reset / Ulang" if language == "Bahasa Indonesia" else "Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.reset = False
        st.experimental_rerun()

    # Display tips in the sidebar
    display_tips(language)


def display_tips(language="English"):
    if language == "English":
        st.sidebar.title("Tips for Better Summaries and Quizzes")
        st.sidebar.write(
            """
        1. Use clear handwriting or typed notes
        2. Include key terms and definitions
        3. Structure your notes with headings and subheadings
        4. Use bullet points for main ideas
        5. Review and edit your notes before uploading
        6. Highlight important concepts or formulas
        7. Include examples to illustrate key points
        8. Use concise language and avoid unnecessary details
        9. Organize information in a logical sequence
        10. Summarize each section in your own words
        """
        )


if __name__ == "__main__":
    show_summarization()
