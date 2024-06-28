import streamlit as st
from io import StringIO


def summarize_text(text, subject):
    return f"This is a summary of your {subject} notes: " + text[:100] + "..."

def show_summarization():
    st.title("NoteSum - Note Summarization")

    uploaded_file = st.file_uploader("Choose your notes file", type=["txt", "pdf"])

    subject = st.selectbox(
        "Select your subject", ["Math", "Science", "History", "Literature", "Other"]
    )

    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            notes_text = stringio.read()
        else:
            st.error(
                "PDF processing not implemented in this example. Please upload a txt file."
            )
            return

        st.write("Original Notes:")
        st.text_area("", notes_text, height=200)

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
