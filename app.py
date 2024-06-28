import streamlit as st


def main():
    st.set_page_config(page_title="NoteSum - Your Study Companion", page_icon="📚")

    st.title("Welcome to NoteSum! 📚✏️")
    st.subheader("Your personal note summarization assistant")

    st.write(
        "NoteSum helps junior high students like you create concise summaries of your class notes."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.header("How it works:")
        st.write("1. Upload your notes")
        st.write("2. Choose the subject")
        st.write("3. Get your summary!")

    with col2:
        st.header("Benefits:")
        st.write("✅ Save time studying")
        st.write("✅ Understand key concepts better")
        st.write("✅ Improve your grades")

    st.header("Ready to get started?")
    if st.button("Create Your First Summary"):
        st.success("Great! Let's begin summarizing your notes.")
        # Here you would typically redirect to the main application page
        # Since Streamlit doesn't support direct redirects, you might use session state to change the app's mode

    st.markdown("---")
    st.write(
        "Have questions? Check out our [FAQ](https://www.notesum.com/faq) or [contact us](mailto:support@notesum.com)."
    )


if __name__ == "__main__":
    main()
