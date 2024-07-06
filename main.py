import streamlit as st
from pages.onboarding import show_onboarding
from pages import show_summarization


def main():
    st.set_page_config(page_title="NoteSum", page_icon="📚", layout="wide")

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "onboarding"

    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        if st.button("Home"):
            st.session_state.page = "onboarding"
        if st.button("Summarize Notes"):
            st.session_state.page = "summarize"

    # Display the appropriate page
    if st.session_state.page == "onboarding":
        show_onboarding()
    elif st.session_state.page == "summarize":
        show_summarization()


if __name__ == "__main__":
    main()
