import streamlit as st
from utils.localizer import load_bundle


def show_onboarding():
    lang_options = {"English (US)": "en_US", "Bahasa Indonesia (ID)": "id_ID"}

    locale = st.radio(label="Language", options=list(lang_options.keys()))

    # ISO locale code from the lang_options dictionary.
    lang_dict = load_bundle(lang_options[locale])

    st.title(lang_dict["title"] + "📚✏️")
    st.subheader(lang_dict["sub_title"])

    st.write(lang_dict["desc"])

    col1, col2 = st.columns(2)

    with col1:
        st.header(lang_dict["steps_title"])
        st.write(lang_dict["steps_1"])
        st.write(lang_dict["steps_2"])
        st.write(lang_dict["steps_3"])

    with col2:
        st.header(lang_dict["benefits_title"])
        st.write(lang_dict["benefits_1"])
        st.write(lang_dict["benefits_2"])
        st.write(lang_dict["benefits_3"])

    st.header(lang_dict["ready"])
    if st.button(lang_dict["summary"]):
        st.session_state.page = "summarize"

    st.markdown("---")
    st.write(lang_dict["faq"])
