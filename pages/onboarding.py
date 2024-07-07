import streamlit as st
from utils.localizer import load_bundle
from PIL import Image
import os


def show_onboarding():
    lang_options = {"English (US)": "en_US", "Bahasa Indonesia (ID)": "id_ID"}

    # Language selection
    locale = st.selectbox("Choose your language", options=list(lang_options.keys()))

    lang_dict = load_bundle(lang_options[locale])

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "..", "static", "edu.png")

        if os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
        else:
            st.warning("Hero image not found. Please check the file path.")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

    # Main content
    st.title("CaDas (Catatan Cerdas) 📚✏️")

    if "sub_title" in lang_dict:
        st.subheader(lang_dict["sub_title"])
    else:
        st.warning("Subtitle key not found in language dictionary.")

    if "desc" in lang_dict:
        st.info(lang_dict["desc"])
    else:
        st.warning("Description key not found in language dictionary.")

    # Features and Benefits in a single column
    if "features_title" in lang_dict:
        st.markdown(f"### {lang_dict['features_title']}")
        features = [
            lang_dict.get("feature_1", "Feature 1 not found"),
            lang_dict.get("feature_2", "Feature 2 not found"),
            lang_dict.get("feature_3", "Feature 3 not found"),
            lang_dict.get("feature_4", "Feature 4 not found"),
        ]
        for feature in features:
            st.markdown(f"- {feature}")
    else:
        st.warning("Features title key not found in language dictionary.")

    if "benefits_title" in lang_dict:
        st.markdown(f"### {lang_dict['benefits_title']}")
        benefits = [
            lang_dict.get("benefits_1", "Benefit 1 not found"),
            lang_dict.get("benefits_2", "Benefit 2 not found"),
            lang_dict.get("benefits_3", "Benefit 3 not found"),
        ]
        for benefit in benefits:
            st.markdown(f"- {benefit}")
    else:
        st.warning("Benefits title key not found in language dictionary.")

    # Call to action
    st.markdown("---")

    if "ready" in lang_dict:
        st.markdown(f"## {lang_dict['ready']}")
    else:
        st.warning("Ready key not found in language dictionary.")

    # Centered button using Streamlit's layout
    if "start" in lang_dict:
        if st.button(lang_dict["start"], key="start_button"):
            st.session_state.page = "summarize"
    else:
        st.warning("Start key not found in language dictionary.")

    # FAQ Section
    st.markdown("---")

    if "faq" in lang_dict:
        with st.expander(lang_dict["faq"]):
            faq_questions = [
                "faq_1_q",
                "faq_2_q",
                "faq_3_q",
                "faq_4_q",
                "faq_5_q",
            ]
            faq_answers = [
                "faq_1_a",
                "faq_2_a",
                "faq_3_a",
                "faq_4_a",
                "faq_5_a",
            ]

            for q, a in zip(faq_questions, faq_answers):
                if q in lang_dict and a in lang_dict:
                    st.markdown(f"**{lang_dict[q]}**")
                    st.write(lang_dict[a])
                else:
                    st.warning(
                        f"FAQ key '{q}' or '{a}' not found in language dictionary."
                    )
    else:
        st.warning("FAQ key not found in language dictionary.")


if __name__ == "__main__":
    show_onboarding()
