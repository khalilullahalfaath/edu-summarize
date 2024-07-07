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
    st.subheader(lang_dict["sub_title"])

    # Description in a box
    st.info(lang_dict["desc"])

    # Features and Benefits in a single column
    st.markdown(f"### {lang_dict['features_title']}")
    features = [
        lang_dict["feature_1"],
        lang_dict["feature_2"],
        lang_dict["feature_3"],
        lang_dict["feature_4"],
    ]
    for feature in features:
        st.markdown(f"- {feature}")

    st.markdown(f"### {lang_dict['benefits_title']}")
    benefits = [
        lang_dict["benefits_1"],
        lang_dict["benefits_2"],
        lang_dict["benefits_3"],
    ]
    for benefit in benefits:
        st.markdown(f"- {benefit}")

    # Call to action
    st.markdown("---")
    st.markdown(f"## {lang_dict['ready']}")

    # Centered button using Streamlit's layout
    if st.button(lang_dict["start"], key="start_button"):
        st.session_state.page = "summarize"

    # FAQ Section
    st.markdown("---")
    with st.expander(lang_dict["faq"]):
        st.markdown(f"**{lang_dict['faq_1_q']}**")
        st.write(lang_dict["faq_1_a"])

        st.markdown(f"**{lang_dict['faq_2_q']}**")
        st.write(lang_dict["faq_2_a"])

        st.markdown(f"**{lang_dict['faq_3_q']}**")
        st.write(lang_dict["faq_3_a"])

        st.markdown(f"**{lang_dict['faq_4_q']}**")
        st.write(lang_dict["faq_4_a"])

        st.markdown(f"**{lang_dict['faq_5_q']}**")
        st.write(lang_dict["faq_5_a"])


if __name__ == "__main__":
    show_onboarding()
