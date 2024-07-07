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
            st.markdown('<div class="hero-image-container">', unsafe_allow_html=True)
            st.image(image_path, use_column_width=False, output_format="PNG", width=300)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Hero image not found. Please check the file path.")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

    # Main content
    st.title(f"CaDas (Catatan Cerdas) 📚✏️")
    st.subheader(lang_dict["sub_title"])

    # Description in a colored box
    st.markdown(
        f"""
    <div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px;">
        <p>{lang_dict["desc"]}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Features in two columns

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {lang_dict['features_title']}")
        features = [
            lang_dict["feature_1"],
            lang_dict["feature_2"],
            lang_dict["feature_3"],
            lang_dict["feature_4"],
        ]
        for feature in features:
            st.markdown(f"- {feature}")

    with col2:
        st.markdown(f"### {lang_dict['benefits_title']}")
        benefits = [
            lang_dict["benefits_1"],
            lang_dict["benefits_2"],
            lang_dict["benefits_3"],
        ]
        for benefit in benefits:
            st.markdown(f"{benefit}")

    # Call to action
    st.markdown("---")
    st.markdown(f"## {lang_dict['ready']}")

    # Button with custom alignment
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button(lang_dict["start"], key="start_button"):
        st.session_state.page = "summarize"
    st.markdown("</div>", unsafe_allow_html=True)

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
