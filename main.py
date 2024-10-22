import streamlit as st

# from pages.onboarding import show_onboarding
from pages import show_summarization
from utils.localizer import load_bundle
import os


def show_onboarding():
    # Custom CSS (same as before)
    st.markdown(
        """
    <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            padding: 15px 30px;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #45a049;
            color: white !important;
        }
        h1, h2, h3 {
            color: #3498DB;
        }
        .stExpander {
            background-color: #E3F2FD;
            border-radius: 10px;
        }
        .button-container {
            display: flex;
            justify-content: flex-start;
        }
        @media (max-width: 640px) {
            .button-container {
                justify-content: center;
            }
        }
        .hero-image {
            width: 100%;
            max-width: 300px;
            height: auto;
            margin: 0 auto;
            display: block;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    lang_options = {"English (US)": "en_US", "Bahasa Indonesia (ID)": "id_ID"}

    # Language selection
    locale = st.selectbox("Choose your language", options=list(lang_options.keys()))

    lang_dict = load_bundle(lang_options[locale])

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "static", "edu.png")

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


def main():
    st.set_page_config(page_title="CaDas", page_icon="📚", layout="wide")

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "onboarding"

    # Display the appropriate page
    if st.session_state.page == "onboarding":
        show_onboarding()
    elif st.session_state.page == "summarize":
        show_summarization()


if __name__ == "__main__":
    main()
