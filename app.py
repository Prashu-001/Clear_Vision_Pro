import streamlit as st
from PIL import Image
import time
import numpy as np
from utils.model_loader import load_wgan, load_srgan, load_srwgan
from utils.image_utils import preprocess_image, postprocess_image

# Page config
st.set_page_config(
    page_title="CLEAR-VISION | AI Image Restoration",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.header {
    background: linear-gradient(135deg, #6e8efb, #a777e3);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    text-align: center;
}
.feature-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}
.team-member {
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
    padding: 1rem;
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stImage {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("# CLEAR-VISION")
    st.markdown("---")
    page = st.radio(
        "Menu",
        ["Home", "Restore Images", "Model Info", "About Team"],
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <small>Powered by Streamlit</small><br>
        <small>© 2023 CLEAR-VISION</small>
    </div>
    """, unsafe_allow_html=True)

# Home Page
if page == "Home":
    st.markdown("""
    <div class="header">
        <h1>AI-Powered Image Restoration</h1>
        <p>Transform degraded images into high-quality visuals</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>✨ Key Features</h3>
            <ul>
                <li>Noise removal</li>
                <li>Super-resolution</li>
                <li>Artifact reduction</li>
                <li>Quality enhancement</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 AI Performance</h3>
            <p>Our models use advanced GANs and perceptual loss to restore details in your degraded images.</p>
        </div>
        """, unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1542744173-8e7e53415bb0?w=800&auto=format&fit=crop",
        use_container_width=True,
        caption="Before and After Comparison"
    )


# Restore Images Page with Model Integration
elif page == "Restore Images":
    st.markdown("""<div class="header"><h1>Image Restoration</h1>
        <p>Upload your degraded image for restoration</p></div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a degraded image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Original Image")
            st.image(uploaded_file, use_container_width=True)

        with col2:
            model_type = st.selectbox("Select Model", ["WGAN (Fast)", "SRGAN (Balanced)", "SRWGAN (High Quality)"])
            if st.button("Restore Image", use_container_width=True):
                with st.spinner("Processing..."):
                    from utils.image_utils import preprocess_image, postprocess_image, to_tensor
                    import numpy as np
                    from PIL import Image
                    import time

                    img = Image.open(uploaded_file).convert("RGB")

                    if model_type == "WGAN (Fast)":
                        model = load_wgan()
                        model_key = "WGAN"
                    elif model_type == "SRGAN (Balanced)":
                        model = load_srgan()
                        model_key = "SRGAN"
                    elif model_type == "SRWGAN (High Quality)":
                        model = load_srwgan()
                        model_key = "SRWGAN"

                    input_tensor = preprocess_image(img, model_key)
                    start_time = time.time()
                    output = model(input_tensor, training=False)
                    elapsed_time = time.time() - start_time

                    output = output.numpy()
                    output_img = ((output[0] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

                    st.session_state.restored_image = postprocess_image(output)
                    st.session_state.time_taken = f"{elapsed_time:.2f}s"
                    st.success("Restoration complete!")

        if 'restored_image' in st.session_state:
            st.markdown("---")
            st.markdown("### Restoration Results")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Restored Image")
                st.image(st.session_state.restored_image, use_container_width=True)
                st.download_button(
                    label="Download Image",
                    data=st.session_state.restored_image,
                    file_name=f"restored_{uploaded_file.name}",
                    mime="image/png"
                )
            with col2:
                st.markdown("#### Processing Time")
                time_taken = st.session_state.get("time_taken",None)
                if time_taken:
                    st.markdown(f"""
                    <div style="background: #f8faff; padding: 1rem; border-radius: 10px; text-align: center;">
                        <div>Time</div>
                        <b>{st.session_state.time_taken}</b>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("click 'Restore Image' to see the processing time.")



# About Team Page
elif page == "About Team":
    st.markdown("""
    <div class="header">
        <h1>Our Team</h1>
        <p>The people behind CLEAR-VISION</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Team Members</h3>
        <div class="team-member">
            <div style="width: 40px; height: 40px; background: #6e8efb; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; color: white; font-weight: bold;">P</div>
            <div><h4 style="margin: 0;">Prashu</h4></div>
        </div>
        <div class="team-member">
            <div style="width: 40px; height: 40px; background: #a777e3; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; color: white; font-weight: bold;">B</div>
            <div><h4 style="margin: 0;">Bidyut</h4></div>
        </div>
        <div class="team-member">
            <div style="width: 40px; height: 40px; background: #6e8efb; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; color: white; font-weight: bold;">V</div>
            <div><h4 style="margin: 0;">Varsha</h4></div>
        </div>
        <div class="team-member">
            <div style="width: 40px; height: 40px; background: #a777e3; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; color: white; font-weight: bold;">K</div>
            <div><h4 style="margin: 0;">Khushbu</h4></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Model Info Page
elif page == "Model Info":
    st.markdown("""
    <div class="header">
        <h1>Model Information</h1>
        <p>Choose the right model for your needs</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🚀 WGAN</h3>
            <p>Fast Wasserstein GAN for quick restorations</p>
            <p><strong>Best for:</strong> Real-time applications</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ SRGAN</h3>
            <p>Super-Resolution GAN for balanced quality</p>
            <p><strong>Best for:</strong> General image enhancement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🎨 SRWGAN</h3>
            <p>Super-Resolution Wasserstein GAN for highest quality</p>
            <p><strong>Best for:</strong> Final quality outputs</p>
        </div>
        """, unsafe_allow_html=True)
