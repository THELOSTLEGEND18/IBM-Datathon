import os
import tempfile
import streamlit as st
from nude_detector import NudeDetector, VideoProcessor
import time

def save_uploaded_file(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.close()
    
    with open(temp_file.name, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    return temp_file.name

def main():
    st.title("Video Content Safety Processor")
    st.write("Upload a video to process and blur sensitive content")

    # Initialize session state for processed video
    if 'processed_video' not in st.session_state:
        st.session_state.processed_video = None

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None and st.session_state.processed_video is None:
        temp_input_path = None
        
        try:
            # Save uploaded file temporarily
            temp_input_path = save_uploaded_file(uploaded_file)

            # Get paths for model and blur rules
            base_dir = os.path.dirname(__file__)
            model_path = os.path.join(base_dir, "Models/best.onnx")
            blur_rules_path = os.path.join(base_dir, "blurexception.rule")
            
            # Check if required files exist
            if not os.path.exists(model_path):
                st.error(f"Model file not found at {model_path}. Please ensure the ONNX model is present in the Models directory.")
                return
                
            if not os.path.exists(blur_rules_path):
                st.warning(f"Blur rules file not found at {blur_rules_path}. Using default blur settings.")

            # Initialize detector with blur rules
            detector = NudeDetector(
                model_path=model_path,
                blur_rules_path=blur_rules_path
            )
            processor = VideoProcessor(detector)

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress):
                progress_bar.progress(int(progress))
                status_text.text(f"Processing: {int(progress)}%")

            # Process video
            with st.spinner('Processing video...'):
                processed_video_bytes = processor.process_video(
                    temp_input_path, 
                    progress_callback=update_progress
                )

            if processed_video_bytes:
                # Store processed video in session state
                st.session_state.processed_video = processed_video_bytes
            else:
                st.error("Failed to process video")

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        
        finally:
            # Cleanup temporary input file
            if temp_input_path and os.path.exists(temp_input_path):
                try:
                    time.sleep(0.1)  # Small delay before deletion
                    os.unlink(temp_input_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_input_path}: {e}")

    # If we have processed video in session state, show download button and video
    if st.session_state.processed_video is not None:
        # Create download button
        st.download_button(
            label="Download processed video",
            data=st.session_state.processed_video,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
        
        # Display video
        st.video(st.session_state.processed_video)

    # Add a clear button to reset the state
    if st.session_state.processed_video is not None:
        if st.button("Process Another Video"):
            st.session_state.processed_video = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()