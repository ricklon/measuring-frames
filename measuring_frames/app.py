import streamlit as st
import tempfile
import os
import subprocess
import json
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from ollama import Client
import base64
import io
import mimetypes
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Ensure OGG and OGV mimetypes are registered
mimetypes.add_type("video/ogg", ".ogg")
mimetypes.add_type("video/ogg", ".ogv")

OLLAMA_HOST = 'http://192.168.4.71:11434'

def get_available_models(client):
    try:
        response = client.list()
        return [model['name'] for model in response['models']]
    except Exception as e:
        logging.error(f"Error fetching models: {str(e)}")
        return []



def get_video_info(video_path):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Extract relevant information
        video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
        if video_stream:
            frame_count = int(video_stream.get('nb_frames', 0))
            duration = float(data['format']['duration'])
            return frame_count, duration
        else:
            logging.error("No video stream found in the file.")
            return 0, 0
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running ffprobe: {e}")
        return 0, 0
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing ffprobe output: {e}")
        return 0, 0
    except Exception as e:
        logging.error(f"Unexpected error in get_video_info: {e}")
        return 0, 0

def extract_frames(video_path, output_dir, interval):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps=1/{interval}',
        f'{output_dir}/frame_%04d.jpg'
    ]
    subprocess.run(cmd, check=True)
    
    logging.debug(f"Files in output directory: {os.listdir(output_dir)}")

def analyze_frame(client, image_path, prompt, model):
    try:
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()
        
        # Construct a more explicit prompt
        full_prompt = f"""Please analyze the following image and respond to this prompt: {prompt}

Remember to describe what you see in the image before answering the prompt. Your response should be directly related to the visual content of the image."""

        logging.debug(f"Sending prompt to model: {full_prompt}")

        response = client.generate(
            model=model,
            prompt=full_prompt,
            images=[image_content],
            stream=False
        )
        
        logging.debug(f"Received response from model: {response['response'][:100]}...") # Log the first 100 characters of the response
        
        return response['response']
    except Exception as e:
        logging.error(f"Error in analyze_frame: {str(e)}")
        return f"Error analyzing frame: {str(e)}"

def main():
    st.title("Video Frame Analysis with Multi-Model Ollama")

    client = Client(host=OLLAMA_HOST)

    available_models = get_available_models(client)
    selected_model = st.selectbox("Select Ollama model", available_models, index=0 if available_models else None)

    if not available_models:
        st.error("No models available. Please check your Ollama server.")
        return

    st.write(f"Using Ollama model: {selected_model}")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "ogg", "ogv"])
    
    if uploaded_file is not None:
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.total_frames = 0
            st.session_state.duration = 0
            st.session_state.estimated_frames = 0
            if 'frame_dir' in st.session_state:
                shutil.rmtree(st.session_state.frame_dir, ignore_errors=True)
            st.session_state.frame_dir = tempfile.mkdtemp()

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        mime_type = mimetypes.guess_type(uploaded_file.name)[0]

        if not mime_type or not mime_type.startswith('video'):
            st.error(f"Unsupported file type: {mime_type}. Please upload a video file.")
            return

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name

        # Get video information
        frame_count, duration = get_video_info(video_path)
        if frame_count == 0 and duration == 0:
            st.error("Failed to extract video information. Please check if the uploaded file is a valid video.")
            return

        st.session_state.total_frames = frame_count
        st.session_state.duration = duration

        st.write(f"Video duration: {st.session_state.duration:.2f} seconds")
        st.write(f"Total frames: {st.session_state.total_frames}")

        max_interval = min(60, max(1, int(st.session_state.duration)))
        
        def update_estimated_frames():
            st.session_state.estimated_frames = int(st.session_state.duration / st.session_state.interval)

        interval = st.slider(
            "Select frame interval (seconds)", 
            1, max_interval, 
            value=5,
            key='interval',
            on_change=update_estimated_frames
        )

        if 'estimated_frames' not in st.session_state:
            st.session_state.estimated_frames = int(st.session_state.duration / interval)

        st.write(f"Estimated frames to be analyzed: {st.session_state.estimated_frames}")

        prompt = st.text_input("Enter analysis prompt", "Describe any potential OSHA safety violations in this image.")

        if st.button("Extract Frames"):
            with st.spinner("Extracting frames..."):
                try:
                    extract_frames(video_path, st.session_state.frame_dir, interval)
                    st.success("Frames extracted successfully!")
                except subprocess.CalledProcessError:
                    st.error("Error extracting frames. Make sure the uploaded file is a valid video.")
                    return

        if st.button("Analyze Frames"):
            if not os.path.exists(st.session_state.frame_dir) or not os.listdir(st.session_state.frame_dir):
                st.error("No frames available. Please extract frames first.")
                return

            with st.spinner("Analyzing frames..."):
                frames = sorted([f for f in os.listdir(st.session_state.frame_dir) if f.endswith('.jpg')])
                logging.debug(f"Extracted frames: {frames}")
                
                results = []
                progress_bar = st.progress(0)
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(st.session_state.frame_dir, frame)
                    try:
                        analysis = analyze_frame(client, frame_path, prompt, selected_model)
                        results.append({
                            'frame': i,
                            'time': i * interval,
                            'analysis': analysis,
                            'filename': frame
                        })
                        st.write(f"Frame {i} analysis:")
                        st.image(Image.open(frame_path), width=300)
                        st.write(analysis)
                    except Exception as e:
                        st.error(f"Error analyzing frame {i}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(frames))
            
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="video_analysis.csv",
                mime="text/csv"
            )
            
            st.subheader("Analysis Results")
            for result in results:
                st.write(f"Frame {result['frame']} (Time: {result['time']}s):")
                try:
                    image_path = os.path.join(st.session_state.frame_dir, result['filename'])
                    if os.path.exists(image_path):
                        st.image(Image.open(image_path), width=300)
                    else:
                        st.warning(f"Image file not found: {image_path}")
                except Exception as e:
                    st.error(f"Error displaying image for frame {result['frame']}: {str(e)}")
                st.write(result['analysis'])
                st.write("---")
            
            st.subheader("Analysis Over Time")
            fig, ax = plt.subplots()
            ax.plot([r['time'] for r in results], range(len(results)))
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Frame Number")
            ax.set_title("Video Timeline")
            st.pyplot(fig)

        os.unlink(video_path)

if __name__ == "__main__":
    main()