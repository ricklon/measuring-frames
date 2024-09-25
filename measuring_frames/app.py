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

# Ensure OGG and OGV mimetypes are registered
mimetypes.add_type("video/ogg", ".ogg")
mimetypes.add_type("video/ogg", ".ogv")

OLLAMA_MODEL = 'llava-phi3:latest'

def get_video_info(video_path):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    
    # Extract relevant information
    video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
    if video_stream:
        frame_count = int(video_stream.get('nb_frames', 0))
        duration = float(data['format']['duration'])
        return frame_count, duration
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

def analyze_frame(client, image_path, prompt):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    response = client.generate(model=OLLAMA_MODEL, prompt=f"{prompt}\n[img]{image_base64}[/img]")
    return response['response']

def main():
    st.title("Video Frame Analysis with AI")

    # Initialize Ollama client with the correct URI
    client = Client(host='http://192.168.4.71:11434')

    st.write(f"Using Ollama model: {OLLAMA_MODEL}")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "ogg", "ogv"])
    
    if uploaded_file is not None:
        # Check if a new file has been uploaded
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            
            # Reset session state for new file
            st.session_state.total_frames = 0
            st.session_state.duration = 0
            st.session_state.estimated_frames = 0

        # Determine the file extension and mime type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        mime_type = mimetypes.guess_type(uploaded_file.name)[0]

        # Check if the file is a video
        if not mime_type or not mime_type.startswith('video'):
            st.error(f"Unsupported file type: {mime_type}. Please upload a video file.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # Get video information if not already in session state
        if st.session_state.total_frames == 0 or st.session_state.duration == 0:
            frame_count, duration = get_video_info(video_path)
            st.session_state.total_frames = frame_count
            st.session_state.duration = duration

        st.write(f"Video duration: {st.session_state.duration:.2f} seconds")

        max_interval = min(60, max(1, int(st.session_state.duration)))
        
        # Function to update estimated frames
        def update_estimated_frames():
            st.session_state.estimated_frames = int(st.session_state.duration / st.session_state.interval)

        # Slider for interval selection
        interval = st.slider(
            "Select frame interval (seconds)", 
            1, max_interval, 
            value=5,
            key='interval',
            on_change=update_estimated_frames
        )

        # Initialize estimated_frames if not in session state
        if 'estimated_frames' not in st.session_state:
            st.session_state.estimated_frames = int(st.session_state.duration / interval)

        # Display estimated frames
        st.write(f"Estimated frames to be analyzed: {st.session_state.estimated_frames}")

        prompt = st.text_input("Enter analysis prompt", "Describe any potential OSHA safety violations in this image.")

        if st.button("Analyze Video"):
            with tempfile.TemporaryDirectory() as output_dir:
                with st.spinner("Extracting and analyzing frames..."):
                    try:
                        extract_frames(video_path, output_dir, interval)
                    except subprocess.CalledProcessError:
                        st.error("Error extracting frames. Make sure the uploaded file is a valid video.")
                        return
                    
                    frames = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
                    
                    results = []
                    progress_bar = st.progress(0)
                    for i, frame in enumerate(frames):
                        frame_path = os.path.join(output_dir, frame)
                        try:
                            analysis = analyze_frame(client, frame_path, prompt)
                            results.append({
                                'frame': i,
                                'time': i * interval,
                                'analysis': analysis
                            })
                        except Exception as e:
                            st.error(f"Error analyzing frame {i}: {str(e)}")
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(frames))
                
                # Create DataFrame and save to CSV
                df = pd.DataFrame(results)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="video_analysis.csv",
                    mime="text/csv"
                )
                
                # Display results
                st.subheader("Analysis Results")
                for result in results:
                    st.write(f"Frame {result['frame']} (Time: {result['time']}s):")
                    st.image(Image.open(os.path.join(output_dir, f"frame_{result['frame']:04d}.jpg")), width=300)
                    st.write(result['analysis'])
                    st.write("---")
                
                # Plot analysis over time
                st.subheader("Analysis Over Time")
                fig, ax = plt.subplots()
                ax.plot([r['time'] for r in results], range(len(results)))
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Frame Number")
                ax.set_title("Video Timeline")
                st.pyplot(fig)

        # Clean up the temporary video file
        os.unlink(video_path)

if __name__ == "__main__":
    main()