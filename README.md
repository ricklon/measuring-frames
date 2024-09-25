# Measuring Frames

A Streamlit application for analyzing video frames using AI.

## Description

This project uses Streamlit to create a web interface for uploading videos, extracting frames at specified intervals, and analyzing each frame using an AI model via Ollama. It's designed to detect potential OSHA safety violations in video footage, but can be adapted for other video analysis tasks.

## Features

- Video upload support for MP4, AVI, MOV, OGG, and OGV formats
- Frame extraction at user-specified intervals
- AI analysis of each extracted frame using Ollama
- Dynamic estimation of frames to be analyzed
- Results displayed in-app and downloadable as CSV

## Prerequisites

- Python 3.7+
- FFmpeg
- Ollama server running with the llava-phi3 model

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/measuring-frames.git
   cd measuring-frames
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

## Usage

1. Ensure your Ollama server is running at the specified address in the code.

2. Run the Streamlit app:
   ```
   poetry run streamlit run app.py
   ```

3. Open a web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

4. Upload a video, set the frame interval, and click "Analyze Video" to start the process.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.