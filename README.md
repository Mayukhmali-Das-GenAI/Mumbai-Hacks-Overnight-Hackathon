# Meeting Assistant Dashboard 🎙️📝

A Streamlit-based web application that helps teams automate the process of converting meeting recordings into actionable JIRA tasks. The app uses Deepgram for speech-to-text transcription and LLaMA for intelligent task extraction.

## Features

- 🎤 Audio file upload and transcription
- 👥 Speaker diarization (automatically identifies different speakers)
- 📝 Manual transcript input option
- 🤖 AI-powered action item extraction
- 📊 JIRA task generation with story points
- 💾 Export tasks as JSON

## Prerequisites

- Python 3.7+
- Deepgram API key
- Hugging Face API token
- Access to meta-llama/Llama-3.2-3B-Instruct model

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install streamlit sounddevice soundfile numpy deepgram-sdk torch transformers huggingface-hub
```

3. Set up your API keys:
   - Add your Deepgram API key in the `get_deepgram_client()` function
   - Add your Hugging Face token in the `HF_TOKEN` variable

## Usage

1. Run the Streamlit app:
```bash
streamlit run dashboard.py
```

2. Choose one of two options:
   - Upload an audio file (WAV or MP3)
   - Paste an existing meeting transcript

3. Process the input:
   - For audio files: Click "Transcribe Audio" to generate the transcript
   - Review the diarized transcript showing speaker segments

4. Generate JIRA tasks:
   - Click "Generate JIRA Tasks" to analyze the transcript
   - Review generated tasks with assignees, story points, and summaries
   - Download tasks as JSON for import into JIRA

## Task Generation

The app uses LLaMA to analyze transcripts and extract:
- Task assignees
- Story point estimates (1-8)
- JIRA-style task titles
- Actionable summaries
