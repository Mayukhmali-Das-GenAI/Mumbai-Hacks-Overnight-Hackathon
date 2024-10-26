import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from deepgram import DeepgramClient, PrerecordedOptions
import tempfile
from datetime import datetime
import time
import wave
import json
import torch
from transformers import pipeline
from huggingface_hub import login
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
HF_TOKEN = ""

st.set_page_config(
    page_title="Meeting Assistant Dashboard",
    page_icon="📝",
    layout="wide"
)

if 'audio_file' not in st.session_state:
    st.session_state['audio_file'] = None
if 'transcription' not in st.session_state:
    st.session_state['transcription'] = None
if 'raw_response' not in st.session_state:
    st.session_state['raw_response'] = None
if 'transcript_text' not in st.session_state:
    st.session_state['transcript_text'] = None
if 'diarized_text' not in st.session_state:
    st.session_state['diarized_text'] = ""

def get_deepgram_client():
    api_key = ""
    return DeepgramClient(api_key)

def setup_huggingface_auth():
    try:
        login(token=HF_TOKEN)
        logger.info("Successfully authenticated with Hugging Face")
        return True
    except Exception as e:
        logger.error(f"Error authenticating with Hugging Face: {e}")
        st.error("Failed to authenticate with Hugging Face. Please check your token.")
        return False

def load_llama_model():
    try:
        if not setup_huggingface_auth():
            return None

        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        return pipe
    except Exception as e:
        logger.error(f"Error loading LLaMA model: {e}")
        st.error(f"Failed to load LLaMA model: {str(e)}")
        return None

def process_diarized_transcript(response_dict):
    try:
        words = response_dict["results"]["channels"][0]["alternatives"][0]["words"]
        current_speaker = None
        diarized_text = ""
        current_text = []

        for word in words:
            speaker = word.get("speaker", "Unknown")
            if speaker != current_speaker:
                if current_text:
                    diarized_text += f"Speaker {current_speaker}: {' '.join(current_text)}\n"
                current_speaker = speaker
                current_text = []
            current_text.append(word["punctuated_word"])

        if current_text:
            diarized_text += f"Speaker {current_speaker}: {' '.join(current_text)}\n"

        return diarized_text
    except Exception as e:
        logger.error(f"Error processing diarized transcript: {e}")
        return None

def transcribe_audio(file_path):
    try:
        client = get_deepgram_client()
        
        with open(file_path, 'rb') as audio:
            source = {'buffer': audio, 'mimetype': 'audio/wav'}
            
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                diarize=True
            )
            
            response = client.listen.prerecorded.v("1").transcribe_file(source, options)
            response_dict = response.to_dict()
            st.session_state['raw_response'] = response_dict
            
            diarized_text = process_diarized_transcript(response_dict)
            if diarized_text:
                st.session_state['diarized_text'] = diarized_text
            
            return response
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def generate_action_items(text, temperature=0.7, max_length=None):
    try:
        pipe = load_llama_model()
        if not pipe:
            return None

        prompt = f"""Analyze this meeting transcript and identify action items assigned to people. Look for contexts like "X will do this" or "Y needs to handle that" or "We need Z to work on".

Meeting Transcript:
{text}

Generate a JSON array of tasks. For each action item mentioned in the conversation, create an object with this structure:
{{
    "assignee": "Person's name who will do the task",
    "story_points": number between 1-8 based on complexity,
    "title": "Clear JIRA task title",
    "summary": "Brief actionable summary"
}}

Important:
- Only include tasks that have clear ownners mentioned in the conversation
- Ensure the output is valid JSON format
- Story points should reflect task complexity. 1 Story point means 1 day. Be a bit strict in giving story points.
- Make titles and summaries actionable and specific

Extract tasks even if they're mentioned informally, like "Maybe John could look into the database issue" or "We should ask Sarah to review the design"."""
            

        if not max_length:
            max_length = min(len(text) + 1000, 4096)
            
        response = pipe(
            prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=1
        )
        
        generated_text = response[0]['generated_text']
        json_start = generated_text.find('[')
        json_end = generated_text.rfind(']') + 1
        if json_start == -1 or json_end == 0:
            st.error("Failed to generate properly formatted JSON output")
            return None
            
        json_str = generated_text[json_start:json_end]
        try:
            tasks = json.loads(json_str)
            return tasks
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse generated JSON: {str(e)}")
            return None
        
    except Exception as e:
        logger.error(f"Error generating action items: {e}")
        st.error(f"Failed to generate action items: {str(e)}")
        return None

# Main UI
st.title("📝 Meeting Assistant Dashboard")
st.write("Upload audio recordings or paste meeting transcripts to generate JIRA tasks")

tab1, tab2 = st.tabs(["Upload Audio", "Paste Transcript"])

with tab1:
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])
    if uploaded_file:
        temp_audio_path = tempfile.mktemp(suffix='.wav')
        with open(temp_audio_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.audio_file = temp_audio_path
        st.success("File uploaded successfully!")
        st.audio(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Transcribe Audio", key="transcribe_btn"):
                with st.spinner("Transcribing audio..."):
                    response = transcribe_audio(st.session_state.audio_file)
                    if response:
                        st.session_state.transcription = response
                        
        if st.session_state.get('diarized_text'):
            st.subheader("Transcription Result")
            st.text_area("Diarized Transcript", st.session_state['diarized_text'], height=200)
            
            if st.button("Generate JIRA Tasks", key="generate_jira_1"):
                with st.spinner("Generating JIRA tasks..."):
                    tasks = generate_action_items(st.session_state['diarized_text'])
                    if tasks:
                        st.session_state['tasks'] = tasks

with tab2:
    pasted_transcript = st.text_area("Paste your meeting transcript here", height=200)
    if pasted_transcript and st.button("Generate JIRA Tasks", key="generate_jira_2"):
        with st.spinner("Generating JIRA tasks..."):
            tasks = generate_action_items(pasted_transcript)
            if tasks:
                st.session_state['tasks'] = tasks

if 'tasks' in st.session_state and st.session_state['tasks']:
    st.subheader("Generated JIRA Tasks")
    for task in st.session_state['tasks']:
        with st.expander(f"Task for {task['assignee']}: {task['title']}"):
            st.write(f"**Story Points:** {task['story_points']}")
            st.write(f"**Summary:** {task['summary']}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="Download JIRA Tasks JSON",
        data=json.dumps(st.session_state['tasks'], indent=2),
        file_name=f"jira_tasks_{timestamp}.json",
        mime="application/json"
    )

if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
    try:
        os.remove(st.session_state.audio_file)
    except:
        pass