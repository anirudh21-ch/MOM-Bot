"""MOM-Bot Streamlit Application

Interactive web interface for the speech processing pipeline.
Allows users to upload audio files and view transcription with speaker diarization.

Run: streamlit run src/ui/streamlit_app.py
"""

import streamlit as st
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline import MOMBotPipeline

# Page configuration
st.set_page_config(
    page_title="MOM-Bot - Meeting Transcription",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal styling
st.markdown("""
<style>
    .main { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎙️ MOM-Bot Meeting Transcription System")
st.markdown("""
**Automated speech processing pipeline for meetings:**
- 🎯 Voice Activity Detection (VAD)
- 📝 Automatic Transcription (Whisper)
- 👥 Speaker Diarization (Who spoke when)
- 🌐 Multi-language Support
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model = st.selectbox(
        "Whisper Model",
        options=['tiny', 'base', 'small', 'medium', 'large'],
        index=0,
        help="""
        **tiny** (39M): Fastest, ~1GB RAM - Good for 15-20 min audio
        **base** (74M): Balanced, ~2GB RAM - Recommended
        **small** (244M): Better quality, ~4GB RAM
        **medium** (769M): Excellent, ~8GB RAM
        **large** (1.5B): Best, ~10GB RAM (needs GPU)
        """
    )
    
    st.divider()
    
    st.caption("️ℹ️ Speakers are automatically detected from audio")
    
    # Display model info
    model_info = {
        'tiny': {'size': '39M', 'ram': '~1GB', 'speed': 'Very Fast'},
        'base': {'size': '74M', 'ram': '~2GB', 'speed': 'Fast'},
        'small': {'size': '244M', 'ram': '~4GB', 'speed': 'Moderate'},
        'medium': {'size': '769M', 'ram': '~8GB', 'speed': 'Slow'},
        'large': {'size': '1.5B', 'ram': '~10GB', 'speed': 'Very Slow'},
    }
    
    info = model_info[model]
    st.write(f"**Model Info** • Size: {info['size']} • RAM: {info['ram']} • Speed: {info['speed']}")


# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📋 Results", "📊 Statistics", "ℹ️ Help"])

# Tab 1: Upload & Process
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'],
            help="Maximum file size: 500MB"
        )
    
    with col2:
        st.subheader("File Info")
        if uploaded_file:
            st.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
            st.metric("Format", uploaded_file.name.split('.')[-1].upper())
    
    # Process button
    if uploaded_file:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            process_button = st.button("▶️ Process Audio", use_container_width=True, type="primary")
        
        with col2:
            st.write("")  # Spacer
        
        with col3:
            st.write("")  # Spacer
        
        if process_button:
            # Save uploaded file temporarily
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Progress tracking
            with st.spinner("🔄 Processing audio... This may take a moment."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Initialize pipeline
                    status_text.text("Initializing pipeline...")
                    progress_bar.progress(10)
                    
                    pipeline = MOMBotPipeline(
                        whisper_model_size=model
                    )
                    
                    # Process
                    status_text.text("Processing audio file...")
                    progress_bar.progress(30)
                    
                    results = pipeline.process(str(temp_path))
                    
                    progress_bar.progress(90)
                    
                    if 'error' in results:
                        st.error(f"❌ Processing failed: {results['error']}")
                    else:
                        # Save to session state
                        st.session_state.results = results
                        st.session_state.filename = uploaded_file.name
                        st.session_state.processed_time = datetime.now()
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")
                        
                        # Success message
                        st.success("Audio processed successfully!")
                        
                        # Summary
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Duration", f"{results.get('total_duration', 0):.1f}s")
                        with col2:
                            st.metric("Speakers", results.get('num_speakers', 0))
                        with col3:
                            st.metric("Segments", results.get('num_segments', 0))
                        with col4:
                            st.metric("Language", results.get('language', 'Unknown'))
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                
                finally:
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
            
            time.sleep(0.5)


# Tab 2: Results
with tab2:
    if 'results' in st.session_state:
        results = st.session_state.results
        
        st.subheader("📄 Transcription Results")
        if 'filename' in st.session_state:
            st.caption(f"File: {st.session_state.filename} | Processed: {st.session_state.processed_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Speaker summary
        st.subheader("👥 Speaker Summary")
        speaker_summary = results.get('speaker_summary', {})
        if speaker_summary:
            cols = st.columns(len(speaker_summary))
            for idx, (speaker, stats) in enumerate(speaker_summary.items()):
                with cols[idx]:
                    st.metric(speaker, f"{stats['duration']:.1f}s", f"{stats['segments']} segments")
        
        # Full transcription
        st.subheader("📝 Full Transcription")
        
        segments = results.get('segments', [])
        if segments:
            # Search/filter options
            col1, col2 = st.columns([3, 1])
            with col1:
                search_query = st.text_input("🔍 Search in transcription...")
            with col2:
                speakers_list = ["All"] + sorted(list(set(s.get('speaker', 'Unknown') for s in segments)))
                speaker_filter = st.selectbox("Filter by speaker", speakers_list)
            
            # Display segments
            segment_count = 0
            for segment in segments:
                speaker = segment.get('speaker', 'Unknown')
                text = segment.get('text', '').strip()
                start_time = segment.get('start_time', '')
                end_time = segment.get('end_time', '')
                confidence = segment.get('confidence', 0)
                
                # Apply filters
                if speaker_filter != "All" and speaker != speaker_filter:
                    continue
                if search_query and search_query.lower() not in text.lower():
                    continue
                
                segment_count += 1
                
                # Display segment
                st.write(f"**{speaker}** • {start_time} → {end_time}")
                st.write(text)
                st.caption(f"Confidence: {confidence:.2f}")
                st.divider()
            
            if segment_count == 0 and search_query:
                st.info("No segments match your search query.")
            elif segment_count == 0:
                st.info("No segments to display.")
        else:
            st.warning("No transcription segments found.")
        
        # Export options
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as JSON
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"{st.session_state.get('filename', 'results').split('.')[0]}_results.json",
                mime="application/json"
            )
        
        with col2:
            # Export as TXT
            txt_content = ""
            for segment in results.get('segments', []):
                txt_content += f"\n[{segment.get('start_time', '')}] {segment.get('speaker', '')}\n"
                txt_content += segment.get('text', '') + "\n"
            
            st.download_button(
                label="📥 Download as TXT",
                data=txt_content,
                file_name=f"{st.session_state.get('filename', 'results').split('.')[0]}_transcription.txt",
                mime="text/plain"
            )
    
    else:
        st.info("👆 Process an audio file in the **Upload & Process** tab to see results here.")


# Tab 3: Statistics
with tab3:
    if 'results' in st.session_state:
        results = st.session_state.results
        
        st.subheader("📊 Statistics")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_duration = results.get('total_duration', 0)
            st.metric("Total Duration", f"{total_duration:.1f}s")
        
        with col2:
            num_segments = results.get('num_segments', 0)
            st.metric("Total Segments", num_segments)
        
        with col3:
            total_words = sum(len(s.get('text', '').split()) for s in results.get('segments', []))
            st.metric("Total Words", total_words)
        
        with col4:
            avg_confidence = sum(s.get('confidence', 0) for s in results.get('segments', [])) / max(num_segments, 1)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        st.divider()
        
        # Speaker breakdown chart
        st.subheader("Speaker Breakdown")
        
        speaker_summary = results.get('speaker_summary', {})
        if speaker_summary:
            # Extract data for chart
            speakers_data = {speaker: stats['duration'] for speaker, stats in speaker_summary.items()}
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.bar_chart(speakers_data)
            
            with col2:
                st.write("**Speaker Duration Distribution**")
                total_duration = sum(speakers_data.values())
                for speaker, duration in sorted(speakers_data.items()):
                    percentage = (duration / total_duration * 100) if total_duration > 0 else 0
                    st.write(f"{speaker}: {duration:.1f}s ({percentage:.1f}%)")
        else:
            st.info("No speaker data available.")
    
    else:
        st.info("👆 Process an audio file to see statistics here.")


# Tab 4: Help
with tab4:
    st.subheader("❓ Frequently Asked Questions")
    
    with st.expander("What models should I use?"):
        st.markdown("""
        - **tiny** (39M): Best for 15-20 minute meetings, very fast
        - **base** (74M): Balanced speed/quality, recommended for most use cases
        - **small** (244M): Better accuracy, slower
        - **medium** (769M): High accuracy, requires 8GB RAM minimum
        - **large** (1.5B): Best accuracy, requires GPU
        """)
    
    with st.expander("How does speaker diarization work?"):
        st.markdown("""
        The system uses advanced hierarchical clustering with speaker embeddings:
        1. **VAD**: Detects speech regions
        2. **Embeddings**: Extracts speaker characteristics
        3. **Clustering**: Groups by speaker identity
        4. **Smoothing**: Cleans up transitions
        """)
    
    with st.expander("Supported audio formats?"):
        st.markdown("""
        - WAV (.wav)
        - MP3 (.mp3)
        - M4A (.m4a)
        - FLAC (.flac)
        - OGG (.ogg)
        - WebM (.webm)
        """)
    
    with st.expander("Performance expectations?"):
        st.markdown("""
        **Processing time** (on Mac M2):
        - tiny: 2-5 min audio in 20-30s
        - base: 2-5 min audio in 30-60s
        - small: 2-5 min audio in 1-2 min
        
        **Memory usage**:
        - tiny: ~1 GB
        - base: ~2 GB
        - small: ~4 GB
        """)
    
    with st.expander("How accurate is speaker detection?"):
        st.markdown("""
        Accuracy depends on:
        - **Audio quality**: Clear audio gives better results
        - **Number of speakers**: Works best for 2-4 speakers
        - **Speaker overlap**: Some inaccuracy when speakers overlap
        
        Expected accuracy: 85-95% for clear 2-speaker conversations
        """)
    
    st.divider()
    
    st.subheader("📚 Documentation")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Project Files:**
        - README.md - Project overview
        - DIARIZATION_IMPROVEMENT_PLAN.md - Technical details
        - DIARIZATION_IMPLEMENTATION.md - Implementation reference
        """)
    
    with col2:
        st.markdown("""
        **API Endpoints:**
        - GET / - Home
        - GET /models - Available models
        - POST /process - Process audio
        - GET /status - Service status
        """)
    
    st.info("💡 **Tip**: Use the API for batch processing or integration with other tools.")


# Footer
st.divider()
st.caption("MOM-Bot v1.0 | Speech Processing Pipeline | Powered by Whisper & NeMo")
