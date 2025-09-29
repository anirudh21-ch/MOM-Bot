import os
import pprint
import re

from flask import Flask, jsonify

from src.audio_processing import run_asr, load_config
from src.diarization import get_transcript, init_diarization, run_diarization, save_transcript
from src.manifest import create_manifest
from src.openai_integration import summarize_transcript
from src.preprocessing import prepare_audio
from src.utils import read_file

# Configuration
USE_WORD_LEVEL = False  # True for word-level diarization
DATA_DIR = "data"
ORIGINAL_AUDIO_PATH = "data/input_audio.wav"
BASE_NAME = "mono_audio"
EXTRACT_PATH = os.path.join(DATA_DIR, "extract.txt")

pp = pprint.PrettyPrinter(indent=4)

# Global caches
cached_transcript = None
cached_summary = None


def parse_transcript(file_path):
    results = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"\[(\d+:\d+\.\d+) - (\d+:\d+\.\d+)\]\s+(\w+):\s+(.*)", line)
            if match:
                start, end, speaker, text = match.groups()
                results.append(
                    {
                        "speaker": speaker,
                        "start_time": start,
                        "end_time": end,
                        "transcription": text,
                    }
                )
    return results


def run_pipeline():
    global cached_transcript

    print("Running ASR + Diarization pipeline...")

    # 1️⃣ Preprocess: Convert stereo to mono
    mono_audio_path, _, _ = prepare_audio(ORIGINAL_AUDIO_PATH, output_dir=DATA_DIR)
    print(f"Mono audio saved at: {mono_audio_path}")

    # 2️⃣ Load config
    cfg = load_config(data_dir=DATA_DIR, domain_type="meeting")

    # 3️⃣ Create manifest
    manifest_path = create_manifest(mono_audio_path, output_dir=DATA_DIR)
    cfg.diarizer.manifest_filepath = manifest_path

    # 4️⃣ Run ASR
    sentence_hyp, sentence_ts_hyp = run_asr(cfg, level="sentence")
    print(f"ASR hypotheses (sample): {sentence_hyp[:2]}...")

    # 5️⃣ Initialize diarization
    asr_diar_offline = init_diarization(cfg)

    # 6️⃣ Run diarization
    diar_hyp, diar_score, asr_diar_offline = run_diarization(cfg, sentence_ts_hyp, asr_diar_offline)
    print("Available diar_hyp keys:", diar_hyp.keys())
    first_key = list(diar_hyp.keys())[0]
    print(f"Diarization hypothesis sample for {first_key}:\n", diar_hyp[first_key])

    # 7️⃣ Extract transcript (as DataFrame for ease of use)
    transcript_df = get_transcript(
        asr_diar_offline, diar_hyp, sentence_hyp, sentence_ts_hyp, return_df=True
    )
    print("Transcript DataFrame sample:")
    print(transcript_df.head())

    # 8️⃣ Save transcript as extract.txt
    save_transcript(base_name=BASE_NAME, data_dir=DATA_DIR)

    # 9️⃣ Parse saved transcript for API serving
    parsed_transcript = parse_transcript(EXTRACT_PATH)
    cached_transcript = parsed_transcript
    return cached_transcript


# Run pipeline ONCE at startup
print("Initializing and executing pipeline...")
run_pipeline()

# Initialize Flask API
app = Flask(__name__)


@app.route("/transcript", methods=["GET"])
def get_transcript_api():
    return jsonify(cached_transcript)


@app.route("/summary", methods=["GET"])
def get_summary_api():
    global cached_summary
    if cached_summary is None:
        print("Generating summary from cached transcript using OpenAI...")
        cached_summary = summarize_transcript(cached_transcript)
    return jsonify({"summary": cached_summary})


if __name__ == "__main__":
    print("Starting Flask API on port 5000...")
    app.run(port=5000)
