import requests
import openai

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"


def fetch_transcript(api_url="http://localhost:5000/transcript"):
    """
    Fetch transcript JSON data from the Flask /transcript endpoint.

    Args:
        api_url (str): URL of the /transcript API.

    Returns:
        list: List of transcript entries containing speaker, start_time, end_time, transcription.
    """
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return []


def summarize_transcript(transcript_entries):
    """
    Summarize transcript content using OpenAI GPT.

    Args:
        transcript_entries (list): List of transcript dicts from the API.

    Returns:
        str: Generated summary text.
    """
    # Prepare the raw text input
    transcript_text = "\n".join(
        f"[{entry['start_time']} - {entry['end_time']}] {entry['speaker']}: {entry['transcription']}"
        for entry in transcript_entries
    )

    prompt = (
        "You are a helpful assistant. Summarize the following meeting transcript "
        "into a concise summary of key discussion points:\n\n"
        f"{transcript_text}\n\nSummary:"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You summarize meeting transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        summary = response["choices"][0]["message"]["content"].strip()
        return summary

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Error generating summary."


if __name__ == "__main__":
    # Example usage
    api_url = "http://localhost:5000/transcript"
    transcript_data = fetch_transcript(api_url)
    if transcript_data:
        summary = summarize_transcript(transcript_data)
        print("\n===== Generated Meeting Summary =====")
        print(summary)
    else:
        print("No transcript data available.")
