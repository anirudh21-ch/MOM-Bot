import openai
import requests

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
        transcript_data = response.json()
        
        # Handle both direct array and wrapped response formats
        if isinstance(transcript_data, list):
            return transcript_data
        elif isinstance(transcript_data, dict) and "transcript" in transcript_data:
            return transcript_data["transcript"]
        else:
            print(f"Unexpected response format: {type(transcript_data)}")
            return transcript_data if isinstance(transcript_data, list) else []
            
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching transcript: {e}")
        return []
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
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=500,
        )

        summary = response["choices"][0]["message"]["content"].strip()
        return summary

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Error generating summary."


def fetch_summary(api_url="http://localhost:5000/summary"):
    """
    Fetch generated summary directly from the Flask /summary endpoint.

    Args:
        api_url (str): URL of the /summary API.

    Returns:
        str: Generated summary text, or error message if failed.
    """
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        summary_data = response.json()
        
        if isinstance(summary_data, dict) and "summary" in summary_data:
            return summary_data["summary"]
        elif isinstance(summary_data, dict) and "error" in summary_data:
            return f"Error: {summary_data['error']}"
        else:
            return str(summary_data)
            
    except requests.exceptions.RequestException as e:
        return f"Network error fetching summary: {e}"
    except Exception as e:
        return f"Error fetching summary: {e}"


if __name__ == "__main__":
    # Example usage - Method 1: Fetch transcript and generate summary locally
    print("Method 1: Fetch transcript and generate summary locally")
    transcript_data = fetch_transcript("http://localhost:5000/transcript")
    if transcript_data:
        summary = summarize_transcript(transcript_data)
        print("\n===== Generated Meeting Summary (Local) =====")
        print(summary)
    else:
        print("No transcript data available.")
    
    print("\n" + "="*60 + "\n")
    
    # Example usage - Method 2: Fetch pre-generated summary from API
    print("Method 2: Fetch pre-generated summary from API")
    api_summary = fetch_summary("http://localhost:5000/summary")
    print("\n===== API Generated Summary =====")
    print(api_summary)