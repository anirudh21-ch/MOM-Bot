"""
OpenAI integration for transcript summarization and analysis.
"""

import os
import logging
from typing import List, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = None

def initialize_openai():
    """Initialize OpenAI client with API key from environment."""
    global client
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully")


def format_transcript_for_summary(transcript: List[Dict[str, Any]]) -> str:
    """
    Format the transcript data for OpenAI summarization.
    
    Args:
        transcript: List of transcript segments with speaker, start_time, end_time, and text
        
    Returns:
        Formatted transcript string
    """
    if not transcript:
        return "No transcript available."
    
    formatted_lines = []
    formatted_lines.append("=== MEETING TRANSCRIPT ===\n")
    
    for segment in transcript:
        speaker = segment.get('speaker', 'Unknown')
        # Try both field names for compatibility
        text = segment.get('text', segment.get('transcription', '')).strip()
        start_time = segment.get('start_time', 0)
        
        if text:
            # Format time in minutes:seconds
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            formatted_lines.append(f"[{time_str}] {speaker}: {text}")
    
    return "\n".join(formatted_lines)


def summarize_transcript(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the meeting transcript using OpenAI.
    
    Args:
        transcript: List of transcript segments
        
    Returns:
        Dictionary containing summary, key points, action items, and participants
    """
    global client
    
    if not client:
        initialize_openai()
    
    # Format transcript
    formatted_transcript = format_transcript_for_summary(transcript)
    
    # Create the prompt for summarization
    system_prompt = """You are an expert meeting assistant that creates comprehensive summaries from meeting transcripts. 

Your task is to analyze the provided meeting transcript and create a structured summary.

You MUST respond with a valid JSON object using exactly these keys:
- executive_summary: A concise overview of the meeting's purpose and main outcomes
- key_points: Array of main topics discussed with important details
- decisions: Array of clear decisions that were reached during the meeting
- action_items: Array of specific tasks assigned to individuals with deadlines if mentioned
- participants: Array of speakers who participated in the meeting
- next_steps: Array of any follow-up meetings, deadlines, or processes mentioned

IMPORTANT: Your response must be valid JSON format only. Do not include any text before or after the JSON object.

Example format:
{
  "executive_summary": "This meeting focused on...",
  "key_points": ["Topic 1 discussed...", "Topic 2 covered..."],
  "decisions": ["Decision 1...", "Decision 2..."],
  "action_items": ["John to complete task by Friday"],
  "participants": ["SPEAKER_0", "SPEAKER_1"],
  "next_steps": ["Follow-up meeting scheduled"]
}"""

    user_prompt = f"""Analyze this meeting transcript and provide a comprehensive summary in JSON format:

{formatted_transcript}

Respond with valid JSON only, following the exact structure specified in the system prompt."""

    try:
        # Make API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using cost-effective model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent, focused summaries
            max_tokens=1500,  # Reasonable limit for comprehensive summaries
        )
        
        # Extract the response content
        summary_text = response.choices[0].message.content
        
        # Try to parse as JSON, fallback to structured text if needed
        try:
            import json
            # Clean the response text (remove any markdown formatting)
            clean_text = summary_text.strip()
            if clean_text.startswith('```json'):
                clean_text = clean_text.replace('```json', '').replace('```', '').strip()
            
            summary_data = json.loads(clean_text)
            
            # Ensure all required keys are present
            required_keys = ['executive_summary', 'key_points', 'decisions', 'action_items', 'participants', 'next_steps']
            for key in required_keys:
                if key not in summary_data:
                    summary_data[key] = [] if key != 'executive_summary' else "Summary information not available"
                    
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}. Using text-based fallback.")
            # If JSON parsing fails, try to extract meaningful content from the text
            lines = summary_text.split('\n')
            executive_summary = summary_text[:200] + "..." if len(summary_text) > 200 else summary_text
            
            # Try to extract key points from the text
            key_points = []
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                    key_points.append(line[1:].strip())
                elif len(line) > 20 and '.' in line:
                    key_points.append(line)
            
            if not key_points:
                # If no structured points found, create from sentences
                sentences = [s.strip() for s in summary_text.split('.') if len(s.strip()) > 20]
                key_points = sentences[:5]  # Take first 5 meaningful sentences
            
            summary_data = {
                "executive_summary": executive_summary,
                "key_points": key_points[:10],  # Limit to 10 points
                "decisions": [],
                "action_items": [],
                "participants": list(set([seg.get('speaker', 'Unknown') for seg in transcript])),
                "next_steps": [],
                "raw_summary": summary_text  # Keep the original for reference
            }
        
        # Add metadata
        summary_data["metadata"] = {
            "total_segments": len(transcript),
            "transcript_length": len(formatted_transcript),
            "model_used": "gpt-4o-mini",
            "processing_status": "success"
        }
        
        logger.info(f"Successfully generated summary for {len(transcript)} transcript segments")
        return summary_data
        
    except Exception as e:
        logger.error(f"Failed to generate summary: {str(e)}")
        print(f"OpenAI API Error: {str(e)}")  # Add console logging for debugging
        print(f"API Key present: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
        print(f"Transcript segments: {len(transcript)}")
        
        # Return error response in expected format
        return {
            "executive_summary": f"Error generating summary: {str(e)}. Please check your OpenAI API key and internet connection.",
            "key_points": [f"API Error: {str(e)}", "Please verify OpenAI API key in .env file", "Check internet connection"],
            "decisions": [],
            "action_items": ["Fix OpenAI integration", "Verify API credentials"],
            "participants": list(set([seg.get('speaker', 'Unknown') for seg in transcript])),
            "next_steps": [],
            "metadata": {
                "total_segments": len(transcript),
                "processing_status": "error",
                "error": str(e)
            }
        }


def generate_action_items(transcript: List[Dict[str, Any]]) -> List[str]:
    """
    Extract specific action items from the transcript.
    
    Args:
        transcript: List of transcript segments
        
    Returns:
        List of identified action items
    """
    global client
    
    if not client:
        initialize_openai()
    
    formatted_transcript = format_transcript_for_summary(transcript)
    
    prompt = f"""Extract specific action items from this meeting transcript. Focus on:
- Tasks assigned to specific people
- Deadlines mentioned
- Follow-up actions required
- Deliverables discussed

Meeting Transcript:
{formatted_transcript}

Return only a JSON list of action items, each as a string describing the action, who should do it, and any deadline mentioned."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        
        import json
        action_items = json.loads(response.choices[0].message.content)
        return action_items if isinstance(action_items, list) else []
        
    except Exception as e:
        logger.error(f"Failed to extract action items: {str(e)}")
        return []


def analyze_sentiment(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the overall sentiment and tone of the meeting.
    
    Args:
        transcript: List of transcript segments
        
    Returns:
        Dictionary with sentiment analysis results
    """
    global client
    
    if not client:
        initialize_openai()
    
    formatted_transcript = format_transcript_for_summary(transcript)
    
    prompt = f"""Analyze the sentiment and tone of this meeting transcript. Provide:
1. Overall meeting sentiment (positive, neutral, negative)
2. Key emotional moments
3. Collaboration level
4. Any tension or conflict areas
5. General meeting effectiveness

Meeting Transcript:
{formatted_transcript}

Return as JSON with keys: overall_sentiment, emotional_highlights, collaboration_score (1-10), tension_areas, effectiveness_rating (1-10)"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        
        import json
        sentiment_data = json.loads(response.choices[0].message.content)
        return sentiment_data
        
    except Exception as e:
        logger.error(f"Failed to analyze sentiment: {str(e)}")
        return {"overall_sentiment": "unknown", "error": str(e)}