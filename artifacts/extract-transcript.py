import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

def extract_transcript(youtube_url):
    """
    Extracts and returns the transcript of a YouTube video.

    Args:
        youtube_url: The URL of the YouTube video.

    Returns:
        A string containing the transcript or None if no transcript is found or
        an error occurs.
    """

    try:
        video_id = yt_dlp.YoutubeDL().extract_info(youtube_url, download=False)['id']
        
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_generated_transcript(['en'])  # Prioritize generated English transcript
        
        transcript_text = " ".join([entry['text'] for entry in transcript.fetch()])
        return transcript_text

    except yt_dlp.utils.DownloadError as e:
       print(f"Error fetching video information: {e}")
       return None
    except Exception as e:
        print(f"Error getting transcript: {e}")
        return None


if __name__ == "__main__":
    video_url = input("Enter the YouTube video URL: ")
    transcript = extract_transcript(video_url)

    if transcript:
        print("\nTranscript:\n")
        print(transcript)
    else:
        print("\nCould not extract transcript.")