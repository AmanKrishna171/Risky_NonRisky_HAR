from pytube import YouTube

# Ask for the YouTube video URL
link = "https://www.youtube.com/watch?v=dxl37zwUCH4"

# Create a YouTube object
youtube_object = YouTube(link)

# Get the highest resolution stream
video_stream = youtube_object.streams.get_highest_resolution()

# Download the video
try:
    video_stream.download()
    print("Download is completed successfully")
except:
    print("An error has occurred")

