#!/usr/bin/env python3

import os
import sys
import torch
import torchaudio
import subprocess
import tempfile
import argparse
import time
from collections import defaultdict

# Import whisper_at only when needed (for tagging)
whisper = None

# Define known vocal and instrumental tags for classification
VOCAL_TAGS = {
    "Singing", "Speech", "Choir", "Female singing", "Male singing",
    "Chant", "Yodeling", "Shout", "Bellow", "Rapping", "Narration",
    "Child singing", "Vocal music", "Opera", "A capella", "Voice",
    "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue", 
    "Babbling", "Speech synthesizer", "Whoop", "Yell", "Battle cry",
    "Children shouting", "Screaming", "Whispering", "Mantra",
    "Synthetic singing", "Humming", "Whistling", "Beatboxing",
    "Gospel music", "Lullaby", "Groan", "Grunt"
}

# Definitive speech tags that guarantee vocal classification
DEFINITIVE_SPEECH_TAGS = {
    "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue"
}

INSTRUMENTAL_TAGS = {
    "Piano", "Electric piano", "Keyboard (musical)", "Musical instrument",
    "Synthesizer", "Organ", "New-age music", "Electronic organ", 
    "Christmas music", "Rhythm and blues", "Independent music", "Soundtrack music", 
    "Pop music", "Jazz", "Soul music", "Christian music", "Instrumental music", 
    "Harpsichord", "Guitar", "Bass guitar", "Drums", "Violin", "Trumpet", 
    "Flute", "Saxophone", "Plucked string instrument", "Electric guitar", 
    "Acoustic guitar", "Steel guitar, slide guitar", "Banjo", "Sitar", "Mandolin", 
    "Ukulele", "Hammond organ", "Sampler", "Percussion", "Drum kit","Musical instrument", 
    "Drum machine", "Drum", "Snare drum", "Bass drum", "Timpani", "Tabla", 
    "Cymbal", "Hi-hat", "Tambourine", "Marimba, xylophone", 
    "Vibraphone", "Orchestra", "Brass instrument",
    "French horn", "Trombone", "Bowed string instrument", "String section", 
    "Violin, fiddle", "Cello", "Double bass", "Wind instrument, woodwind instrument", 
    "Clarinet", "Harp", "Harmonica", "Accordion", 
    "Rock music", "Heavy metal", "Punk rock", "Grunge", "Progressive rock", 
    "Rock and roll", "Psychedelic rock", "Reggae", "Country", "Swing music", 
    "Bluegrass", "Funk", "Folk music", "Middle Eastern music", "Disco", 
    "Classical music", "Electronic music", "House music", "Techno", "Dubstep", 
    "Drum and bass", "Electronica", "Electronic dance music", "Ambient music", 
    "Trance music", "Music of Latin America", "Salsa music", "Flamenco", "Blues", 
    "Music for children", "Music of Africa", "Afrobeat", "Music of Asia", 
    "Carnatic music", "Music of Bollywood", "Ska", "Traditional music", 
    "Song", "Theme music", "Video game music", "Dance music", "Wedding music", 
    "Happy music", "Funny music", "Sad music", "Tender music", "Exciting music", 
    "Angry music", "Scary music"
}

# Genre tags for fancy music classification
GENRE_TAGS = {
    # Main genres
    "Pop music", "Rock music", "Jazz", "Classical music", "Electronic music",
    "Blues", "Country", "Folk music", "Reggae", "Funk", "Soul music",
    "Rhythm and blues", "Gospel music", "Opera", "Hip hop music",
    
    # Electronic subgenres
    "House music", "Techno", "Dubstep", "Drum and bass", "Electronica",
    "Electronic dance music", "Ambient music", "Trance music",
    
    # Rock subgenres
    "Heavy metal", "Punk rock", "Grunge", "Progressive rock", "Rock and roll",
    "Psychedelic rock",
    
    # World music
    "Music of Latin America", "Salsa music", "Flamenco", "Music of Africa",
    "Afrobeat", "Music of Asia", "Carnatic music", "Music of Bollywood",
    "Middle Eastern music", "Traditional music",
    
    # Other genres
    "Swing music", "Bluegrass", "Ska", "Disco", "New-age music",
    "Independent music", "Christian music", "Soundtrack music",
    "Theme music", "Video game music", "Dance music", "Wedding music",
    "Christmas music", "Music for children",
    
    # Mood/style tags
    "Happy music", "Funny music", "Sad music", "Tender music",
    "Exciting music", "Angry music", "Scary music",
    
    # Vocal styles
    "A capella", "Vocal music", "Choir", "Chant", "Mantra", "Lullaby",
    "Beatboxing", "Rapping", "Yodeling"
}

def load_vad_model():
    """Load Silero VAD model"""
    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
    
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps

def convert_audio_with_ffmpeg(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path, 
            '-ar', '16000',  # Set sample rate to 16kHz
            '-ac', '1',      # Convert to mono
            '-y',            # Overwrite output file
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def detect_vocals_vad(file_path, model, get_speech_timestamps):
    """Detect vocals using VAD and return detection results"""
    
    if not os.path.exists(file_path):
        return None, f"Error: Audio file '{file_path}' not found."
    
    waveform = None
    sample_rate = None
    temp_file = None
    
    try:
        # Try to load the audio file directly with torchaudio
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as e1:
            print(f"Direct loading failed: {e1}")
            print("Trying to convert with ffmpeg...")
            
            # Try converting with ffmpeg
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            if convert_audio_with_ffmpeg(file_path, temp_file.name):
                try:
                    waveform, sample_rate = torchaudio.load(temp_file.name)
                    print("Successfully converted and loaded audio")
                except Exception as e2:
                    return None, f"Error loading converted audio: {str(e2)}"
            else:
                # Try using Silero's read_audio utility as last resort
                try:
                    print("Trying Silero's read_audio utility...")
                    model_temp, utils_temp = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                           model='silero_vad',
                                                           force_reload=False)
                    read_audio = utils_temp[2]  # read_audio is the 3rd utility
                    waveform = read_audio(file_path, sampling_rate=16000)
                    sample_rate = 16000
                    print("Successfully loaded with Silero's read_audio")
                except Exception as e3:
                    return None, f"All loading methods failed. Last error: {str(e3)}"
        
        # Convert to mono if stereo
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ensure sample rate is 16000 Hz as required by Silero VAD
        if sample_rate != 16000:
            print(f"Resampling from {sample_rate} Hz to 16000 Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Flatten to 1D array if needed
        if len(waveform.shape) > 1:
            waveform = waveform.squeeze()
        
        # Get speech timestamps using Silero VAD
        speech_timestamps = get_speech_timestamps(waveform, model, threshold=0.5, sampling_rate=16000)
        
        # Calculate speech statistics
        total_duration_seconds = len(waveform) / sample_rate
        speech_duration = sum([t['end'] - t['start'] for t in speech_timestamps]) / sample_rate
        speech_percentage = (speech_duration / total_duration_seconds) * 100 if total_duration_seconds > 0 else 0
        
        # Determine if vocal is detected
        vocal_detected = len(speech_timestamps) > 0 and speech_percentage > 1.0
        
        return {
            'vocal_detected': vocal_detected,
            'speech_percentage': round(speech_percentage, 2),
            'total_duration': round(total_duration_seconds, 2),
            'speech_duration': round(speech_duration, 2)
        }, None
        
    except Exception as e:
        return None, f"Error processing audio: {str(e)}"
    
    finally:
        # Clean up temporary file if created
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def extract_genre_tags(top_tags):
    """Extract genre tags from detected top tags"""
    detected_genres = []
    for tag in top_tags:
        if tag in GENRE_TAGS:
            detected_genres.append(tag)
    return detected_genres

def classify_audio_tags(top_tags):
    """Classify audio based on detected tags"""
    # Check for definitive speech tags first - if any are present, it's definitely vocal
    has_definitive_speech = any(tag in DEFINITIVE_SPEECH_TAGS for tag in top_tags)
    
    if has_definitive_speech:
        return "Speech/Vocal"
    
    # Regular classification logic as fallback
    has_vocal = any(tag in VOCAL_TAGS for tag in top_tags)
    has_instrumental = any(tag in INSTRUMENTAL_TAGS for tag in top_tags)

    if has_vocal and not has_instrumental:
        return "Vocal/Song"
    elif has_instrumental and not has_vocal:
        return "Instrumental"
    elif has_vocal and has_instrumental:
        return "Song (Mixed)"
    else:
        return "Unknown"

def classify_with_tagging(audio_path, model_size="small"):
    """Classify audio using Whisper-AT tagging"""
    global whisper
    
    # Import whisper_at when needed
    if whisper is None:
        try:
            import whisper_at as whisper
        except ImportError:
            return "Error: whisper-at not installed. Please run the setup script.", []
    
    print(f"Loading Whisper-AT model ({model_size}) for detailed classification...")
    
    audio_tagging_time_resolution = 4.8
    
    try:
        start_time = time.time()
        model = whisper.load_model(model_size)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

        result = model.transcribe(audio_path, at_time_res=audio_tagging_time_resolution)

        audio_tag_result = whisper.parse_at_label(
            result,
            language='en',
            top_k=15,
            p_threshold=-5
        )

        all_tags_set = set()
        tag_freq = defaultdict(int)

        for segment in audio_tag_result:
            # Update tag set and frequency
            for tag, score in segment['audio tags']:
                all_tags_set.add(tag)
                tag_freq[tag] += 1

        # Find top tags (those that appear more than once)
        top_tags = [tag for tag, freq in tag_freq.items() if freq > 1]
        
        print(f"Detected tags: {', '.join(top_tags[:5])}...")  # Show first 5 tags
        
        # Extract genre tags
        genre_tags = extract_genre_tags(top_tags)
        
        classification = classify_audio_tags(top_tags)
        return classification, genre_tags
        
    except Exception as e:
        return f"Error in tagging: {str(e)}", []

def main():
    parser = argparse.ArgumentParser(description='Classify audio files using VAD and audio tagging')
    parser.add_argument('audio_file', type=str, help='Path to the audio file')
    parser.add_argument('--model', type=str, default='small', 
                        choices=['tiny', 'base', 'small', 'medium', 'large-v1'],
                        help='Whisper model size for tagging (default: small)')
    parser.add_argument('--vad-only', action='store_true',
                        help='Only run VAD detection without detailed tagging')
    
    args = parser.parse_args()
    
    audio_path = args.audio_file
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        sys.exit(1)
    
    print(f"Processing audio file: {audio_path}")
    print("=" * 60)
    
    # Step 1: VAD Detection
    try:
        vad_model, get_speech_timestamps = load_vad_model()
    except Exception as e:
        print(f"Error loading VAD model: {e}")
        sys.exit(1)
    
    vad_result, vad_error = detect_vocals_vad(audio_path, vad_model, get_speech_timestamps)
    
    if vad_error:
        print(vad_error)
        sys.exit(1)
    
    print(f"\n  VAD Results:")
    print(f"   Vocal Detected: {'Yes' if vad_result['vocal_detected'] else 'No'}")
    print(f"   Speech Percentage: {vad_result['speech_percentage']}%")
    print(f"   Total Duration: {vad_result['total_duration']} seconds")
    print(f"   Speech Duration: {vad_result['speech_duration']} seconds")
    
    # Step 2: Classification Logic
    if not vad_result['vocal_detected']:
        # No vocals detected - classify as instrumental
        final_classification = "Instrumental"
        detected_genres = []
        print(f"\n Final Classification: {final_classification}")
        print("   Reason: No vocals detected by VAD")
    else:
        # Vocals detected - use tagging for detailed classification
        if args.vad_only:
            final_classification = "Vocal (VAD detected)"
            detected_genres = []
            print(f"\n Final Classification: {final_classification}")
            print("Reason: Vocals detected by VAD (detailed tagging skipped)")
        else:
            print(f"\n  Running audio tagging for detailed classification...")
            tag_classification, detected_genres = classify_with_tagging(audio_path, args.model)
            final_classification = tag_classification
            print(f"\n Final Classification: {final_classification}")
            print("Reason: Vocals detected by VAD, classified using audio tagging")
    
    # Display genre information
    if detected_genres:
        print(f"\n  Detected Genres/Styles:")
        for i, genre in enumerate(detected_genres, 1):
            print(f"   {i}. {genre}")
    else:
        print(f"\n  Detected Genres/Styles: None detected")
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULT: {final_classification}")
    if detected_genres:
        print(f"GENRES: {', '.join(detected_genres)}")
    print("=" * 60)

if __name__ == '__main__':
    main()
