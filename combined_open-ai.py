import json
from openai import OpenAI
from typing import List, Dict, Any
from dataclasses import dataclass
import os
from collections import defaultdict
import re

# Allowed genre tags - your specific list
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

# Allowed classification types
CLASSIFICATION_TYPES = ["vocal", "instrumental", "song"]

@dataclass
class Song:
    """Individual song data structure - simplified"""
    tags: List[str]  # Genre/style tags (must be from GENRE_TAGS)
    classification: str  # Must be: vocal/instrumental/song
    
@dataclass
class CoreStyle:
    """Core style data structure"""
    name: str
    description: str
    tags: List[str]
    signature_sound_tags: List[str]
    songs: List[int]  # indices of songs in this style

@dataclass
class ArtistDNA:
    """Artist DNA level summary"""
    dna_tags: List[str]

class OpenAIMusicCategorizer:
    """
    Music style categorization system using OpenAI API
    Creates core styles, signature sounds, and DNA-level tags
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        # Set up OpenAI client with new API format
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key must be provided either as parameter or environment variable OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
            
        # Use a model that supports JSON mode
        self.model = model
        if model == "gpt-4":
            print("  Note: Switching to gpt-4o for JSON response format support")
            self.model = "gpt-4o"
            
        self.songs = []
        self.core_styles = []
        self.artist_dna = None
        
    def validate_song_input(self, tags: List[str], classification: str) -> bool:
        """Validate that tags and classification are from allowed lists"""
        # Check classification
        if classification.lower() not in [c.lower() for c in CLASSIFICATION_TYPES]:
            print(f"Warning: Classification '{classification}' not in allowed types: {CLASSIFICATION_TYPES}")
            return False
            
        # Check tags
        invalid_tags = []
        for tag in tags:
            if tag not in GENRE_TAGS:
                invalid_tags.append(tag)
                
        if invalid_tags:
            print(f"Warning: Invalid tags found: {invalid_tags}")
            print("Allowed tags are from GENRE_TAGS list")
            return False
            
        return True
        
    def add_song(self, tags: List[str], classification: str):
        """Add a song to the collection with validation"""
        # Validate input
        if not self.validate_song_input(tags, classification):
            print(f"Skipping song with invalid input")
            return False
            
        song = Song(tags, classification.lower())
        self.songs.append(song)
        return True
        
    def extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON from response text, handling cases where it might not be pure JSON"""
        try:
            # First try to parse as pure JSON
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            # If that fails, try to find JSON within the response
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            if matches:
                try:
                    return json.loads(matches[0])
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, return None
            print(f"Could not extract JSON from response: {response_text[:200]}...")
            return None
        
    def call_openai_api(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.3):
        """Make API call to OpenAI using new client format"""
        try:
            # Try with JSON mode first
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            except Exception as json_error:
                print(f"JSON mode failed, falling back to regular mode: {json_error}")
                # Fallback to regular mode without JSON format
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return None
            
    def classify_songs_into_core_styles(self, artist_name: str = "Artist") -> Dict[str, Any]:
        """Use OpenAI to classify songs into core styles (1-10 max)"""
        
        # Prepare song data for OpenAI
        songs_data = []
        for i, song in enumerate(self.songs):
            song_info = {
                "song_id": i,
                "tags": song.tags,
                "classification": song.classification
            }
            songs_data.append(song_info)
            
        system_prompt = """You are a music industry expert specializing in artist style analysis. Your task is to analyze songs and group them into core musical styles (minimum 1, maximum 10 core styles).

IMPORTANT GUIDELINES:
- DO NOT be overly sensitive with categorization
- TRY TO COMBINE similar styles of music rather than creating too many separate categories
- Focus on major stylistic differences, not minor variations
- Group songs that share similar musical DNA, production style, or thematic elements
- Each core style should have at least 1-2 songs minimum
- Remember that classification can only be: vocal, instrumental, or song

You must return a JSON response with this exact structure:
{
  "core_styles": [
    {
      "style_name": "Style Name",
      "song_ids": [0, 1, 2],
      "primary_characteristics": ["char1", "char2", "char3"]
    }
  ]
}

IMPORTANT: Your response must be valid JSON only, no additional text or explanations."""

        user_prompt = f"""Analyze these {len(songs_data)} songs for artist "{artist_name}" and group them into core musical styles.

Songs to analyze:
{json.dumps(songs_data, indent=2)}

Remember:
- Combine similar styles, don't be overly sensitive
- Maximum 10 core styles, minimum 1
- Focus on major musical differences
- Each style needs clear musical identity
- Classifications are: vocal, instrumental, song
- Return only valid JSON, no additional text"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_openai_api(messages, max_tokens=1500)
        
        if response:
            result = self.extract_json_from_response(response)
            if result:
                return result
            else:
                print("Error parsing OpenAI response")
                return None
        return None
        
    def generate_core_style_details(self, style_name: str, song_ids: List[int], 
                                  primary_characteristics: List[str], artist_name: str) -> Dict[str, Any]:
        """Use OpenAI to generate detailed core style information"""
        
        # Get songs for this style
        style_songs = [self.songs[i] for i in song_ids]
        songs_info = []
        
        for i, song in enumerate(style_songs):
            songs_info.append({
                "tags": song.tags,
                "classification": song.classification
            })
            
        system_prompt = f"""You are a music industry expert writing detailed style analysis. Create comprehensive details for a core musical style.

You must return a JSON response with this exact structure:
{{
  "style_name": "Final Style Name",
  "description": "Detailed 200-400 character description in the style: '{artist_name}'s [STYLE] vision merges [key elements]. The productions focus on [characteristics]. By [approach], {artist_name} [impact/contribution].'",
  "core_style_tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "signature_sound_tags": ["sound1", "sound2", "sound3", "sound4", "sound5"]
}}

The description should sound professional and music-industry focused, similar to how you'd describe an artist's style in a music magazine or streaming platform.

IMPORTANT: Your response must be valid JSON only, no additional text or explanations."""

        user_prompt = f"""Create detailed information for this core style:

Style Name: {style_name}
Artist: {artist_name}
Primary Characteristics: {primary_characteristics}
Number of Songs: {len(song_ids)}

Songs in this style:
{json.dumps(songs_info, indent=2)}

Generate:
1. A refined style name (if needed)
2. A compelling 200-400 character description following the format shown
3. 5 core style tags (genre/feel descriptors)
4. 5 signature sound tags (sonic characteristics)

Note: Classifications are limited to: vocal, instrumental, song
Return only valid JSON, no additional text."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_openai_api(messages, max_tokens=1200)
        
        if response:
            result = self.extract_json_from_response(response)
            if result:
                return result
            else:
                print(f"Error parsing style details for {style_name}")
                return None
        return None
        
    def generate_artist_dna_tags(self, core_styles_data: List[Dict], artist_name: str) -> List[str]:
        """Use OpenAI to generate 5 DNA-level tags summarizing all core styles"""
        
        system_prompt = """You are a music industry expert. Generate 5 super-tags that summarize an artist's overall musical DNA based on all their core styles.

These DNA tags should capture the artist's overarching musical identity across all styles.

You must return a JSON response:
{
  "dna_tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}

IMPORTANT: Your response must be valid JSON only, no additional text or explanations."""

        # Prepare core styles data
        styles_summary = []
        for style in core_styles_data:
            style_info = {
                "name": style.get('style_name', ''),
                "core_tags": style.get('core_style_tags', []),
                "signature_tags": style.get('signature_sound_tags', [])
            }
            styles_summary.append(style_info)
            
        user_prompt = f"""Generate 5 DNA-level tags for artist "{artist_name}" based on these core styles:

{json.dumps(styles_summary, indent=2)}

The DNA tags should represent the artist's overall musical identity that spans across all their core styles.

Return only valid JSON, no additional text."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_openai_api(messages, max_tokens=300)
        
        if response:
            result = self.extract_json_from_response(response)
            if result:
                return result.get('dna_tags', [])
            else:
                print("Error parsing DNA tags response")
                return []
        return []
        
    def analyze_artist(self, artist_name: str = "Artist") -> Dict[str, Any]:
        """Complete analysis pipeline using OpenAI API"""
        
        if not self.songs:
            return {"error": "No songs provided for analysis"}
            
        print(f" Analyzing {len(self.songs)} songs for {artist_name}...")
        print(f" Using model: {self.model}")
        
        # Step 1: Classify songs into core styles
        print(" Classifying songs into core styles...")
        classification_result = self.classify_songs_into_core_styles(artist_name)
        
        if not classification_result:
            return {"error": "Failed to classify songs into core styles"}
            
        core_styles_data = []
        
        # Step 2: Generate detailed information for each core style
        print(" Generating detailed core style information...")
        for style_info in classification_result.get('core_styles', []):
            style_name = style_info.get('style_name', '')
            song_ids = style_info.get('song_ids', [])
            characteristics = style_info.get('primary_characteristics', [])
            
            print(f"    Processing: {style_name}")
            
            style_details = self.generate_core_style_details(
                style_name, song_ids, characteristics, artist_name
            )
            
            if style_details:
                style_details['song_ids'] = song_ids
                style_details['song_count'] = len(song_ids)
                core_styles_data.append(style_details)
                
        # Step 3: Generate Artist DNA tags
        print(" Generating Artist DNA tags...")
        dna_tags = self.generate_artist_dna_tags(core_styles_data, artist_name)
        
        # Format final result
        result = {
            "artist_name": artist_name,
            "total_songs_analyzed": len(self.songs),
            "total_core_styles": len(core_styles_data),
            "core_styles": core_styles_data,
            "artist_dna": {
                "dna_tags": dna_tags
            }
        }
        
        print(" Analysis complete!")
        return result
        
    def print_analysis_results(self, results: Dict[str, Any]):
        """Print results in a formatted way"""
        
        if "error" in results:
            print(f" Error: {results['error']}")
            return
            
        print("=" * 80)
        print(f" {results['artist_name']} - AI-POWERED MUSIC DNA ANALYSIS")
        print("=" * 80)
        print(f" Songs Analyzed: {results['total_songs_analyzed']}")
        print(f" Core Styles Found: {results['total_core_styles']}")
        print("=" * 80)
        
        print("\n CORE STYLES:")
        print("-" * 60)
        
        for i, style in enumerate(results['core_styles'], 1):
            print(f"\n[{i}] {style['style_name'].upper()}")
            print(f"      {results['artist_name']}")
            print(f"      {style['description']}")
            print(f"       Core Tags: {', '.join(style['core_style_tags'])}")
            print(f"      Signature Sound: {', '.join(style['signature_sound_tags'])}")
            print(f"      Songs: {style['song_count']} tracks")
            print("-" * 60)
        
        print(f"\n🧬 ARTIST DNA TAGS:")
        print("-" * 60)
        print(f"    {', '.join(results['artist_dna']['dna_tags'])}")
        
        print("\n" + "=" * 80)

    def save_results_to_json(self, results: Dict[str, Any], filename: str = "music_analysis_results.json"):
        """Save analysis results to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f" Results saved to {filename}")
            
            # Also display file download link in Colab
            try:
                from google.colab import files
                print(f" Downloading {filename} for you...")
                files.download(filename)
            except ImportError:
                print(" File saved locally (not in Colab environment)")
                
        except Exception as e:
            print(f" Error saving results: {e}")

def print_allowed_genres():
    """Print all allowed genre tags for reference"""
    print(" ALLOWED GENRE TAGS:")
    print("=" * 50)
    genres_list = sorted(list(GENRE_TAGS))
    for i, genre in enumerate(genres_list, 1):
        print(f"{i:2d}. {genre}")
    print("=" * 50)
    print(f"Total: {len(genres_list)} allowed genres")

def run_music_dna_analysis(artist_name: str, songs_data: List[Dict], api_key: str = None):
    """
    Main function to run music DNA analysis
    
    Args:
        artist_name: Name of the artist
        songs_data: List of dicts with 'tags' and 'classification' keys
        api_key: OpenAI API key (optional if set as environment variable)
    
    Returns:
        Dict with analysis results
    """
    
    try:
        # Initialize categorizer
        categorizer = OpenAIMusicCategorizer(api_key=api_key, model="gpt-4o")
        
        # Add songs
        print(f" Adding {len(songs_data)} songs...")
        successful_adds = 0
        
        for i, song_data in enumerate(songs_data):
            if 'tags' not in song_data or 'classification' not in song_data:
                print(f"  Skipping song {i+1}: Missing 'tags' or 'classification'")
                continue
                
            success = categorizer.add_song(
                tags=song_data["tags"],
                classification=song_data["classification"]
            )
            
            if success:
                successful_adds += 1
                
        print(f" Successfully added {successful_adds} songs")
        
        if successful_adds == 0:
            print(" No valid songs to analyze!")
            return None
            
        # Run analysis
        results = categorizer.analyze_artist(artist_name=artist_name)
        
        # Print and save results
        if results and "error" not in results:
            categorizer.print_analysis_results(results)
            categorizer.save_results_to_json(results, f"{artist_name.replace(' ', '_')}_dna_analysis.json")
            return results
        else:
            print(" Analysis failed!")
            return results
            
    except Exception as e:
        print(f" Error in analysis: {e}")
        return None

# EXAMPLE USAGE FOR COLAB
def example_analysis():
    """Example analysis you can run in Colab"""
    
    # Sample artist data with corrected tags and classifications
    artist_name = "SAMPLE ARTIST"
    
    sample_songs = [
        {
            "tags": ["Electronic music", "House music", "Dance music", "Electronic dance music"],
            "classification": "instrumental"
        },
        {
            "tags": ["Pop music", "Electronic music", "Happy music", "Dance music"],
            "classification": "vocal"
        },
        {
            "tags": ["Jazz", "Soul music", "Vocal music"],
            "classification": "vocal"
        },
        {
            "tags": ["Rock music", "Heavy metal", "Exciting music"],
            "classification": "song"
        },
        {
            "tags": ["Ambient music", "Electronic music", "Tender music"],
            "classification": "instrumental"
        },
        {
            "tags": ["Hip hop music", "Rapping", "Electronic music"],
            "classification": "vocal"
        },
        {
            "tags": ["Classical music", "Opera", "Vocal music"],
            "classification": "vocal"
        },
        {
            "tags": ["Reggae", "Happy music", "Vocal music"],
            "classification": "song"
        }
    ]
    
    print(" RUNNING EXAMPLE MUSIC DNA ANALYSIS")
    print("=" * 60)
    
    # Run analysis
    results = run_music_dna_analysis(artist_name, sample_songs)
    
    return results

# FOR COLAB: Uncomment these lines to run
if __name__ == "__main__":
    print(" Music DNA Analyzer - Ready for Colab!")
    print("=" * 60)
    
    # Show allowed genres
    print_allowed_genres()
    
    print("\n" + "=" * 60)
    print(" ALLOWED CLASSIFICATIONS: vocal, instrumental, song")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n  OpenAI API key not found!")
        print(" Please set your API key:")
        print("   import os")
        print("   os.environ['OPENAI_API_KEY'] = 'your-api-key-here'")
        print("\n Then run: example_analysis()")
    else:
        print(f"\n API key found! Running example analysis...")
        example_analysis()
