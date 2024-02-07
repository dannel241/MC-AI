import lyricsgenius
import time
import re

''' Script to scrap lyrics using the Genius API '''

genius = lyricsgenius.Genius("OhdcL8Ig670Hb4S4YHXYWZWe7Q2-Zq9vK3dy01NWCL0KXEbskP1aHxFvPRXjqOrV",timeout = 50)

def clean_lyrics(raw_lyrics):
    # Remove the title of the song (the first line)
    cleaned_lyrics = re.sub(r'^.*\n', '', raw_lyrics, count=1)

    # Remove lines with "ContributorsTranslations"
    cleaned_lyrics = re.sub(r'\d+ ContributorsTranslations.*\n', '', cleaned_lyrics)

    # Remove the pattern of number.numberKEmbed
    cleaned_lyrics = re.sub(r'\d+\.\d+KEmbed', '', cleaned_lyrics)

    return cleaned_lyrics


# Artists from whom I get lyrics
artists = [
    '2Pac',
    'Guru',
    'Wu-Tang Clan',
    'Masta Ace',
    'Yasiin Bey',
    'MF DOOM',
    'The Notorious B.I.G.',
    'André 3000',
    'Kanye West',
    '50 Cent',
    'Kendrick Lamar'
]


def save_lyrics_for_artist(artist, max_songs_per_artist=2, max_retries=3):
    artist_name = artist.name.replace(" ", "_")  # Replace spaces in artist name with underscores
    artist_lyrics = ""

    for song in artist.songs[:max_songs_per_artist]:
        retries = 0
        while retries < max_retries:
            try:
                song_lyrics = clean_lyrics(song.lyrics)
                artist_lyrics += f"{song.title}\n\n{song_lyrics}\n\n"  # Concatenate song title and lyrics
                break  # Success, exit the loop
            except TimeoutError as e:
                print(f"Timeout error: {e}. Retrying...")
                retries += 1
                time.sleep(2)  # Wait for 2 seconds before retrying
            except Exception as e:
                print(f"An error occurred: {e}")
                break  # Exit the loop for other exceptions

    with open(f"{artist_name}_lyrics.txt", "w", encoding="utf-8") as file:
        file.write(artist_lyrics)

genius.verbose = True  # Turn off status messages
genius.remove_section_headers = True  # Remove section headers (e.g. [Chorus]) from lyrics when searching
genius.skip_non_songs = False  # Include hits thought to be non-songs (e.g. track lists)
genius.excluded_terms = ["(Remix)", "(Live)"]  # Exclude songs with these words in their title

for artist_name in artists:
    artist = genius.search_artist(artist_name, max_songs=50, sort="popularity", include_features=False)
    if artist:
        save_lyrics_for_artist(artist, max_songs_per_artist=50, max_retries=5)
    else:
        print(f"Failed to fetch data for {artist_name}")
