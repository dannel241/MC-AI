import tkinter as tk
from tkinter import messagebox
from main_script import get_lyric, load_model_and_data



# Function to generate lyrics
def generate_lyrics(model, prompt, num_words, idx_to_word, word_to_idx):
    # Include your code to generate lyrics here
    lyrics = get_lyric(model, prompt, False, num_words, idx_to_word, word_to_idx)

    # Create a new window for displaying lyrics
    lyrics_window = tk.Toplevel(app)
    lyrics_window.title("Generated Lyrics")

    # Create a Label widget to display the lyrics with a larger font
    lyrics_label = tk.Label(lyrics_window, text=lyrics, font=("Arial", 24))  # Change 14 to your desired font size
    lyrics_label.pack(padx=10, pady=10)

# Function to handle generate button click
def on_generate_button_click(model, idx_to_word, word_to_idx):
    # Get the prompt from the entry widget
    prompt = prompt_entry.get()

    # Call the generate_lyrics function with the provided prompt and other parameters
    total_number_of_words = 100
    generate_lyrics(model, prompt, total_number_of_words, idx_to_word, word_to_idx)

# Load model and data
model, word_to_idx, idx_to_word = load_model_and_data()

# Create the main application window
app = tk.Tk()
app.title("Lyrics Generator")

# Create an entry widget for the user to input a prompt with a larger font
prompt_entry = tk.Entry(app, width=40, font=("Arial", 24))  # Change 12 to your desired font size
prompt_entry.pack(pady=10)

# Create a button to trigger lyrics generation
generate_button = tk.Button(app, text="Generate Lyrics", command=lambda: on_generate_button_click(model, idx_to_word, word_to_idx))
generate_button.pack(pady=20)

# Start the main event loop
app.mainloop()
