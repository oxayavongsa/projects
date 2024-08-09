# %% [markdown]
# # Music Genre and Composer Classification Using Deep Learning 
# 
# ` Authors: Zain Ali, Angel Benitez, and Outhai Xayavongsa`
# 
# 

# %% [markdown]
# ## Introduction
# 
# In this project, we employ deep learning to classify classical music compositions by their composers. Leveraging a dataset of 3,929 MIDI files from 175 composers—including Bach, Beethoven, Chopin, and Mozart—we develop Long Short-Term Memory (LSTM)  and Convolutional Neural Network (CNN) models to identify the composer of a given piece. Initially, we concentrate on the four mentioned composers to fine-tune our approach. In the end, we created a model encompassing all 147 composers in the dataset, assessing its generalization capabilities across diverse musical styles. We also performed optimizations and many other techniques to get the best models within the last few weeks. 
# 
# 
# If you would like more information about the files or need access to the full project, please go to our GitHub repository: https://github.com/zainnobody/AAI-511-Final-Project. Feel free to fork or clone it. The README file also contains more information. 

# %% [markdown]
# ### Libraries Import
# 
# Following are all the libraries and packages used within our project.

# %%
import os
import shutil
import zipfile
import random
import time
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mido
from mido import MidiFile, bpm2tempo, tick2second
import pretty_midi
import pygame

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical

from keras import backend


# %% [markdown]
# ### Global Variables
# 
# The following variables were used throughout the project. Although the variables are used globally, they were not used as constants, so they are not all capitalized. If you are cloning the GitHub, feel free to change the values. 

# %%
# Directory where the raw data will be extracted
raw_data_zip = "raw_data/midi_classic_music_data.zip"  # Location of the zip file
raw_data_extracted = "raw_data_unzipped"  # Location where you would like the zip file to extract everything
specific_artists = [
    "Bach",
    "Beethoven",
    "Chopin",
    "Mozart",
]  # These are used for the initial LSTM and CNN analysis


# %% [markdown]
# ## Data Collection
# 
# The data was quite unorganized and downloaded in a zip format. Several steps were taken to make the data useful and well organized. Get more information about the data within Kaggle at: https://www.kaggle.com/datasets/blanderbuss/midi-classic-music. 

# %%
# Function to delete a directory and its contents
def delete_dir(dir_to_delete):
    try:
        file_count = sum([len(files) for r, d, files in os.walk(dir_to_delete)])
        shutil.rmtree(dir_to_delete)
        print(f"Directory {dir_to_delete} and all its contents ({file_count} files) have been successfully deleted.")
    except Exception as e:
        print(f"An error occurred while trying to delete the directory: {e}")

# %%
# Function to unzip
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

# %%
# Function to move contents of a directory up one level
def move_contents_up_one_dir(path):
    path = os.path.abspath(path)
    parent_dir = os.path.dirname(path)
    if path == parent_dir or not os.path.exists(path):
        print("Operation not allowed or path does not exist.")
        return
    for item in os.listdir(path):
        shutil.move(os.path.join(path, item), os.path.join(parent_dir, item))
    os.rmdir(path)
    print(f"All contents moved from {path} to {parent_dir} and directory removed.")

# %%
# Function to rename .MID files to .mid for consistency 
def rename_mid_files(directory):
    rename_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.MID'):
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, file[:-4] + '.mid')
                os.rename(old_file_path, new_file_path)
                rename_count += 1
    return rename_count

# %%
# Function to delete .zip files
def delete_zip_files(directory):
    delete_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                delete_count += 1
    return delete_count

# %%
# Function to Move Folder Contents
def move_folder_contents(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    for item in os.listdir(src_folder):
        src_item = os.path.join(src_folder, item)
        dest_item = os.path.join(dest_folder, item)
        
        if os.path.isdir(src_item):
            shutil.move(src_item, dest_folder)
        else:
            shutil.move(src_item, dest_item)
    
    delete_dir(src_folder)

# %%
# Funtion to move the content of the corrected directory
def directory_name_corrections(name_corrections_dirs):
    for src_folder, dest_folder in name_corrections_dirs.items():
        src_path = os.path.join(raw_data_extracted, src_folder)
        dest_path = os.path.join(raw_data_extracted, dest_folder)
        print(f"Moving contents from {src_path} to {dest_path}...")
        move_folder_contents(src_path, dest_path)

    print("Folder contents moved and directories deleted successfully.")

# %%
# Function to Categorize Files by Directory
def categorize_files_by_dir(path):
    files_and_dirs = os.listdir(path)
    directories = {name for name in files_and_dirs if os.path.isdir(os.path.join(path, name))}
    categorized_files = {}
    unassigned_files = {}

    for file_name in files_and_dirs:
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.mid'):
            first_word = file_name.split()[0]
            if first_word in directories:
                if first_word not in categorized_files:
                    categorized_files[first_word] = []
                categorized_files[first_word].append(file_name)
            else:
                if first_word not in unassigned_files:
                    unassigned_files[first_word] = []
                unassigned_files[first_word].append(file_name)

    print("Categorized Files Summary:")
    for key, files in categorized_files.items():
        print(f"Artist {key}: {len(files)} files")

    print("\nUnassigned Files Summary:")
    for key, files in unassigned_files.items():
        print(f"Artist {key}: {len(files)} files")
    
    return categorized_files, unassigned_files, sorted(directories)

# %%
# Function to Display Information about Categorized and Unassigned Files
def display_info(categorized_files, unassigned_files):
    print("Categorized Files Summary:")
    for key, files in categorized_files.items():
        print(f"Artist '{key}': {len(files)} files")
    
    print("\nUnassigned Files Summary:")
    if unassigned_files:
        for key, files in unassigned_files.items():
            print(f"Artist '{key}': {len(files)} files")
    else:
        print("No unassigned files found.")

# %%
# Correcting placement of files.
def corrections_to_file_placements(unassigned_files, corrections_to_file_placement):    
    for old_key, new_key in corrections_to_file_placement.items():
        if old_key in unassigned_files:
            unassigned_files[new_key] = unassigned_files.pop(old_key)


# %%
# Function to move files to their respective directories
def move_files_to_directories(base_path, files_to_move):
    for directory, files in files_to_move.items():
        dir_path = os.path.join(base_path, directory)
        # Create directory if it doesn't exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # Move each file to the new directory
        for file_name in files:
            shutil.move(
                os.path.join(base_path, file_name), os.path.join(dir_path, file_name)
            )


# %%
# We wanted to have own exception
class ArtistNotFoundError(Exception):
    def __init__(self, missing_artists):
        self.missing_artists = missing_artists
        super().__init__(
            f"The following specific artists are not in the all artists list: {', '.join(missing_artists)}"
        )


# %%
# Get list of all the artist dirs
def get_all_artists(raw_data_extracted):
    all_artists = {
        name
        for name in os.listdir(raw_data_extracted)
        if os.path.isdir(os.path.join(raw_data_extracted, name))
    }
    return all_artists


# %%
# Correct misnamed folders and move contents accordingly
name_corrections_dirs = {
    "Albe'niz": "Albeniz",
    "Albe╠üniz": "Albeniz",
    "Mendelsonn": "Mendelssohn",
    "Tchakoff": "Tchaikovsky",
    "Handel": "Handel",
    "Haendel": "Handel",
    "Straus": "Strauss",
    "Strauss, J": "Strauss",
}

corrections_to_file_placement = {"Pachebel": "Pachelbel", "Lizt": "Liszt"}


# %% [markdown]
# ### Full Steps
# 
# Above are the functions, and all are used within `initial_start`. 

# %%
def initial_start(raw_data_zip, raw_data_extracted, specific_artists):
    # This is in case of testing and if the initial raw files need to be deleted.
    if os.path.exists(raw_data_extracted):
        delete_dir(raw_data_extracted)
    raw_data_extracted = unzip_file(raw_data_zip, raw_data_extracted)

    display(f"Extracted to: {raw_data_extracted}")

    # There is a directory 'midiclassics' that needs to be moved one directory up to make all the structure similar.
    move_contents_up_one_dir(os.path.join(raw_data_extracted, "midiclassics"))

    # Rename .MID files to .mid for consistency
    renamed_files_count = rename_mid_files(raw_data_extracted)
    print(f"Total .MID files renamed: {renamed_files_count}")

    # Delete .zip files
    deleted_files_count = delete_zip_files(raw_data_extracted)
    print(f"Total .zip files deleted: {deleted_files_count}")

    # Categorize files and corrections to dir and files
    categorized_files, unassigned_files, all_artists = categorize_files_by_dir(
        raw_data_extracted
    )
    corrections_to_file_placements(unassigned_files, corrections_to_file_placement)
    directory_name_corrections(name_corrections_dirs)

    # Move categorized and unassigned files to their respective directories
    move_files_to_directories(raw_data_extracted, categorized_files)
    move_files_to_directories(raw_data_extracted, unassigned_files)

    # Final check to see the artists from the project are present within list of artists
    all_artists = get_all_artists(raw_data_extracted)

    missing_artists = [
        artist for artist in specific_artists if artist not in all_artists
    ]

    if not missing_artists:
        print("\n\nAll specific artists are in the all artists list.")
    else:
        raise ArtistNotFoundError(missing_artists)


# Processes data, only need once in the beginning.
initial_start(raw_data_zip, raw_data_extracted, specific_artists)


# %% [markdown]
# ## Data Pre-Processing

# %%
# Function to calculate the length of a MIDI file
def calculate_midi_length(file_path, debug=True):
    try:
        midi_file = MidiFile(file_path)
        total_time = 0.0

        for track in midi_file.tracks:
            current_time = 0.0
            tempo = bpm2tempo(120)  # Default tempo is 120 BPM
            for msg in track:
                if msg.is_meta and msg.type == "set_tempo":
                    tempo = msg.tempo
                current_time += tick2second(msg.time, midi_file.ticks_per_beat, tempo)
            if current_time > total_time:
                total_time = current_time

        return total_time
    except Exception as e:
        if debug:
            print(f"Error processing {file_path}: {e}")
        return None


# %%
# Function to walk through directories and calculate MIDI lengths for a specific artist
def get_midi_lengths_for_artist(raw_data_extracted, artist, debug = True):
    artist_directory = os.path.join(raw_data_extracted, artist)
    midi_lengths = {}
    file_count = 0
    
    for root, dirs, files in os.walk(artist_directory):
        for file in files:
            if file.endswith('.mid'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, raw_data_extracted)
                midi_length = calculate_midi_length(file_path, debug = debug)
                if midi_length is not None:
                    midi_lengths[relative_path] = midi_length
                    file_count += 1
    
    return midi_lengths, file_count

# %% [markdown]
# ### Understanding length

# %%
def get_midi_lengths_for_artists(
    raw_data_extracted, specific_artists, graph=True, debug=True
):

    # Dictionary to hold all results
    all_midi_lengths = {}
    artist_file_counts = {}

    # Get the MIDI lengths and file counts for each artist
    for artist in specific_artists:
        midi_lengths, file_count = get_midi_lengths_for_artist(
            raw_data_extracted, artist, debug=debug
        )
        all_midi_lengths.update(midi_lengths)
        artist_file_counts[artist] = file_count

    if debug:
        # Print the count of MIDI files for each artist
        for artist, count in artist_file_counts.items():
            print(f"{artist}: {count} MIDI files")

    # Create the initial DataFrame directly from the dictionary
    midi_file_lengths_df = pd.DataFrame(
        list(all_midi_lengths.items()), columns=["Path", "Length"]
    )
    midi_file_lengths_df["Artist"] = midi_file_lengths_df["Path"].apply(
        lambda x: next(
            (artist for artist in specific_artists if artist in x), "Unknown"
        )
    )

    if graph:
        # Create horizontal box plots
        plt.figure(figsize=(12, 8))
        midi_file_lengths_df.boxplot(by="Artist", column=["Length"], vert=False)
        plt.scatter(
            midi_file_lengths_df["Length"], midi_file_lengths_df["Artist"], alpha=0.5
        )
        plt.title("MIDI File Lengths by Artist")
        plt.suptitle("")
        plt.xlabel("Length (seconds)")
        plt.ylabel("Artist")
        plt.yticks(rotation=0)
        plt.show()

    return midi_file_lengths_df


paths_artist_length_data = get_midi_lengths_for_artists(
    raw_data_extracted, specific_artists
)
paths_artist_length_data.to_pickle("paths_artist_length_data.pkl")


# %% [markdown]
# ### Temple Change Augmentation to handle class imbalance.

# %%
# Data Augmentation (Pitch Shifting)
def augment_midi_pitch_shift(file_path, output_dir, shift=2):
    try:
        midi_file = MidiFile(file_path)
        new_midi_file = MidiFile()

        for track in midi_file.tracks:
            new_track = mido.MidiTrack()
            new_midi_file.tracks.append(new_track)
            for msg in track:
                if msg.type == "note_on" or msg.type == "note_off":
                    msg.note = min(max(msg.note + shift, 0), 127)
                new_track.append(msg)

        output_path = os.path.join(
            output_dir,
            os.path.basename(file_path).replace(".mid", f"_pitch_{shift}.mid"),
        )
        new_midi_file.save(output_path)

    except mido.KeySignatureError as e:
        print(f"Error processing {file_path}: {e}")
    except KeyError as e:
        print(f"KeyError processing {file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")


def process_and_augment_midi_files(
    raw_data_extracted,
    specific_artists,
    output_subdir="augmented_pitch",
    shifts=[2, -2],
):
    # Create the output directory
    augmented_pitch_dir = os.path.join(raw_data_extracted, output_subdir)
    os.makedirs(augmented_pitch_dir, exist_ok=True)

    # Walk through the directory and process MIDI files
    for root, dirs, files in os.walk(raw_data_extracted):
        for file in files:
            if file.endswith(".mid"):
                # Check if any artist name in specific_artists is in the file path
                if any(
                    artist in os.path.join(root, file) for artist in specific_artists
                ):
                    file_path = os.path.join(root, file)
                    for shift in shifts:
                        augment_midi_pitch_shift(
                            file_path, augmented_pitch_dir, shift=shift
                        )


process_and_augment_midi_files(raw_data_extracted, specific_artists)


# %% [markdown]
# Ignoring the few files that are not working, as we have a good amount of data.

# %% [markdown]
# ## Long Short-Term Memory (LSTM)
# 
# From here and down, the content is divided into types of models tried within the project: Long short-term memory (LSTM) and Convolutional Neural Network (CNN). 

# %% [markdown]
# ### Feature Extraction

# %% [markdown]
# #### Extracting Features Function

# %%
# Feature Extraction
def extract_features(file_path):
    try:
        midi_file = MidiFile(file_path)
        features = {
            "length": 0,
            "num_notes": 0,
            "note_freq": Counter(),
            "tempo_changes": [],
            "velocities": [],
            "time_sigs": Counter(),
            "key_sigs": Counter(),
            "polyphony": [],
        }

        note_on_times = {}
        polyphony_count = Counter()

        for track in midi_file.tracks:
            current_time = 0.0
            for msg in track:
                current_time += tick2second(
                    msg.time, midi_file.ticks_per_beat, bpm2tempo(120)
                )

                if msg.type == "note_on" and msg.velocity > 0:
                    features["num_notes"] += 1
                    features["note_freq"][msg.note] += 1
                    features["velocities"].append(msg.velocity)
                    if current_time in note_on_times:
                        note_on_times[current_time].append(msg.note)
                    else:
                        note_on_times[current_time] = [msg.note]
                elif msg.type == "set_tempo":
                    features["tempo_changes"].append(mido.tempo2bpm(msg.tempo))
                elif msg.type == "time_signature":
                    features["time_sigs"][(msg.numerator, msg.denominator)] += 1
                elif msg.type == "key_signature":
                    features["key_sigs"][msg.key] += 1

        features["length"] = current_time

        for time, notes in note_on_times.items():
            polyphony_count[len(notes)] += 1
        features["polyphony"] = polyphony_count

    except mido.KeySignatureError as e:
        print(f"Error processing {file_path}: {e}")
        return None
    except KeyError as e:
        print(f"KeyError processing {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
        return None

    return features


# %%
# Extract features from all MIDI files, including augmented files
features_list = []

for root, dirs, files in os.walk(raw_data_extracted):
    for file in files:
        if file.endswith(".mid"):
            for artist in specific_artists:
                if artist in os.path.join(root, file):
                    features = extract_features(os.path.join(root, file))
                    if features:
                        features["path"] = os.path.join(root, file)
                        features["artist"] = artist
                        features_list.append(features)

# Also include features from the augmented directory
for root, dirs, files in os.walk(augmented_pitch_dir):
    for file in files:
        if file.endswith(".mid"):
            for artist in specific_artists:
                if artist in os.path.join(root, file):
                    features = extract_features(os.path.join(root, file))
                    if features:
                        features["path"] = os.path.join(root, file)
                        features["artist"] = artist
                        features_list.append(features)

# Convert to DataFrame for analysis
features_list_df = pd.DataFrame(features_list)

# Print extracted features
print("Extracted features:")
features_list_df.head()


# %%
# Handling Outliers
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])


for col in ["length", "num_notes"]:
    handle_outliers(features_list_df, col)


# %% [markdown]
# #### EDA Visuals

# %%
# Distribution of MIDI file lengths by composer after outlier removal
plt.figure(figsize=(12, 6))
features_list_df.boxplot(by="artist", column=["length"], grid=False)
plt.title("Distribution of MIDI File Lengths by Composer without outliers")
plt.suptitle("")
plt.xlabel("Composer")
plt.ylabel("Length (seconds)")
plt.xticks(rotation=45)
plt.show()

# %%
# Number of notes per MIDI file by composer
plt.figure(figsize=(6, 4))
features_list_df.boxplot(by='artist', column=['num_notes'], grid=False)
plt.title('Number of Notes per MIDI File by Composer')
plt.suptitle('')
plt.xlabel('Composer')
plt.ylabel('Number of Notes')
plt.xticks(rotation=45)
plt.show()

# %%
# Distribution of note frequencies
note_freqs = Counter()
for note_counter in features_list_df['note_freq']:
    note_freqs.update(note_counter)

plt.figure(figsize=(6, 4))
plt.bar(note_freqs.keys(), note_freqs.values(), alpha=0.75)
plt.title('Distribution of Note Frequencies')
plt.xlabel('MIDI Note')
plt.ylabel('Frequency')
plt.show()

# %%
# Distribution of tempo changes
tempo_changes = [
    tempo for sublist in features_list_df["tempo_changes"] for tempo in sublist
]

plt.figure(figsize=(6, 4))
plt.hist(tempo_changes, bins=50, alpha=0.75, edgecolor="black")
plt.title("Distribution of Tempo Changes")
plt.xlabel("Tempo (BPM)")
plt.ylabel("Frequency")
plt.show()


# %%
# Distribution of polyphony
polyphony_count = Counter()
for polyphony_counter in features_list_df['polyphony']:
    polyphony_count.update(polyphony_counter)

plt.figure(figsize=(8, 6))
plt.bar(polyphony_count.keys(), polyphony_count.values(), alpha=0.75)
plt.title('Distribution of Polyphony')
plt.xlabel('Number of Simultaneous Notes')
plt.ylabel('Frequency')
plt.show()

# %%
# Time signatures by composer
time_sigs_flat = []
for idx, row in features_list_df.iterrows():
    for time_sig, count in row["time_sigs"].items():
        time_sigs_flat.append(
            {
                "artist": row["artist"],
                "time_signature": f"{time_sig[0]}/{time_sig[1]}",
                "count": count,
            }
        )

time_sigs_df = pd.DataFrame(time_sigs_flat)

plt.figure(figsize=(15, 5))
sns.countplot(data=time_sigs_df, x="time_signature", hue="artist", palette="Set3")
plt.title("Time Signatures by Composer")
plt.xlabel("Time Signature")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# %%
# Key signatures by composer
key_sigs_flat = []
for idx, row in features_list_df.iterrows():
    for key_sig, count in row["key_sigs"].items():
        key_sigs_flat.append(
            {"artist": row["artist"], "key_signature": key_sig, "count": count}
        )

key_sigs_df = pd.DataFrame(key_sigs_flat)

plt.figure(figsize=(14, 8))
sns.countplot(data=key_sigs_df, x="key_signature", hue="artist", palette="Set2")
plt.title("Key Signatures by Composer")
plt.xlabel("Key Signature")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# %%
features_list_df.to_pickle('extracted_features.pkl')

# %% [markdown]
# #### Ten Second Random Audio Samples

# %%
# Function to list MIDI files for a composer
def list_midi_files(directory, composer):
    composer_dir = os.path.join(directory, composer)
    return [
        os.path.join(composer_dir, file)
        for file in os.listdir(composer_dir)
        if file.endswith(".mid")
    ]


# Function to play a MIDI file for a specified duration
def play_midi(file_path, duration=10):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    time.sleep(duration)
    pygame.mixer.music.stop()


# Dictionary to hold a randomly selected MIDI file for each composer
selected_files = {}

# Select one random MIDI file for each composer
for composer in specific_artists:
    midi_files = list_midi_files(raw_data_extracted, composer)
    if midi_files:
        selected_files[composer] = random.choice(midi_files)
    else:
        print(f"No MIDI files found for {composer}")

# Play the selected MIDI files
for composer, file_path in selected_files.items():
    print(f"Playing {composer}'s selected MIDI file: {file_path}")
    play_midi(file_path)


# %% [markdown]
# ### Loading Dataset

# %%
## In case if we need to directly load in
# features_list_df = pd.read_pickle('extracted_features.pkl')
# features_list_df.head()

# %% [markdown]
# ### Preparing Data

# %%
# Handle missing values if any
features_list_df.fillna(0, inplace=True)

# Encode the artist labels
label_encoder = LabelEncoder()
features_list_df["artist_encoded"] = label_encoder.fit_transform(
    features_list_df["artist"]
)

# Standardize the features
scaler = StandardScaler()
numeric_features = ["length", "num_notes"]
scaled_features = scaler.fit_transform(features_list_df[numeric_features])

# Prepare sequences
X = []
y = []
sequence_length = 10  # Adjust as necessary

for i in range(len(scaled_features) - sequence_length):
    X.append(scaled_features[i : i + sequence_length])
    y.append(features_list_df["artist_encoded"].iloc[i + sequence_length])

X = np.array(X)
y = np.array(y)
y = to_categorical(y, num_classes=len(label_encoder.classes_))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# %% [markdown]
# ### Defining the LSTM Model

# %%
# Define the LSTM model
model = Sequential()
model.add(
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)
)
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(len(label_encoder.classes_), activation="softmax"))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()


# %% [markdown]
# ### Training the Model

# %%
backend.clear_session()
tf.compat.v1.reset_default_graph()

# %%
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=0,
)


# %% [markdown]
# ### Evaluating the Model

# %% [markdown]
# #### Visualizing Training History

# %%
def plot_training_history(history, figsize=(12, 4)):
    metrics = ["accuracy", "loss"]
    plt.figure(figsize=figsize)
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i + 1)

        if metric in history.history:
            plt.plot(history.history[metric])
            plt.plot(history.history[f"val_{metric}"])
            plt.title(f"Model {metric}")
            plt.ylabel(metric.capitalize())
            plt.xlabel("Epoch")
            plt.legend(["Train", "Test"], loc="upper left")
        else:
            plt.text(
                0.5,
                0.5,
                f"No {metric} data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.title(f"Model {metric}")
            plt.ylabel(metric.capitalize())
            plt.xlabel("Epoch")
    plt.tight_layout()
    plt.show()


plot_training_history(history)


# %%
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Convert encoded labels back to original
predicted_labels = label_encoder.inverse_transform(predicted_classes)
true_labels = label_encoder.inverse_transform(true_classes)

# Display some predictions
for i in range(10):
    print(f"True: {true_labels[i]}, Predicted: {predicted_labels[i]}")

# %% [markdown]
# #### Evaluation Metrics

# %%
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Convert encoded labels back to original
predicted_labels = label_encoder.inverse_transform(predicted_classes)
true_labels = label_encoder.inverse_transform(true_classes)

# Classification report
print("Classification Report:")
print(
    classification_report(
        true_classes, predicted_classes, target_names=label_encoder.classes_
    )
)

# Confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Normalize the confusion matrix
conf_matrix_normalized = (
    conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
)

# Plot normalized confusion matrix
plt.figure(figsize=(10, 7))
heatmap = sns.heatmap(
    conf_matrix_normalized,
    annot=True,
    fmt=".2f",
    cmap="rocket",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha="right")
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right")

# Annotate each cell with the numeric value
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(
            j + 0.5,
            i + 0.5,
            f"{conf_matrix_normalized[i, j]:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
        )

plt.title("Normalized Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# %% [markdown]
# ## Convolutional Neural Network (CNN)

# %% [markdown]
# ### Data Exploration
# 
# #### Understanding Structure
# 
# Understanding how data can be used for CNN using a `test_file`.

# %%
# checking how the data will look like
test_file = raw_data_extracted + "/Bach/Bwv0526 Sonate en trio n2.mid"
# Load MIDI file
midi_data = pretty_midi.PrettyMIDI(test_file)

# Generate piano roll
piano_roll = midi_data.get_piano_roll(fs=100)

# Plot piano roll
plt.figure(figsize=(30, 10))
plt.imshow(
    piano_roll, aspect="auto", origin="lower", cmap="gray_r", interpolation="nearest"
)
plt.xlabel("Time (frames)")
plt.ylabel("Pitch")
plt.title("Piano Roll Visualization")
plt.colorbar()
plt.show()


# %% [markdown]
# ### Feature Extraction
# 
# In this feature extraction process, we convert MIDI files into a multichannel piano roll to capture various aspects of the musical performance:
# 
# 1. **Binary Roll**: Captures note presence.
# 2. **Velocity Roll**: Reflects note intensity.
# 3. **Instrumentation Roll**: Shows which instruments play each note.
# 4. **Expressive Timing Roll**: Details the timing of notes.
# 

# %%
# Processes a MIDI file into a multichannel piano roll (binary, velocity, instrumentation, timing).
def process_multichannel_midi(file_path, fs=10, max_length=100):
    midi_data = pretty_midi.PrettyMIDI(file_path)

    # Binary and velocity piano rolls
    piano_roll = midi_data.get_piano_roll(fs=fs)
    binary_piano_roll = (piano_roll > 0).astype(int)
    velocity_roll = piano_roll / 127  # Normalize velocity

    # Combining instrument rolls, adjusting for length
    instrument_rolls = []
    for instrument in midi_data.instruments:
        inst_roll = instrument.get_piano_roll(fs=fs)
        instrument_rolls.append(inst_roll)

    max_instrument_length = max(inst.shape[1] for inst in instrument_rolls)
    combined_instrument_roll = np.zeros((128, max_instrument_length))
    for inst_roll in instrument_rolls:
        if inst_roll.shape[1] < max_instrument_length:
            # Pad to the right if shorter
            padding = np.zeros((128, max_instrument_length - inst_roll.shape[1]))
            inst_roll = np.hstack((inst_roll, padding))
        combined_instrument_roll += (inst_roll > 0).astype(int)

    # Creating expressive timing roll
    expressive_timing_roll = np.zeros((128, max_instrument_length))
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = int(note.start * fs)
            end = int(note.end * fs)
            expressive_timing_roll[note.pitch, start:end] = 1

    # Adjusting rolls to match the maximum length
    if max_instrument_length > max_length:
        combined_instrument_roll = combined_instrument_roll[:, :max_length]
        expressive_timing_roll = expressive_timing_roll[:, :max_length]
    elif max_instrument_length < max_length:
        padding = np.zeros((128, max_length - max_instrument_length))
        combined_instrument_roll = np.hstack((combined_instrument_roll, padding))
        expressive_timing_roll = np.hstack((expressive_timing_roll, padding))

    binary_piano_roll = binary_piano_roll[:, :max_length]
    velocity_roll = velocity_roll[:, :max_length]

    # Stacking all channels into a multichannel roll
    multichannel_roll = np.stack(
        [
            binary_piano_roll,
            velocity_roll,
            combined_instrument_roll,
            expressive_timing_roll,
        ],
        axis=-1,
    )

    return multichannel_roll


# %%
# Plotting each channel of the processed multichannel piano roll data
def plot_multichannel_piano_roll(processed_data):
    # Unpacking the channels
    binary_channel = processed_data[:, :, 0]
    velocity_channel = processed_data[:, :, 1]
    instrument_channel = processed_data[:, :, 2]
    expressive_timing_channel = processed_data[:, :, 3]

    # Setting up the plot
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 16))
    titles = [
        "Binary Channel",
        "Velocity Channel",
        "Instrumentation Channel",
        "Expressive Timing Channel",
    ]

    # Plotting each channel
    for ax, channel, title in zip(
        axes,
        [
            binary_channel,
            velocity_channel,
            instrument_channel,
            expressive_timing_channel,
        ],
        titles,
    ):
        cax = ax.imshow(channel, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Pitch")
        fig.colorbar(cax, ax=ax, orientation="vertical")

    plt.tight_layout()
    plt.show()


# %% [markdown]
# #### Testing Frames Per Second (FPS)
# 
# Determining the optimal placement for frames per second (FPS) using visual aids. Trying to see how much of the visual detail is being compressed.

# %%
processed_data = process_multichannel_midi(test_file, fs=16)
plot_multichannel_piano_roll(processed_data)

# %%
processed_data = process_multichannel_midi(test_file, fs=8)
plot_multichannel_piano_roll(processed_data)

# %%
processed_data = process_multichannel_midi(test_file, fs=4)
plot_multichannel_piano_roll(processed_data)

# %% [markdown]
# A number around 8 sounds good, will be using 10, just to make it a little compressed.

# %% [markdown]
# ### Preparing Data

# %% [markdown]
# #### Chunks
# 
# We divided MIDI files into chunks to classify segments by the artist. This approach reduces bias from different file lengths and focuses on smaller sections instead of the full piece. Also, there is a function to visualize the chunk if needed. 
# 
# Also, there is an overlap of information between Binary and Expressive Timing Roll and also between Velocity and Instrumentation Roll. So, we ended up only using the Binary and Velocity. 

# %%
def midi_to_chunks(file_path, chunk_size=150, fs=10):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    piano_roll = midi_data.get_piano_roll(fs=fs)
    num_chunks = piano_roll.shape[1] // chunk_size

    chunks = []

    # Creating fixed-size chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = piano_roll[:, start:end]

        # Converting to binary and velocity channels
        binary = (chunk > 0).astype(int)
        velocity = chunk / 127

        # Stacking channels
        multichannel_chunk = np.stack([binary, velocity], axis=-1)
        chunks.append(multichannel_chunk)

    # Handling the last chunk if it doesn't fit perfectly
    if piano_roll.shape[1] % chunk_size != 0:
        last_chunk = piano_roll[:, num_chunks * chunk_size :]
        if last_chunk.shape[1] < chunk_size:
            padding = np.zeros((128, chunk_size - last_chunk.shape[1], 2))
            last_chunk_padded = np.stack(
                [(last_chunk > 0).astype(int), last_chunk / 127], axis=-1
            )
            last_chunk_padded = np.concatenate([last_chunk_padded, padding], axis=1)
            chunks.append(last_chunk_padded)

    return chunks


file_path = test_file
chunks = midi_to_chunks(file_path)


# %%
def visualize_chunks(chunks):
    num_chunks = len(chunks)
    fig, axes = plt.subplots(
        num_chunks, 2, figsize=(15, 3 * num_chunks)
    )  # 2 columns for binary and velocity

    if num_chunks == 1:
        axes = [axes]

    for i, chunk in enumerate(chunks):
        # Binary Channel
        ax1 = axes[i][0] if num_chunks > 1 else axes[0]
        binary_channel = chunk[:, :, 0]  # Assuming binary channel is the first channel
        cax1 = ax1.imshow(
            binary_channel,
            aspect="auto",
            origin="lower",
            cmap="gray",
            interpolation="none",
        )
        ax1.set_title(f"Chunk {i+1} - Binary Channel")
        ax1.set_xlabel("Time (frames)")
        ax1.set_ylabel("Pitch")
        fig.colorbar(cax1, ax=ax1, orientation="vertical")

        # Velocity Channel
        ax2 = axes[i][1] if num_chunks > 1 else axes[1]
        velocity_channel = chunk[
            :, :, 1
        ]  # Assuming velocity channel is the second channel
        cax2 = ax2.imshow(
            velocity_channel,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="none",
        )
        ax2.set_title(f"Chunk {i+1} - Velocity Channel")
        ax2.set_xlabel("Time (frames)")
        ax2.set_ylabel("Pitch")
        fig.colorbar(cax2, ax=ax2, orientation="vertical")

    plt.tight_layout()
    plt.show()


# %% [markdown]
# #### Creating `DataFrame` of MIDI Files

# %%
# In case if we want to read from a previous run file. 
# paths_artist_length_data  = pd.read_pickle('paths_artist_length_data.pkl')

# %%
# Constructing the full file path if necessary
def construct_file_path(base_url, relative_path):
    if not relative_path.startswith(base_url):
        return f"{base_url}/{relative_path}"
    return relative_path


# Iterating over each file to create chunks


def process_all_files(df, base_url, fs=10, chunk_size=150):
    all_chunks = []

    for idx, row in df.iterrows():
        file_path = construct_file_path(base_url, row["path"])
        artist = row["artist"]
        chunks = midi_to_chunks(file_path, chunk_size=chunk_size, fs=fs)

        # Collecting chunks with additional metadata
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                [chunk, artist, row["path"], i + 1, chunk.shape[1] < chunk_size]
            )

    # Creating a DataFrame
    columns = ["Chunk", "Artist", "Original Path", "Chunk Number", "Padding Added"]
    chunk_df = pd.DataFrame(all_chunks, columns=columns)

    return chunk_df


processed_chunk_df = process_all_files(
    paths_artist_length_data, raw_data_extracted, fs=8, chunk_size=150
)
processed_chunk_df.to_pickle("processed_chunk_df.pkl")


# %% [markdown]
# #### Synthetic Data Check

# %%
print('How many chunks has padding')
print(processed_chunk_df["Padding Added"].value_counts())
print(processed_chunk_df["Padding Added"].value_counts(normalize=True) * 100)

# %% [markdown]
# Pretty good percentage.

# %%
processed_chunk_df['Chunk'].iloc[0].shape

# %% [markdown]
# #### Input Feature (X)

# %%
# Preprocessing chunks from a DataFrame into a format suitable (numpy array) for CNN input
def preprocess_chunks(dataframe, chunk_size=150):

    processed_chunks = []

    for chunk in dataframe["Chunk"]:
        if isinstance(chunk, np.ndarray):
            if chunk.shape[1] != chunk_size:
                if chunk.shape[1] < chunk_size:
                    padding = np.zeros((128, chunk_size - chunk.shape[1]))
                    chunk = np.hstack((chunk, padding))
                else:
                    chunk = chunk[:, :chunk_size]
            processed_chunks.append(chunk)
        else:
            print("Chunk is not a numpy array. Check data preparation steps.")

    # Normalizing data as well
    X = np.stack(processed_chunks) / 127.0
    X = X.reshape(-1, 128, chunk_size, 1)
    return X


X = preprocess_chunks(processed_chunk_df)


# %%
X.shape

# %% [markdown]
# #### Target Variable (y)

# %%
# Encoding labels into one-hot format and returning the encoder
def encode_labels(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(-1, 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded, label_encoder


y, label_encoder = encode_labels(processed_chunk_df["Artist"])

# %%
y.shape

# %%
X, y = shuffle(X, y, random_state=42)

# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ### Defining the CNN Model

# %%
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=X.shape[1:]),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(len(np.unique(processed_chunk_df["Artist"])), activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


# %% [markdown]
# ### Training the Model

# %%
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# %% [markdown]
# ### Evaluating the Model

# %% [markdown]
# #### Visualizing Training History

# %%
plot_training_history(history)

# %% [markdown]
# #### Evaluation Metrics

# %%
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation accuracy: {val_accuracy:.2f}, Validation loss: {val_loss:.2f}')

# %%
predictions = model.predict(X_val)
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
# Print the top 5 predictions
for i in range(min(5, len(predictions))):
    print(f"{predicted_labels[i]}")


# %% [markdown]
# Beethoven
# 
# Bach
# 
# Bach
# 
# Bach
# 
# Bach

# %% [markdown]
# ### Optimization
# 
# We built a CNN model with adjustable parameters and used `RandomizedSearchCV` to find the optimal combination of hyperparameters like optimizer, initializer, dropout rate, epochs, and batch size to achieve the best model performance.

# %%
# Model to test different parameters against
def create_model(optimizer="adam", init="glorot_uniform", dropout_rate=0.5):
    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_initializer=init,
                input_shape=X.shape[1:],
            ),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate),
            Conv2D(64, (3, 3), activation="relu", kernel_initializer=init),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate),
            Flatten(),
            Dense(128, activation="relu", kernel_initializer=init),
            Dropout(dropout_rate),
            Dense(
                len(np.unique(processed_chunk_df["Artist"])),
                activation="softmax",
                kernel_initializer=init,
            ),
        ]
    )

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


model = KerasClassifier(build_fn=create_model, verbose=1)


# %%
# Parameter grid for RandomizedSearchCV
param_grid = {
    "optimizer": ["adam", "sgd"],
    "init": ["glorot_uniform", "he_normal"],
    "dropout_rate": [0.4, 0.5],
    "epochs": [10, 20],
    "batch_size": [20, 30],
}

# Initialize and run RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model, param_distributions=param_grid, n_iter=10, cv=3, verbose=1
)
random_search_result = random_search.fit(X, y)

print(
    "Best: %f using %s"
    % (random_search_result.best_score_, random_search_result.best_params_)
)

# Please do not mind the scrolling. We initially decided to remove the output
# but decided to keep the results at the last minute as they provide useful
# information to track back.


# %% [markdown]
# #### Best Model Parameters
# 
# Following are the specific value of the best model:
# 
# | Parameter           | Value           |
# |---------------------|-----------------|
# | Optimizer           | adam            |
# | Initialization      | glorot_uniform  |
# | Epochs              | 10              |
# | Dropout Rate        | 0.4             |
# | Batch Size          | 30              |

# %%
# Saving the best model
best_model = random_search_result.best_estimator_
best_model.model.save('best_trained_model.h5')
print("Model saved to best_trained_model.h5")

# %% [markdown]
# #### Best Model Evaluation

# %%
# Evaluate on the validation set
val_loss, val_accuracy = best_model.model.evaluate(X_val, y_val)
print(f'Validation accuracy: {val_accuracy:.2f}, Validation loss: {val_loss:.2f}')

# %% [markdown]
# The best cross-validation accuracy achieved during optimization was 0.7702.
# 
# In terms of the Validation and Training, here is the comparison: 
# 
# | Metric               | Before Optimization | After Optimization | Improvement Amount |
# |----------------------|---------------------|--------------------|--------------------|
# | **Validation Accuracy** | 0.80                | 0.98               | +0.18               |
# | **Validation Loss**     | 0.73                | 0.08               | -0.65               |
# | **Training Accuracy**   | 0.8021              | 0.9797             | +0.1776             |
# | **Training Loss**       | 0.7263              | 0.0836             | -0.6427             |
# 
# 
# Overall a huge increase in both accuracies and decrease in losses.

# %% [markdown]
# ## All Artists Inclusive Analysis
# 
# Rather than just ending the project with small part of the data, we decided to take the best CNN model and test the full data set against it.

# %% [markdown]
# ### Loading Dataset

# %% [markdown]
# Loading all the artists.

# %%
all_artists = get_all_artists(raw_data_extracted)

paths_artist_length_data_all = get_midi_lengths_for_artists(
    raw_data_extracted, all_artists, graph=False, debug=False
)
paths_artist_length_data_all.to_pickle("paths_artist_length_data_all.pkl")


# %% [markdown]
# Not a lot of files within some of artists, but since we have chunks, we should have more than one data points of each artist. 

# %% [markdown]
# ### Preparing Data

# %%
processed_chunk_all_df = process_all_files(
    paths_artist_length_data_all, raw_data_extracted, fs=8, chunk_size=150
)
processed_chunk_all_df.to_pickle("processed_chunk_all_artist.pkl")


# %%
X_all = preprocess_chunks(processed_chunk_all_df)
y_all, label_encoder_all = encode_labels(processed_chunk_all_df['Artist'])

# %%
X_all, y_all = shuffle(X_all, y_all, random_state=42)

# %%
X_train_all, X_val_all, y_train_all, y_val_all = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)


# %%
X_all.shape

# %%
y_all.shape

# %% [markdown]
# ### Defining Model

# %%
def create_best_model(optimizer="adam", init="glorot_uniform", dropout_rate=0.4):
    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_initializer=init,
                input_shape=X.shape[1:],
            ),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate),
            Conv2D(64, (3, 3), activation="relu", kernel_initializer=init),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate),
            Flatten(),
            Dense(128, activation="relu", kernel_initializer=init),
            Dropout(dropout_rate),
            Dense(
                len(np.unique(processed_chunk_df["Artist"])),
                activation="softmax",
                kernel_initializer=init,
            ),
        ]
    )

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# %%
model = KerasClassifier(build_fn=create_best_model, epochs=10, batch_size=30, verbose=0)

# %% [markdown]
# ### Training Model

# %%
history = model.fit(X_train_all, y_train_all, validation_data=(X_val_all, y_val_all))

# %%
model.model.save("best_model_all_artists.h5")

# %% [markdown]
# ### Evaluating the Model

# %%
plot_training_history(history)

# %%
print(f"Training Loss: {history.history['loss'][-1]}")
print(f"Training Accuracy: {history.history['accuracy'][-1]}")
print(f"Validation Loss: {history.history['val_loss'][-1]}")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]}")

# %% [markdown]
# These are quite lower than the first four artists, especially with the best-case model. We have 147 unique artists for this analysis, which is quite high compared to only four. 
# 
# Here's the table comparing the model performance for the evaluation with four artists and all 147 artists:
# 
# 
# | Metric                 | Four Artists       | All Artists         | Difference (All - Four) |
# |------------------------|--------------------|---------------------|-------------------------|
# | **Training Loss** | 0.0836             | 0.3015              | +0.2179                 |
# | **Training Accuracy** | 97.97%             | 87.88%              | -10.09%                 |
# | **Validation Loss** | 0.08               | 0.6082              | +0.5282                 |
# | **Validation Accuracy** | 98%                | 79.06%              | -18.94%                 |
# 

# %% [markdown]
# Overall, this project was quite fun for all of us. Not only did we learn quite a lot, but we also achieved great accuracy and optimization. We also got to try on full data, which was initially the main wish, as our data preparation was designed to include all the MIDI files and structure all the files quite nicely. 

# %% [markdown]
# ## Future Plan
# 
# If we had more GPU power, it would be great to optimization the best model for all the artists. Just the small optimization of CNN took us several hours of training within our machines, and even on NVIDIA A100 (40 GB), it took quite a while to get everything running and optimized. We had to pull some parameters out due to minimum resources. 
# 
# Additionally, it would be great to create a demo where we can give it a random MIDI chunk and get a prediction, similar to Shazam. We had written some code for this, but nothing was complete for an MVP. It would be great to go back and get the MVP done. 


