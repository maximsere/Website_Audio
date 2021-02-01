import numpy as np
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/user1/desktop/STT/My First Project-2c6ce4f254eb.json"
from google.cloud import speech_v1 as speech
import argparse
import io
import soundfile as sf
import openpyxl
import pandas as pd
from urllib.request import urlopen
from google.cloud import storage
from pydub import AudioSegment

def levenshtein_ratio_and_distance(s, t, ratio_calc = True):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])

def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    from google.cloud import speech
    import io

    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        audio_channel_count=2,
        enable_separate_recognition_per_channel=True,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        return "{}".format(result.alternatives[0].transcript)

# Open the QAnalyst script and process it into individual utterances
file = open("QAnalyst_script.data")
data = file.read()
data_split = data.split(".")

# Clean the individual utterances to reduce mistranscriptions
new_data_split = []
for lines in data_split:
    if len(lines) == 0:
        data_split.remove(lines)
    else:
        lines = lines[1:]
        new_data_split.append(lines)

df = pd.DataFrame(columns = ['filename', 'stt_text', 'max_lev_dist_text', 'ratio'])

for filename in os.listdir('./qanalyst_phrases'):
    print(filename)
    stt_output = transcribe_file('./qanalyst_phrases/' + filename)
    lev_ratio_list = []
    # Calculate the L-distance between the transcribe utterance and all the possible utterances
    if stt_output != None:
        for lines in new_data_split:
            lev_number = levenshtein_ratio_and_distance(stt_output, lines)
            lev_ratio_list.append(lev_number)
        # Take the max of the list and output it into a new df row
        max_lev = max(lev_ratio_list)
        index_of_max = lev_ratio_list.index(max_lev)
        df.loc[len(df.index)] = [filename, stt_output, new_data_split[index_of_max], max_lev]
    else:
        df.loc[len(df.index)] = [filename, stt_output, "", ""]

df.to_csv('QAnalyst_stt_data.csv')