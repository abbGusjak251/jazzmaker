import mido 
import sys

def get_len(file):
    try:    
        mid = mido.MidiFile(file)
    except:
        print("error")
        exit()
    return len(mid.tracks)

