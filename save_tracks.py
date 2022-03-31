import os
import sys
from get_number_of_tracks import get_len

def save_tracks(file):
    length = get_len(file)
    for i in range(1, length):
        os.system(f"python read_midi.py {i} {file}")

try:
    file = sys.argv[1]
except Exception as e:
    file = os.listdir('.\midi')
    print(file)
    for f in file:
        save_tracks(f"midi/{f}")
    exit()
save_tracks(file)