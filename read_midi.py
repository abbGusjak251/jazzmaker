import mido 
import sys
import os

arg = 0
try: 
    arg = int(sys.argv[1])
    fname = sys.argv[2]
except:
    print("Failed at lines 6 n 7")
    exit()
mid = mido.MidiFile(fname)
#print(mid.tracks)
for t in mid.tracks[1:]:
    print(t)
    dataset = []
    note_on = 0
    if t.note == 'control_change':
        note_on = 1
    dataset.append(note_on)
    dataset.append(t.velocity)
    dataset.append(t.time)
    print(dataset)
#exit()

if not ".mid" in fname:
    fname += ".mid"
if "midi/" in fname:
    fname = fname.replace("midi/", "")
print(f"Filename: {fname}")
print(f"Tracks: {len(mid.tracks)}")
print(f"Bouncing track: {arg}")
print("-"*30)
#for i, track in enumerate(mid.tracks):
    #print('Track {}: {}'.format(i, track.name))
    #for msg in track:
        #print(msg)
try:
    new_tracks = mid.tracks[:1]
    new_tracks.append(mid.tracks[arg])
except:
    print("error")
    exit()

mid.tracks = new_tracks

dirname = f'.\midi_tracks\{fname.replace(".mid", "")}'
if not os.path.exists(dirname):
    os.system(f'mkdir {dirname}')
mid.save(f'{dirname}/track_{arg}_{fname}')


