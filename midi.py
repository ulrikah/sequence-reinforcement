import mido
import numpy as np

class Player:
    def __init__(self, port=None):
        self.bpm = 120
        if port is not None:
            assert port in mido.get_output_names(), f"{port} not recognized as a MIDI port"
            self.port = port
        else:
            ports = mido.get_output_names()
            print("No output port selected. Please choose one of:", ", ".join(ports))

    def play_midi_file(self, mid: mido.MidiFile):
        with mido.open_output(self.port) as port:
            for msg in mid.play():
                print(msg)
                port.send(msg)

    def seq_to_track(self, seq : np.ndarray, note):
        '''Parses a numpy sequence to a mido.MidiTrack'''
        DUR = 300
        delta = DUR
        track = mido.MidiTrack()
        for step in seq.tolist():
            if step == 1:
                track.append(mido.Message('note_on', note=note, time=DUR, velocity=120))
                track.append(mido.Message('note_off', note=note, time=0, velocity=120))
            delta += delta
        return track

if __name__ == "__main__":
    player = Player(port="IAC-driver Buss 1")
    kick = np.array([1.0, 0.0, 0.0, 0.0] * 2)
    snare = np.array([0.0, 0.0, 0.0, 1.0] * 2)
    t = player.seq_to_track(kick, note=40)
    s = player.seq_to_track(snare, note=60)
    outfile = mido.MidiFile()
    outfile.tracks.append(t)
    outfile.tracks.append(s)
    player.play_midi_file(outfile)
