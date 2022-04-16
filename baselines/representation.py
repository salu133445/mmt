"""Representation utilities."""
import pathlib
import pprint
from collections import defaultdict
from email.policy import default

import muspy
import numpy as np
import pretty_midi

import utils

# Configuration
RESOLUTION = 24
MAX_BEAT = 1024
MAX_TIME_SHIFT = RESOLUTION * 4

# Instrument
PROGRAM_INSTRUMENT_MAP = {
    # Pianos
    0: "piano",
    1: "piano",
    2: "piano",
    3: "piano",
    4: "electric-piano",
    5: "electric-piano",
    6: "harpsichord",
    7: "clavinet",
    # Chromatic Percussion
    8: "celesta",
    9: "glockenspiel",
    10: "music-box",
    11: "vibraphone",
    12: "marimba",
    13: "xylophone",
    14: "tubular-bells",
    15: "dulcimer",
    # Organs
    16: "organ",
    17: "organ",
    18: "organ",
    19: "church-organ",
    20: "organ",
    21: "accordion",
    22: "harmonica",
    23: "bandoneon",
    # Guitars
    24: "nylon-string-guitar",
    25: "steel-string-guitar",
    26: "electric-guitar",
    27: "electric-guitar",
    28: "electric-guitar",
    29: "electric-guitar",
    30: "electric-guitar",
    31: "electric-guitar",
    # Basses
    32: "bass",
    33: "electric-bass",
    34: "electric-bass",
    35: "electric-bass",
    36: "slap-bass",
    37: "slap-bass",
    38: "synth-bass",
    39: "synth-bass",
    # Strings
    40: "violin",
    41: "viola",
    42: "cello",
    43: "contrabass",
    44: "strings",
    45: "strings",
    46: "harp",
    47: "timpani",
    # Ensemble
    48: "strings",
    49: "strings",
    50: "synth-strings",
    51: "synth-strings",
    52: "voices",
    53: "voices",
    54: "voices",
    55: "orchestra-hit",
    # Brass
    56: "trumpet",
    57: "trombone",
    58: "tuba",
    59: "trumpet",
    60: "horn",
    61: "brasses",
    62: "synth-brasses",
    63: "synth-brasses",
    # Reed
    64: "soprano-saxophone",
    65: "alto-saxophone",
    66: "tenor-saxophone",
    67: "baritone-saxophone",
    68: "oboe",
    69: "english-horn",
    70: "bassoon",
    71: "clarinet",
    # Pipe
    72: "piccolo",
    73: "flute",
    74: "recorder",
    75: "pan-flute",
    76: None,
    77: None,
    78: None,
    79: "ocarina",
    # Synth Lead
    80: "lead",
    81: "lead",
    82: "lead",
    83: "lead",
    84: "lead",
    85: "lead",
    86: "lead",
    87: "lead",
    # Synth Pad
    88: "pad",
    89: "pad",
    90: "pad",
    91: "pad",
    92: "pad",
    93: "pad",
    94: "pad",
    95: "pad",
    # Synth Effects
    96: None,
    97: None,
    98: None,
    99: None,
    100: None,
    101: None,
    102: None,
    103: None,
    # Ethnic
    104: "sitar",
    105: "banjo",
    106: "shamisen",
    107: "koto",
    108: "kalimba",
    109: "bag-pipe",
    110: "violin",
    111: "shehnai",
    # Percussive
    112: None,
    113: None,
    114: None,
    115: None,
    116: None,
    117: "melodic-tom",
    118: "synth-drums",
    119: "synth-drums",
    120: None,
    # Sound effects
    121: None,
    122: None,
    123: None,
    124: None,
    125: None,
    126: None,
    127: None,
    128: None,
}
INSTRUMENT_PROGRAM_MAP = {
    # Pianos
    "piano": 0,
    "electric-piano": 4,
    "harpsichord": 6,
    "clavinet": 7,
    # Chromatic Percussion
    "celesta": 8,
    "glockenspiel": 9,
    "music-box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "tubular-bells": 14,
    "dulcimer": 15,
    # Organs
    "organ": 16,
    "church-organ": 19,
    "accordion": 21,
    "harmonica": 22,
    "bandoneon": 23,
    # Guitars
    "nylon-string-guitar": 24,
    "steel-string-guitar": 25,
    "electric-guitar": 26,
    # Basses
    "bass": 32,
    "electric-bass": 33,
    "slap-bass": 36,
    "synth-bass": 38,
    # Strings
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "harp": 46,
    "timpani": 47,
    # Ensemble
    "strings": 49,
    "synth-strings": 50,
    "voices": 52,
    "orchestra-hit": 55,
    # Brass
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "horn": 60,
    "brasses": 61,
    "synth-brasses": 62,
    # Reed
    "soprano-saxophone": 64,
    "alto-saxophone": 65,
    "tenor-saxophone": 66,
    "baritone-saxophone": 67,
    "oboe": 68,
    "english-horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    # Pipe
    "piccolo": 72,
    "flute": 73,
    "recorder": 74,
    "pan-flute": 75,
    "ocarina": 79,
    # Synth Lead
    "lead": 80,
    # Synth Pad
    "pad": 88,
    # Ethnic
    "sitar": 104,
    "banjo": 105,
    "shamisen": 106,
    "koto": 107,
    "kalimba": 108,
    "bag-pipe": 109,
    "shehnai": 111,
    # Percussive
    "melodic-tom": 117,
    "synth-drums": 118,
}
KNOWN_INSTRUMENTS = list(dict.fromkeys(INSTRUMENT_PROGRAM_MAP.keys()))


class Indexer:
    def __init__(self, data=None, is_learning=False):
        self._dict = dict() if data is None else data
        self._n_words = 0 if data is None else len(data)
        self._is_learning = is_learning

    def __getitem__(self, key):
        if self._is_learning and key not in self._dict:
            self._dict[key] = self._n_words
            self._n_words += 1
            return self._n_words - 1
        return self._dict[key]

    def __len__(self):
        return self._n_words

    def __contain__(self, item):
        return item in self._dict

    def get_dict(self):
        """Return the internal dictionary."""
        return self._dict

    def learn(self, is_learning):
        """Set learning mode."""
        self._is_learning = is_learning


def get_encoding():
    """Return the encoding configurations."""
    return {
        "resolution": RESOLUTION,
        "max_beat": MAX_BEAT,
        "max_time_shift": MAX_TIME_SHIFT,
        "program_instrument_map": PROGRAM_INSTRUMENT_MAP,
        "instrument_program_map": INSTRUMENT_PROGRAM_MAP,
    }


def load_encoding(filename):
    """Load encoding configurations from a JSON file."""
    encoding = utils.load_json(filename)
    encoding["program_instrument_map"] = {
        int(k) if k != "null" else None: v
        for k, v in encoding["program_instrument_map"].items()
    }
    return encoding


def encode(music, encoding, indexer):
    """Encode the notes into a sequence of code tuples.

    Each row of the output is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Get variables
    resolution = encoding["resolution"]
    max_beat = encoding["max_beat"]
    max_time_shift = encoding["max_time_shift"]

    # Get maps
    program_instrument_map = encoding["program_instrument_map"]

    # Check resolution
    assert music.resolution == resolution

    # Extract notes
    events = defaultdict(list)
    for track in music:
        instrument = program_instrument_map[track.program]
        # Skip unknown instruments
        if instrument is None:
            continue
        for note in track:
            if note.time // resolution > max_beat:
                continue
            events[instrument].append((note.time, f"note-on_{note.pitch}"))
            events[instrument].append((note.end, f"note-off_{note.pitch}"))

    # Deduplicate and sort the notes
    for instrument in events:
        events[instrument] = sorted(set(events[instrument]))

    # Start the codes with an SOS row
    codes = [indexer["start-of-song"]]

    # Encode the instruments
    for instrument in events:
        codes.append(indexer["start-of-track"])
        codes.append(indexer[f"instrument_{instrument}"])
        time = 0
        for event_time, event in events[instrument]:
            while event_time < time:
                time_shift = min(event_time - time, max_time_shift)
                codes.append(indexer[f"time-shift_{time_shift}"])
                time += time_shift
            codes.append(indexer[event])
        codes.append(indexer["end-of-track"])

    # End the codes with an EOS row
    codes.append(indexer["end-of-song"])

    return np.array(codes)


def decode(data, encoding, vocabulary):
    """Decode codes to notes.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Get variables and maps
    resolution = encoding["resolution"]
    instrument_program_map = encoding["instrument_program_map"]

    # Decode the codes into a sequence of notes
    tracks = []
    time = 0
    track = None
    for code in data:
        event = vocabulary[code]
        if event == "start-of-song":
            continue
        elif event == "end-of-song":
            break
        elif event in ("start-of-track", "end-of-track"):
            # Append decoded track to the list
            tracks.append(track)
            # Reset variables
            track = muspy.Track()
            time = 0
            note_ons = {}
        elif event.startswith("instrument"):
            if track is None:
                continue
            instrument = event.split("_")[1]
            track.program = instrument_program_map[instrument]
        elif event.startswith("time-shift"):
            time += event.split("_")[1]
        elif event.startswith("note-on"):
            if track is None:
                continue
            pitch = int(event.split("_")[1])
            note_ons[pitch] = time
        elif event.startswith("note-off"):
            if track is None:
                continue
            pitch = int(event.split("_")[1])
            # Skip a note-off event without a corresponding note-on event
            if pitch not in note_ons:
                continue
            track.notes.append(
                muspy.Note(note_ons[pitch], pitch, time - note_ons[pitch])
            )
        else:
            raise ValueError("Unknown event type.")

    # Construct the MusPy Music object
    music = muspy.Music(resolution=resolution, tracks=tracks)

    return music


def dump(data, vocabulary):
    """Decode the codes and dump as a string."""
    # Iterate over the rows
    lines = []
    for code in data:
        event = vocabulary[code]
        if (
            event in ("start-of-song", "start-of-track", "end-of-track")
            or event.startswith("instrument")
            or event.startswith("time-shift")
            or event.startswith("note-on")
            or event.startswith("note-off")
        ):
            lines.append(event)
        elif event == "end-of-song":
            lines.append(event)
            break
        else:
            raise ValueError("Unknown event type.")

    return "\n".join(lines)


def save_csv(filename, data):
    """Save the representation as a CSV file."""
    utils.save_csv(
        filename, data, header="type,beat,position,pitch,duration,instrument"
    )


def main():
    """Main function."""
    # Get the encoding
    encoding = get_encoding()

    # Save the encoding
    filename = pathlib.Path(__file__).parent / "encoding.json"
    utils.save_json(filename, encoding)

    # Load encoding
    encoding = load_encoding(filename)

    # Print the maps
    print(f"{' Maps ':=^40}")
    for key, value in encoding.items():
        if key in ("program_instrument_map", "instrument_program_map"):
            print("-" * 40)
            print(f"{key}:")
            pprint.pprint(value, indent=2)

    # Load the example
    music = muspy.load(pathlib.Path(__file__).parent / "example.json")

    # Get the indexer
    indexer = Indexer(is_learning=True)

    # Encode the music
    encoded = encode(music, encoding, indexer)
    print(f"Codes:\n{encoded}")

    # Save the learned indexer
    filename = pathlib.Path(__file__).parent / "indexer.json"
    utils.save_json(filename, indexer.get_dict())

    # Load the indexer
    loaded = utils.load_json(filename)
    loaded_indexer = Indexer(loaded)

    # Get the learned vocabulary
    vocabulary = utils.inverse_dict(loaded_indexer.get_dict())

    print("-" * 40)
    print(f"Decoded:\n{dump(encoded, vocabulary)}")

    music = decode(encoded, encoding, vocabulary)
    print(f"Decoded musics:\n{music}")


if __name__ == "__main__":
    main()
