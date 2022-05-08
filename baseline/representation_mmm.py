"""Representation utilities."""
import pathlib
import pprint
from collections import defaultdict

import muspy
import numpy as np

import utils

# Configuration
RESOLUTION = 12
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
KNOWN_PROGRAMS = list(
    k for k, v in INSTRUMENT_PROGRAM_MAP.items() if v is not None
)
KNOWN_INSTRUMENTS = list(dict.fromkeys(INSTRUMENT_PROGRAM_MAP.keys()))

KNOWN_EVENTS = [
    "start-of-song",
    "end-of-song",
    "start-of-track",
    "end-of-track",
]
KNOWN_EVENTS.extend(
    f"instrument_{instrument}" for instrument in KNOWN_INSTRUMENTS
)
KNOWN_EVENTS.extend(f"note-on_{i}" for i in range(128))
KNOWN_EVENTS.extend(f"note-off_{i}" for i in range(128))
KNOWN_EVENTS.extend(f"time-shift_{i}" for i in range(1, MAX_TIME_SHIFT + 1))
EVENT_CODE_MAPS = {event: i for i, event in enumerate(KNOWN_EVENTS)}
CODE_EVENT_MAPS = utils.inverse_dict(EVENT_CODE_MAPS)


class Indexer:
    def __init__(self, data=None, is_training=False):
        self._dict = dict() if data is None else data
        self._is_training = is_training

    def __getitem__(self, key):
        if self._is_training and key not in self._dict:
            self._dict[key] = len(self._dict)
            return len(self._dict) - 1
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __contain__(self, item):
        return item in self._dict

    def get_dict(self):
        """Return the internal dictionary."""
        return self._dict

    def train(self):
        """Set training mode."""
        self._is_training = True

    def eval(self):
        """Set evaluation mode."""
        self._is_learning = False


def get_encoding():
    """Return the encoding configurations."""
    return {
        "resolution": RESOLUTION,
        "max_beat": MAX_BEAT,
        "max_time_shift": MAX_TIME_SHIFT,
        "program_instrument_map": PROGRAM_INSTRUMENT_MAP,
        "instrument_program_map": INSTRUMENT_PROGRAM_MAP,
        "event_code_map": EVENT_CODE_MAPS,
        "code_event_map": CODE_EVENT_MAPS,
    }


def load_encoding(filename):
    """Load encoding configurations from a JSON file."""
    encoding = utils.load_json(filename)
    for key in ("program_instrument_map", "code_event_map"):
        encoding[key] = {
            int(k) if k != "null" else None: v
            for k, v in encoding[key].items()
        }
    return encoding


def extract_notes(music, resolution):
    """Return a MusPy music object as a note sequence.

    Each row of the output is a note specified as follows.

        (beat, position, pitch, duration, program)

    """
    # Check resolution
    assert music.resolution == resolution

    # Extract notes
    notes = []
    for track in music:
        if track.program not in KNOWN_PROGRAMS:
            continue
        for note in track:
            beat, position = divmod(note.time, resolution)
            notes.append(
                (beat, position, note.pitch, note.duration, track.program)
            )

    # Deduplicate and sort the notes
    notes = sorted(set(notes))

    return np.array(notes)


def encode_notes(notes, encoding, indexer):
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
    instrument_program_map = encoding["instrument_program_map"]

    # Extract notes
    instruments = defaultdict(list)
    for note in notes:
        instrument = program_instrument_map[note[-1]]
        # Skip unknown instruments
        if instrument is None:
            continue
        instruments[instrument].append(note)

    # Sort the instruments
    instruments = dict(
        sorted(
            instruments.items(),
            key=lambda x: instrument_program_map[x[0]],
        )
    )

    # Collect events
    events = defaultdict(list)
    for instrument, instrument_notes in instruments.items():
        for beat, position, pitch, duration, _ in instrument_notes:
            if beat > max_beat:
                continue
            time = beat * resolution + position
            events[instrument].append((time, f"note-on_{pitch}"))
            events[instrument].append((time + duration, f"note-off_{pitch}"))

    # Deduplicate and sort the events
    for instrument in events:
        events[instrument] = sorted(set(events[instrument]))

    # Start the codes with an SOS event
    codes = [indexer["start-of-song"]]

    # Encode the instruments
    for instrument in events:
        codes.append(indexer["start-of-track"])
        codes.append(indexer[f"instrument_{instrument}"])
        time = 0
        for event_time, event in events[instrument]:
            while time < event_time:
                time_shift = min(event_time - time, max_time_shift)
                codes.append(indexer[f"time-shift_{time_shift}"])
                time += time_shift
            codes.append(indexer[event])
        codes.append(indexer["end-of-track"])

    # End the codes with an EOS event
    codes.append(indexer["end-of-song"])

    return np.array(codes)


def encode(music, encoding, indexer):
    """Encode a MusPy music object into a sequence of codes.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Extract notes
    notes = extract_notes(music, encoding["resolution"])

    # Encode the notes
    codes = encode_notes(notes, encoding, indexer)

    return codes


def decode_notes(data, encoding, vocabulary):
    """Decode codes into a note sequence."""
    # Get variables and maps
    resolution = encoding["resolution"]
    instrument_program_map = encoding["instrument_program_map"]

    # Initialize variables
    program = 0
    time = 0
    note_ons = {}

    # Decode the codes into a sequence of notes
    notes = []
    for code in data:
        event = vocabulary[code]
        if event == "start-of-song":
            continue
        elif event == "end-of-song":
            break
        elif event in ("start-of-track", "end-of-track"):
            # Reset variables
            program = 0
            time = 0
            note_ons = {}
        elif event.startswith("instrument"):
            instrument = event.split("_")[1]
            program = instrument_program_map[instrument]
        elif event.startswith("time-shift"):
            time += int(event.split("_")[1])
        elif event.startswith("note-on"):
            pitch = int(event.split("_")[1])
            note_ons[pitch] = time
        elif event.startswith("note-off"):
            pitch = int(event.split("_")[1])
            # Skip a note-off event without a corresponding note-on event
            if pitch not in note_ons:
                continue
            beat, position = divmod(note_ons[pitch], resolution)
            notes.append(
                (beat, position, pitch, time - note_ons[pitch], program)
            )
        else:
            raise ValueError(f"Unknown event type for: {event}")

    return notes


def reconstruct(notes, resolution):
    """Reconstruct a note sequence to a MusPy Music object."""
    # Construct the MusPy Music object
    music = muspy.Music(resolution=resolution, tempos=[muspy.Tempo(0, 100)])

    # Append the tracks
    programs = sorted(set(note[-1] for note in notes))
    for program in programs:
        music.tracks.append(muspy.Track(program))

    # Append the notes
    for beat, position, pitch, duration, program in notes:
        time = beat * resolution + position
        track_idx = programs.index(program)
        music[track_idx].notes.append(muspy.Note(time, pitch, duration))

    return music


def decode(codes, encoding, vocabulary):
    """Decode codes into a MusPy Music object.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Get resolution
    resolution = encoding["resolution"]

    # Decode codes into a note sequence
    notes = decode_notes(codes, encoding, vocabulary)

    # Reconstruct the music object
    music = reconstruct(notes, resolution)

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
            raise ValueError(f"Unknown event type for: {event}")

    return "\n".join(lines)


def save_txt(filename, data, vocabulary):
    """Dump the codes into a TXT file."""
    with open(filename, "w") as f:
        f.write(dump(data, vocabulary))


def save_csv_notes(filename, data):
    """Save the representation as a CSV file."""
    assert data.shape[1] == 5
    np.savetxt(
        filename,
        data,
        fmt="%d",
        delimiter=",",
        header="beat,position,pitch,duration,program",
        comments="",
    )


def save_csv_codes(filename, data):
    """Save the representation as a CSV file."""
    assert data.ndim == 1
    np.savetxt(
        filename,
        data,
        fmt="%d",
        delimiter=",",
        header="code",
        comments="",
    )


def main():
    """Main function."""
    # Get the encoding
    encoding = get_encoding()

    # Save the encoding
    filename = pathlib.Path(__file__).parent / "encoding_mmm.json"
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

    # Print the variables
    print(f"{' Variables ':=^40}")
    print(f"resolution: {encoding['resolution']}")
    print(f"max_beat: {encoding['max_beat']}")
    print(f"max_time_shift: {encoding['max_time_shift']}")

    # Load the example
    music = muspy.load(pathlib.Path(__file__).parent / "example.json")

    # Get the indexer
    indexer = Indexer(is_training=True)

    # Encode the music
    encoded = encode(music, encoding, indexer)
    print(f"Codes:\n{encoded}")

    # Get the learned vocabulary
    vocabulary = utils.inverse_dict(indexer.get_dict())

    print("-" * 40)
    print(f"Decoded:\n{dump(encoded, vocabulary)}")

    music = decode(encoded, encoding, vocabulary)
    print(f"Decoded musics:\n{music}")


if __name__ == "__main__":
    main()
