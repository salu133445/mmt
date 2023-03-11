"""Convert MIDI and MusicXML files into music JSON files."""
import argparse
import logging
import pathlib
import pprint
import sys

import joblib
import muspy
import tqdm

import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert MIDI and MusicXML files into music JSON files."
    )
    parser.add_argument(
        "-n",
        "--names",
        default="data/snd/original-names.txt",
        type=pathlib.Path,
        help="input names",
    )
    parser.add_argument(
        "-i",
        "--in_dir",
        default="data/snd/SymphonyNet_Dataset/",
        type=pathlib.Path,
        help="input data directory",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="data/snd/processed/json/",
        type=pathlib.Path,
        help="output directory",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        default=12,
        type=int,
        help="number of time steps per quarter note",
    )
    parser.add_argument(
        "-s",
        "--skip_existing",
        action="store_true",
        help="whether to skip existing outputs",
    )
    parser.add_argument(
        "-e",
        "--ignore_exceptions",
        action="store_true",
        help="whether to ignore all exceptions",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, help="number of jobs"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def adjust_resolution(music, resolution):
    """Adjust the resolution of the music."""
    music.adjust_resolution(resolution)
    for track in music:
        for note in track:
            if note.duration == 0:
                note.duration = 1
    music.remove_duplicate()


def convert(name, in_dir, out_dir, resolution, skip_existing):
    """Convert MIDI and MusicXML files into MusPy JSON files."""
    # Get output filename
    collection, filename = name.split("/")
    idx = filename.split(".")[0]
    out_name = f"{collection}/{collection}-{idx}"
    out_filename = out_dir / f"{out_name}.json"

    # Skip if the output file exists
    if skip_existing and out_filename.is_file():
        return

    # Read the MIDI file
    music = muspy.read(in_dir / name)

    # Adjust the resolution
    adjust_resolution(music, resolution)

    # Filter bad files
    end_time = music.get_end_time()
    if end_time > resolution * 4 * 2000 or end_time < resolution * 4 * 10:
        return

    # Save as a MusPy JSON file
    out_filename.parent.mkdir(exist_ok=True, parents=True)
    music.save(out_filename)

    return out_name


@utils.ignore_exceptions
def convert_ignore_expections(
    name, in_dir, out_dir, resolution, skip_existing
):
    """Convert MIDI files into music JSON files, ignoring all expections."""
    return convert(name, in_dir, out_dir, resolution, skip_existing)


def process(
    name, in_dir, out_dir, resolution, skip_existing, ignore_exceptions=True
):
    """Wrapper for multiprocessing."""
    if ignore_exceptions:
        return convert_ignore_expections(
            name, in_dir, out_dir, resolution, skip_existing
        )
    return convert(name, in_dir, out_dir, resolution, skip_existing)


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Make sure output directory exists
    args.out_dir.mkdir(exist_ok=True)

    # Set up the logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Get names
    logging.info("Loading names...")
    with open(args.names, encoding="utf8") as f:
        names = [line.strip() for line in f]

    # Iterate over names
    logging.info("Iterating over names...")
    if args.jobs == 1:
        converted_names = []
        for name in (pbar := tqdm.tqdm(names)):
            pbar.set_postfix_str(name)
            result = process(
                name,
                args.in_dir,
                args.out_dir,
                args.resolution,
                args.skip_existing,
                args.ignore_exceptions,
            )
            if result is not None:
                converted_names.append(result)
    else:
        results = joblib.Parallel(
            n_jobs=args.jobs, verbose=0 if args.quiet else 5
        )(
            joblib.delayed(process)(
                name,
                args.in_dir,
                args.out_dir,
                args.resolution,
                args.skip_existing,
                args.ignore_exceptions,
            )
            for name in names
        )
        converted_names = [result for result in results if result is not None]
    converted_names = sorted(set(converted_names))
    logging.info(
        f"Converted {len(converted_names)} out of {len(names)} files."
    )

    # Save successfully converted names
    out_filename = args.out_dir.parent / "json-names.txt"
    utils.save_txt(out_filename, converted_names)
    logging.info(f"Saved the converted filenames to: {out_filename}")


if __name__ == "__main__":
    main()
