"""Extract note sequences from music."""
import argparse
import logging
import pathlib
import pprint
import sys

import joblib
import muspy
import numpy as np
import tqdm

import representation
import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract note sequences from music"
    )
    parser.add_argument(
        "-d", "--dataset", choices=("sod", "lmd"), help="dataset key"
    )
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-e",
        "--ignore_exceptions",
        action="store_true",
        help="whether to ignore all exceptions",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="number of jobs",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def extract(name, in_dir, out_dir, resolution):
    """Encode a note sequence into the beat representation."""
    # Load the score
    music = muspy.load(in_dir / f"{name}.json")

    # Encode the score
    notes = representation.extract_notes(music, resolution)

    # Filter out bad files
    if len(notes) < 10:
        return

    # Save the notes as CSV files
    out_filename = out_dir / f"{name}.csv"
    out_filename.parent.mkdir(exist_ok=True)
    representation.save_csv_notes(out_filename, notes)

    # Save the notes as NPY files
    out_filename = out_dir / f"{name}.npy"
    out_filename.parent.mkdir(exist_ok=True)
    np.save(out_filename, notes)

    return name


@utils.ignore_exceptions
def extract_ignore_exceptions(name, in_dir, out_dir, resolution):
    """Encode a note sequence into machine-learning friendly codes,
    ignoring all exceptions."""
    return extract(name, in_dir, out_dir, resolution)


def process(name, in_dir, out_dir, resolution, ignore_exceptions=True):
    """Wrapper for multiprocessing."""
    if ignore_exceptions:
        return extract_ignore_exceptions(name, in_dir, out_dir, resolution)
    return extract(name, in_dir, out_dir, resolution)


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = args.names or pathlib.Path(
                f"data/{args.dataset}/processed/json-names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/json/")
        if args.out_dir is None:
            args.out_dir = pathlib.Path(
                f"data/{args.dataset}/processed/notes/"
            )

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

    # Get the encoding
    encoding = representation.get_encoding()

    # Save the encoding
    encoding_filename = args.out_dir / "encoding.json"
    utils.save_json(encoding_filename, encoding)
    logging.info(f"Saved the encoding to: {encoding_filename}")

    # Get filenames
    logging.info("Loading names...")
    names = utils.load_txt(args.names)

    # Iterate over names
    logging.info("Iterating over names...")
    extracted_names = []
    if args.jobs == 1:
        for name in (pbar := tqdm.tqdm(names)):
            pbar.set_postfix_str(name)
            result = process(
                name,
                args.in_dir,
                args.out_dir,
                encoding["resolution"],
                args.ignore_exceptions,
            )
            if result is not None:
                extracted_names.append(result)
    else:
        results = joblib.Parallel(n_jobs=args.jobs, verbose=5)(
            joblib.delayed(process)(
                name,
                args.in_dir,
                args.out_dir,
                encoding["resolution"],
                args.ignore_exceptions,
            )
            for name in names
        )
        extracted_names = [result for result in results if result is not None]
    logging.info(
        f"Extracted {len(extracted_names)} out of {len(names)} files."
    )

    # Save successfully encoded names
    out_filename = args.out_dir.parent / "names.txt"
    utils.save_txt(out_filename, extracted_names)
    logging.info(f"Saved the extracted filenames to: {out_filename}")


if __name__ == "__main__":
    main()
