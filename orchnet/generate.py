import argparse
import logging
import pathlib
import pprint
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import tqdm

import dataset
import music_x_transformer
import representation
import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
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
        "-ns",
        "--n_samples",
        default=10,
        type=int,
        help="number of samples to generate",
    )
    parser.add_argument(
        "-c",
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        help="step of the trained model to load (default to the best model)",
    )
    parser.add_argument(
        "-sl",
        "--seq_len",
        default=1024,
        type=int,
        help="sequence length to generate",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        default=1.0,
        type=float,
        help="sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "-ts",
        "--temperatures",
        nargs="+",
        type=float,
        help="sampling temperatures",
    )
    parser.add_argument(
        "-f",
        "--filter",
        default="top_k",
        type=str,
        help="sampling filter (default: 'top_k')",
    )
    parser.add_argument(
        "-fs",
        "--filters",
        nargs="+",
        type=str,
        help="sampling filters",
    )
    parser.add_argument(
        "-ft",
        "--filter_threshold",
        default=0.9,
        type=float,
        help="sampling filter threshold (default: 0.9)",
    )
    parser.add_argument(
        "-fts",
        "--filter_thresholds",
        nargs="+",
        type=float,
        help="sampling filter thresholds",
    )
    parser.add_argument("-g", "--gpu", type=int, help="gpu number")
    parser.add_argument(
        "-j", "--jobs", default=0, type=int, help="number of jobs"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def save_pianoroll(filename, music, size=None, **kwargs):
    """Save the piano roll to file."""
    music.show_pianoroll(track_label="program")
    if size is not None:
        plt.gcf().set_size_inches(size)
    plt.savefig(filename)
    plt.close()


def save_result(filename, data, sample_dir, encoding):
    """Save the results in multiple formats."""
    # Save as a numpy array
    np.save(sample_dir / "npy" / f"{filename}.npy", data)

    # Save as a CSV file
    representation.save_csv(sample_dir / "csv" / f"{filename}.csv", data)

    # Convert to a MusPy Music object
    music_truth = representation.decode(data, encoding)

    # Save as a piano roll
    save_pianoroll(
        sample_dir / "png" / f"{filename}.png",
        music_truth,
        (20, 5),
        preset="frame",
    )

    # Save as a MIDI file
    music_truth.write(sample_dir / "mid" / f"{filename}.mid")

    # Save as a WAV file
    music_truth.write(
        sample_dir / "wav" / f"{filename}.wav",
        options="-o synth.polyphony=4096",
    )

    # Save also as a MP3 file
    subprocess.check_output(
        ["ffmpeg", "-loglevel", "error", "-y", "-i"]
        + [str(sample_dir / "wav" / f"{filename}.wav")]
        + ["-b:a", "192k"]
        + [str(sample_dir / "mp3" / f"{filename}.mp3")]
    )

    # Trim the music
    music_truth.trim(music_truth.resolution * 64)

    # Save the trimmed version as a piano roll
    save_pianoroll(
        sample_dir / "png-trimmed" / f"{filename}.png", music_truth, (10, 5)
    )

    # Save the trimmed version as a WAV file
    music_truth.write(
        sample_dir / "wav-trimmed" / f"{filename}.wav",
        options="-o synth.polyphony=4096",
    )

    # Save also as a MP3 file
    subprocess.check_output(
        ["ffmpeg", "-loglevel", "error", "-y", "-i"]
        + [str(sample_dir / "wav-trimmed" / f"{filename}.wav")]
        + ["-b:a", "192k"]
        + [str(sample_dir / "mp3-trimmed" / f"{filename}.mp3")]
    )


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = pathlib.Path(
                f"data/{args.dataset}/processed/test-names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes/")
        if args.out_dir is None:
            args.out_dir = pathlib.Path(f"exp/test_{args.dataset}")
    temperature = args.temperatures or args.temperature
    logits_filter = args.filters or args.filter
    filter_thres = args.filter_thresholds or args.filter_threshold
    if args.temperatures:
        args.temperature = None
    if args.filters:
        args.filter = None
    if args.filter_thresholds:
        args.filter_threshold = None

    # Save command-line arguments
    utils.save_args(args.out_dir / "generate-args.json", args)

    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "generate.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Make sure the sample directory exists
    sample_dir = args.out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    (sample_dir / "npy").mkdir(exist_ok=True)
    (sample_dir / "csv").mkdir(exist_ok=True)
    (sample_dir / "png").mkdir(exist_ok=True)
    (sample_dir / "png-trimmed").mkdir(exist_ok=True)
    (sample_dir / "mid").mkdir(exist_ok=True)
    (sample_dir / "wav").mkdir(exist_ok=True)
    (sample_dir / "wav-trimmed").mkdir(exist_ok=True)
    (sample_dir / "mp3").mkdir(exist_ok=True)
    (sample_dir / "mp3-trimmed").mkdir(exist_ok=True)

    # Get the specified device
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Load training configurations
    train_args = utils.load_json(args.out_dir / "train-args.json")

    # Load the encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Create the dataset and data loader
    logging.info(f"Creating the data loader...")
    test_dataset = dataset.SODDataset(
        args.names,
        args.in_dir,
        encoding,
        max_seq_len=train_args["max_seq_len"],
        use_csv=args.use_csv,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        num_workers=args.jobs,
        collate_fn=dataset.SODDataset.collate,
    )

    # Create the model
    logging.info(f"Creating the model...")
    model = music_x_transformer.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rel_pos_bias=not train_args["use_absolute_positional_embedding"],
        rotary_pos_emb=not train_args["use_absolute_positional_embedding"],
        emb_dropout=train_args["dropout"],
        attn_dropout=train_args["dropout"],
        ff_dropout=train_args["dropout"],
    ).to(device)

    # Load the checkpoint
    checkpoint_dir = args.out_dir / "checkpoints"
    if args.step is None:
        checkpoint_filename = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_filename = checkpoint_dir / f"model_{args.step}.pt"
    model.load_state_dict(torch.load(checkpoint_filename, map_location=device))
    logging.info(f"Loaded the model weights from: {checkpoint_filename}")
    model.eval()

    # Get EOS token
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]
    beat_0 = encoding["beat_code_map"][0]
    beat_4 = encoding["beat_code_map"][4]
    beat_16 = encoding["beat_code_map"][16]

    # Iterate over the dataset
    with torch.no_grad():
        data_iter = iter(test_loader)
        for i in tqdm.tqdm(range(args.n_samples), ncols=80):
            batch = next(data_iter)

            # ------------------------
            # Unconditioned generation
            # ------------------------

            # Get output start tokens
            tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            tgt_start[:, 0, 0] = sos

            # Generate new samples
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token=eos,
                temperature=temperature,
                filter_logits_fn=logits_filter,
                filter_thres=filter_thres,
                monotonicity_dim=("type", "beat"),
            )
            generated = torch.cat((tgt_start, generated), 1)

            # Convert to numpy arrays
            truth_np = batch["seq"][0].numpy()
            generated_np = generated[0].cpu().numpy()

            # Save the results
            save_result(f"{i}_truth", truth_np, sample_dir, encoding)
            save_result(
                f"{i}_unconditioned", generated_np, sample_dir, encoding
            )

            # ------------------------------
            # Instrument-informed generation
            # ------------------------------

            # Get output start tokens
            prefix_len = int(np.argmax(batch["seq"][0, :, 1] >= beat_0))
            tgt_start = batch["seq"][:1, :prefix_len].to(device)

            # Generate new samples
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token=eos,
                temperature=temperature,
                filter_logits_fn=logits_filter,
                filter_thres=filter_thres,
                monotonicity_dim=("type", "beat"),
            )
            generated = torch.cat((tgt_start, generated), 1)

            # Convert to numpy arrays
            truth_np = batch["seq"][0].numpy()
            generated_np = generated[0].cpu().numpy()

            # Save the results
            save_result(
                f"{i}_instrument-informed", generated_np, sample_dir, encoding
            )

            # -------------------
            # 4-beat continuation
            # -------------------

            # Get output start tokens
            cond_len = int(np.argmax(batch["seq"][0, :, 1] >= beat_4))
            tgt_start = batch["seq"][:1, :cond_len].to(device)

            # Generate new samples
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token=eos,
                temperature=temperature,
                filter_logits_fn=logits_filter,
                filter_thres=filter_thres,
                monotonicity_dim=("type", "beat"),
            )
            generated = torch.cat((tgt_start, generated), 1)

            # Convert to numpy arrays
            truth_np = batch["seq"][0].numpy()
            generated_np = generated[0].cpu().numpy()

            # Save the results
            save_result(
                f"{i}_4-beat-continuation", generated_np, sample_dir, encoding
            )

            # --------------------
            # 16-beat continuation
            # --------------------

            # Get output start tokens
            cond_len = int(np.argmax(batch["seq"][0, :, 1] >= beat_16))
            tgt_start = batch["seq"][:1, :cond_len].to(device)

            # Generate new samples
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token=eos,
                temperature=temperature,
                filter_logits_fn=logits_filter,
                filter_thres=filter_thres,
                monotonicity_dim=("type", "beat"),
            )
            generated = torch.cat((tgt_start, generated), 1)

            # Convert to numpy arrays
            truth_np = batch["seq"][0].numpy()
            generated_np = generated[0].cpu().numpy()

            # Save results
            save_result(
                f"{i}_16-beat-continuation", generated_np, sample_dir, encoding
            )


if __name__ == "__main__":
    main()
