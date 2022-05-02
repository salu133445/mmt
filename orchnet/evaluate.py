import argparse
import logging
import pathlib
import pprint
import sys
from collections import defaultdict

import muspy
import numpy as np
import torch
import torch.utils.data
import tqdm

import dataset
import music_x_transformers
import representation
import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd"),
        required=True,
        help="dataset key",
    )
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=8,
        type=int,
        help="batch size",
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


def evaluate(data, encoding, filename, eval_dir):
    """Evaluate the results."""
    # Save as a numpy array
    np.save(eval_dir / "npy" / f"{filename}.npy", data)

    # Save as a CSV file
    representation.save_csv_codes(eval_dir / "csv" / f"{filename}.csv", data)

    # Convert to a MusPy Music object
    music = representation.decode(data, encoding)

    # Save as a MusPy JSON file
    music.save(eval_dir / "json" / f"{filename}.json")

    return {
        "pitch_class_entropy": muspy.pitch_class_entropy(music),
        "scale_consistency": muspy.scale_consistency(music),
        "groove_consistency": muspy.groove_consistency(
            music, 4 * music.resolution
        ),
    }


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

    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "evaluate.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(f"Saved arguments to {args.out_dir / 'evaluate-args.json'}")
    utils.save_args(args.out_dir / "evaluate-args.json", args)

    # Load training configurations
    logging.info(
        f"Loading training arguments from: {args.out_dir / 'train-args.json'}"
    )
    train_args = utils.load_json(args.out_dir / "train-args.json")
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")

    # Make sure the output directory exists
    eval_dir = args.out_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    for key in (
        "truth",
        "unconditioned",
        "instrument-informed",
        "4-beat-continuation",
        "16-beat-continuation",
    ):
        (eval_dir / key).mkdir(exist_ok=True)
        (eval_dir / key / "npy").mkdir(exist_ok=True)
        (eval_dir / key / "csv").mkdir(exist_ok=True)
        (eval_dir / key / "json").mkdir(exist_ok=True)

    # Get the specified device
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Load the encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Create the dataset and data loader
    logging.info(f"Creating the data loader...")
    test_dataset = dataset.MusicDataset(
        args.names,
        args.in_dir,
        encoding,
        max_seq_len=train_args["max_seq_len"],
        use_csv=args.use_csv,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
    )

    # Create the model
    logging.info(f"Creating the model...")
    disable_absolute_positional_embedding = train_args.get(
        "disable_absolute_positional_embedding"
    )
    if disable_absolute_positional_embedding is None:
        disable_absolute_positional_embedding = False
    model = music_x_transformers.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
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

    # Get special tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]
    beat_0 = encoding["beat_code_map"][0]
    beat_4 = encoding["beat_code_map"][4]
    beat_16 = encoding["beat_code_map"][16]

    results = defaultdict(list)

    # Iterate over the dataset
    with torch.no_grad():
        for idx, batch in enumerate(tqdm.tqdm(test_loader, ncols=80)):
            # ------------
            # Ground truth
            # ------------

            truth_np = batch["seq"].numpy()
            result = evaluate(
                truth_np[0], encoding, f"{idx}_0", eval_dir / "truth"
            )
            results["truth"].append(result)

            # ------------------------
            # Unconditioned generation
            # ------------------------

            # Get output start tokens
            tgt_start = torch.zeros(
                (args.batch_size, 1, 6), dtype=torch.long, device=device
            )
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
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # Evaluate the results
            result = evaluate(
                generated_np[0],
                encoding,
                f"{idx}_0",
                eval_dir / "unconditioned",
            )
            results["unconditioned"].append(result)

            # # ------------------------------
            # # Instrument-informed generation
            # # ------------------------------

            # # Get output start tokens
            # prefix_len = int(np.argmax(batch["seq"][0, :, 1] >= beat_0))
            # tgt_start = batch["seq"][:1, :prefix_len].to(device)

            # # Generate new samples
            # generated = model.generate(
            #     tgt_start,
            #     args.seq_len,
            #     eos_token=eos,
            #     temperature=temperature,
            #     filter_logits_fn=logits_filter,
            #     filter_thres=filter_thres,
            #     monotonicity_dim=("type", "beat"),
            # )
            # generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # # Evaluate the results
            # result = evaluate(
            #     generated_np[0],
            #     encoding,
            #     f"{idx}_0",
            #     eval_dir / "instrument-informed",
            # )
            # results["instrument-informed"].append(result)

            # # -------------------
            # # 4-beat continuation
            # # -------------------

            # # Get output start tokens
            # cond_len = int(np.argmax(batch["seq"][0, :, 1] >= beat_4))
            # tgt_start = batch["seq"][:1, :cond_len].to(device)

            # # Generate new samples
            # generated = model.generate(
            #     tgt_start,
            #     args.seq_len,
            #     eos_token=eos,
            #     temperature=temperature,
            #     filter_logits_fn=logits_filter,
            #     filter_thres=filter_thres,
            #     monotonicity_dim=("type", "beat"),
            # )
            # generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # # Evaluate the results
            # result = evaluate(
            #     generated_np[0],
            #     encoding,
            #     f"{idx}_0",
            #     eval_dir / "4-beat-continuation",
            # )
            # results["4-beat-continuation"].append(result)

            # # --------------------
            # # 16-beat continuation
            # # --------------------

            # # Get output start tokens
            # cond_len = int(np.argmax(batch["seq"][0, :, 1] >= beat_16))
            # tgt_start = batch["seq"][:1, :cond_len].to(device)

            # # Generate new samples
            # generated = model.generate(
            #     tgt_start,
            #     args.seq_len,
            #     eos_token=eos,
            #     temperature=temperature,
            #     filter_logits_fn=logits_filter,
            #     filter_thres=filter_thres,
            #     monotonicity_dim=("type", "beat"),
            # )
            # generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # # Evaluate the results
            # result = evaluate(
            #     generated_np[0],
            #     encoding,
            #     f"{idx}_0",
            #     eval_dir / "16-beat-continuation",
            # )
            # results["16-beat-continuation"].append(result)

    for exp, result in results.items():
        logging.info(exp)
        for key in result[0]:
            logging.info(
                f"{key}: mean={np.nanmean([r[key] for r in result]):.4f}, "
                f"steddev={np.nanstd([r[key]for r in result]):.4f}"
            )


if __name__ == "__main__":
    main()
