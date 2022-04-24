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
    sample_dir = args.out_dir / "visualizations"
    sample_dir.mkdir(exist_ok=True)

    # Get the specified device
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Load training configurations
    train_args = utils.load_json(args.out_dir / "train-args.json")

    # Load the encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Create the model
    logging.info(f"Creating the model...")
    model = music_x_transformer.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rel_pos_bias=not train_args["disable_relative_positional_embedding"],
        rotary_pos_emb=not train_args["disable_relative_positional_embedding"],
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

    # Iterate over the dataset
    with torch.no_grad():
        for i in tqdm.tqdm(range(args.n_samples), ncols=80):

            # Get output start tokens
            tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            tgt_start[:, 0, 0] = sos

            # Sequence length
            n = 50
            n2 = 50

            # Generate new samples
            generated, attns = model.generate(
                tgt_start,
                max(n, n2),
                eos_token=eos,
                temperature=temperature,
                filter_logits_fn=logits_filter,
                filter_thres=filter_thres,
                monotonicity_dim=("type", "beat"),
                return_attn=True,
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            m = attns[0].shape[1]

            attn_maps = np.zeros((m, n, n))
            for j in range(n):
                attn_maps[:, j, : (j + 1)] = attns[j]

            for h in range(m):
                plt.figure(figsize=(8, 8))
                plt.imshow(attn_maps[h].T, cmap="Blues", origin="upper")
                codes = [
                    f"({c[0]},{c[1]},{c[2]:2},{c[3]:2},{c[4]:2},{c[5]:2})"
                    for c in generated_np[0, :n]
                ]
                plt.xticks(
                    np.arange(n), codes, family="monospace", rotation=90
                )
                plt.yticks(np.arange(n), codes, family="monospace")
                plt.gca().xaxis.tick_top()
                plt.tight_layout()
                plt.savefig(sample_dir / f"{i}_head_{h}.png")
                plt.close()

            attn_maps = np.zeros((m, n2, n2))
            for j in range(n2):
                attn_maps[:, j, : (j + 1)] = attns[j]

            mean_attn_maps = np.mean(attn_maps, 0)
            plt.figure(figsize=(8, 8))
            plt.imshow(mean_attn_maps.T, cmap="Blues", origin="upper")
            codes = [
                f"({c[0]},{c[1]},{c[2]:2},{c[3]:2},{c[4]:2},{c[5]:2})"
                for c in generated_np[0, :n2]
            ]
            plt.xticks(np.arange(n2), codes, family="monospace", rotation=90)
            plt.yticks(np.arange(n2), codes, family="monospace")
            plt.gca().xaxis.tick_top()
            plt.tight_layout()
            plt.savefig(sample_dir / f"{i}_mean.png")
            plt.close()

            max_attn_maps = np.max(attn_maps, 0)
            plt.figure(figsize=(8, 8))
            plt.imshow(max_attn_maps.T, cmap="Blues", origin="upper")
            codes = [
                f"({c[0]},{c[1]},{c[2]:2},{c[3]:2},{c[4]:2},{c[5]:2})"
                for c in generated_np[0, :n2]
            ]
            plt.xticks(np.arange(n2), codes, family="monospace", rotation=90)
            plt.yticks(np.arange(n2), codes, family="monospace")
            plt.gca().xaxis.tick_top()
            plt.tight_layout()
            plt.savefig(sample_dir / f"{i}_max.png")
            plt.close()


if __name__ == "__main__":
    main()
