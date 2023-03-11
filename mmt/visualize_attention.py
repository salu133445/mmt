import argparse
import logging
import pathlib
import pprint
import sys

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import tqdm

import dataset
import music_x_transformers
import representation
import utils

plt.rc("font", family="serif")
plt.rc("axes", linewidth=1.5)
plt.rc("savefig", dpi="150")


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "snd"),
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
        "-ns",
        "--n_samples",
        default=100,
        type=int,
        help="number of samples to generate",
    )
    # Data
    parser.add_argument(
        "-s",
        "--shuffle",
        action="store_true",
        help="whether to shuffle the test data",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    # Model
    parser.add_argument(
        "--model_steps",
        type=int,
        help="step of the trained model to load (default to the best model)",
    )
    parser.add_argument(
        "--seq_len", default=1024, type=int, help="sequence length to generate"
    )
    parser.add_argument(
        "--temperature",
        nargs="+",
        default=1.0,
        type=float,
        help="sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default="top_k",
        type=str,
        help="sampling filter (default: 'top_k')",
    )
    parser.add_argument(
        "--filter_threshold",
        nargs="+",
        default=0.9,
        type=float,
        help="sampling filter threshold (default: 0.9)",
    )
    # Others
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

    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "visualize-attention.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(
        f"Saved arguments to {args.out_dir / 'visualize-attention-args.json'}"
    )
    utils.save_args(args.out_dir / "visualize-attention-args.json", args)

    # Load training configurations
    logging.info(
        f"Loading training arguments from: {args.out_dir / 'train-args.json'}"
    )
    train_args = utils.load_json(args.out_dir / "train-args.json")
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")

    # Make sure the sample directory exists
    sample_dir = args.out_dir / "visualizations"
    sample_dir.mkdir(exist_ok=True)

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
        max_beat=train_args["max_beat"],
        use_csv=args.use_csv,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=args.shuffle,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
    )

    # Create the model
    logging.info(f"Creating the model...")
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
    if args.model_steps is None:
        checkpoint_filename = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_filename = checkpoint_dir / f"model_{args.model_steps}.pt"
    model.load_state_dict(torch.load(checkpoint_filename, map_location=device))
    logging.info(f"Loaded the model weights from: {checkpoint_filename}")
    model.eval()

    # Get EOS token
    eos = encoding["type_code_map"]["end-of-song"]

    # Initialize vairables
    n_dim = len(encoding["dimensions"])
    n_heads = train_args["heads"]
    event_attn = [np.zeros((n_heads, n, n)) for n in encoding["n_tokens"]]
    rel_event_attn = [
        np.zeros((n_heads, 2 * n)) for n in encoding["n_tokens"]
    ]
    rel_random_attn = [
        np.zeros((1, 2 * n)) for n in encoding["n_tokens"]
    ]

    # Iterate over the dataset
    with torch.no_grad():
        test_iter = iter(test_loader)
        for _ in tqdm.tqdm(range(args.n_samples), ncols=80):

            batch = next(test_iter)

            tgt_start = batch["seq"][:, : args.seq_len - 1].to(device)

            # Generate new samples
            generated, attns = model.generate(
                tgt_start,
                1,
                eos_token=eos,
                temperature=args.temperature,
                filter_logits_fn=args.filter,
                filter_thres=args.filter_threshold,
                monotonicity_dim=("type", "beat"),
                return_attn=True,
            )

            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()
            attn = attns[-1][0].cpu().numpy()

            seq_len = generated_np.shape[1]

            # Absolute event-event attention
            for h in range(n_heads):
                for d in range(n_dim):
                    for k in range(1, seq_len - 1):
                        for k_ in range(seq_len - 2):
                            if d > 0 and (
                                generated_np[0, k_, d] == 0
                                or generated_np[0, k, d] == 0
                            ):
                                continue
                            event_attn[d][
                                h,
                                generated_np[0, k, d],
                                generated_np[0, k_, d],
                            ] += attn[h, k, k_]

            # Relative event-event attention
            for h in range(n_heads):
                for d, n in enumerate(encoding["n_tokens"]):
                    for k in range(1, seq_len - 1):
                        for k_ in range(seq_len - 2):
                            if (
                                generated_np[0, k_, d] == 0
                                or generated_np[0, k, d] == 0
                            ):
                                continue
                            rel_event_attn[d][
                                h,
                                generated_np[0, k_, d]
                                - generated_np[0, k, d]
                                + n,
                            ] += attn[h, k, k_]
                            if h == 0:
                                rel_random_attn[d][
                                    0,
                                    generated_np[0, k_, d]
                                    - generated_np[0, k, d]
                                    + n,
                                ] += 1

        ticklabels = [
            [
                encoding["code_type_map"].get(i)
                for i in range(encoding["n_tokens"][0])
            ],
            [
                encoding["code_beat_map"].get(i)
                for i in range(encoding["n_tokens"][1])
            ],
            [
                encoding["code_position_map"].get(i)
                for i in range(encoding["n_tokens"][2])
            ],
            [
                encoding["code_pitch_map"].get(i)
                for i in range(encoding["n_tokens"][3])
            ],
            [
                encoding["code_duration_map"].get(i)
                for i in range(encoding["n_tokens"][4])
            ],
            [
                encoding["code_instrument_map"].get(i)
                for i in range(encoding["n_tokens"][5])
            ],
        ]
        with np.errstate(divide="ignore", invalid="ignore"):
            for h in range(n_heads):
                for d, key in enumerate(encoding["dimensions"]):
                    if key not in ("type", "duration", "instrument"):
                        continue
                    if key == "type":
                        plt.figure(figsize=(2.5, 2.5))
                    elif key == "instrument":
                        plt.figure(figsize=(8, 8))
                    elif key == "duration":
                        plt.figure(figsize=(6, 6))
                    plt.imshow(
                        np.nan_to_num(
                            event_attn[d][h]
                            / event_attn[d][h].sum(-1, keepdims=True)
                        ).T,
                        cmap="Blues",
                        interpolation="none",
                        origin="upper",
                    )
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d]),
                        ticklabels[d],
                        rotation="vertical",
                    )
                    plt.yticks(
                        np.arange(encoding["n_tokens"][d]), ticklabels[d]
                    )
                    plt.gca().xaxis.tick_top()
                    plt.tick_params(
                        bottom=False,
                        top=True,
                        left=True,
                        right=False,
                        labelbottom=False,
                        labeltop=True,
                        labelleft=True,
                        labelright=False,
                    )
                    if key != "type":
                        plt.xlim(0.5, encoding["n_tokens"][d] - 0.5)
                        plt.ylim(encoding["n_tokens"][d] - 0.5, 0.5)
                    plt.xlabel(key.capitalize())
                    plt.ylabel(key.capitalize())
                    plt.tight_layout()
                    plt.savefig(
                        sample_dir / f"{key}_head-{h}.png", bbox_inches="tight"
                    )
                    plt.savefig(
                        sample_dir / f"{key}_head-{h}.pdf", bbox_inches="tight"
                    )
                    plt.close()

            for d, key in enumerate(encoding["dimensions"]):
                if key not in ("type", "duration", "instrument"):
                    continue
                if key == "type":
                    plt.figure(figsize=(2.5, 2.5))
                elif key == "instrument":
                    plt.figure(figsize=(8, 8))
                elif key == "duration":
                    plt.figure(figsize=(6, 6))
                plt.imshow(
                    np.nan_to_num(
                        event_attn[d] / event_attn[d].sum(-1, keepdims=True)
                    )
                    .mean(0)
                    .T,
                    cmap="Blues",
                    interpolation="none",
                    origin="upper",
                )
                plt.xticks(
                    np.arange(encoding["n_tokens"][d]),
                    ticklabels[d],
                    rotation="vertical",
                )
                plt.yticks(np.arange(encoding["n_tokens"][d]), ticklabels[d])
                plt.tick_params(
                    bottom=False,
                    top=True,
                    left=True,
                    right=False,
                    labelbottom=False,
                    labeltop=True,
                    labelleft=True,
                    labelright=False,
                )
                if key != "type":
                    plt.xlim(0.5, encoding["n_tokens"][d] - 0.5)
                    plt.ylim(encoding["n_tokens"][d] - 0.5, 0.5)
                plt.xlabel(key.capitalize())
                plt.ylabel(key.capitalize())
                plt.tight_layout()
                plt.savefig(
                    sample_dir / f"{key}_mean.png",
                    bbox_inches="tight",
                )
                plt.savefig(
                    sample_dir / f"{key}_mean.pdf", bbox_inches="tight"
                )
                plt.close()

            for d, key in enumerate(encoding["dimensions"]):
                if key not in ("beat", "position", "pitch"):
                    continue
                plt.figure(figsize=(6, 2))
                plt.imshow(
                    np.nan_to_num(
                        rel_event_attn[d]
                        / rel_event_attn[d].sum(-1, keepdims=True)
                    ),
                    cmap="Blues",
                    aspect="auto",
                    interpolation="none",
                )
                if key == "beat":
                    s = encoding["n_tokens"][d] % 4
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2)[s::4],
                        np.arange(
                            -encoding["n_tokens"][d],
                            encoding["n_tokens"][d],
                        )[s::4],
                    )
                    plt.xlim(
                        encoding["n_tokens"][d] - 40.5,
                        encoding["n_tokens"][d] + 4.5,
                    )
                elif key == "position":
                    s = encoding["n_tokens"][d] % 4
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2)[s::4],
                        np.arange(
                            -encoding["n_tokens"][d],
                            encoding["n_tokens"][d],
                        )[s::4],
                    )
                    plt.xlim(
                        encoding["n_tokens"][d] - 12,
                        encoding["n_tokens"][d] + 12,
                    )
                elif key == "pitch":
                    s = encoding["n_tokens"][d] % 5
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2)[s::5],
                        np.arange(
                            -encoding["n_tokens"][d],
                            encoding["n_tokens"][d],
                        )[s::5],
                    )
                    plt.xlim(
                        encoding["n_tokens"][d] - 25.5,
                        encoding["n_tokens"][d] + 25.5,
                    )
                else:
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2),
                        np.arange(
                            -encoding["n_tokens"][d], encoding["n_tokens"][d]
                        ),
                    )
                plt.ylabel("Attention\nhead")
                plt.yticks(np.arange(n_heads), np.arange(n_heads) + 1)
                plt.xlabel(f"{key.capitalize()} difference")
                plt.tight_layout()
                plt.savefig(sample_dir / f"{key}_rel.png", bbox_inches="tight")
                plt.savefig(sample_dir / f"{key}_rel.pdf", bbox_inches="tight")
                plt.close()

            for d, key in enumerate(encoding["dimensions"]):
                if key not in ("beat", "position", "pitch"):
                    continue
                plt.figure(figsize=(6, 2))
                plt.imshow(
                    np.concatenate(
                        (
                            np.nan_to_num(
                                rel_event_attn[d]
                                / rel_event_attn[d].sum(-1, keepdims=True)
                            ),
                            np.nan_to_num(
                                rel_random_attn[d]
                                / rel_random_attn[d].sum(-1, keepdims=True)
                            )
                        )
                    ),
                    cmap="Blues",
                    aspect="auto",
                    interpolation="none",
                )
                if key == "beat":
                    s = encoding["n_tokens"][d] % 4
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2)[s::4],
                        np.arange(
                            -encoding["n_tokens"][d],
                            encoding["n_tokens"][d],
                        )[s::4],
                    )
                    plt.xlim(
                        encoding["n_tokens"][d] - 40.5,
                        encoding["n_tokens"][d] + 4.5,
                    )
                elif key == "position":
                    s = encoding["n_tokens"][d] % 4
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2)[s::4],
                        np.arange(
                            -encoding["n_tokens"][d],
                            encoding["n_tokens"][d],
                        )[s::4],
                    )
                    plt.xlim(
                        encoding["n_tokens"][d] - 12,
                        encoding["n_tokens"][d] + 12,
                    )
                elif key == "pitch":
                    s = encoding["n_tokens"][d] % 5
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2)[s::5],
                        np.arange(
                            -encoding["n_tokens"][d],
                            encoding["n_tokens"][d],
                        )[s::5],
                    )
                    plt.xlim(
                        encoding["n_tokens"][d] - 25.5,
                        encoding["n_tokens"][d] + 25.5,
                    )
                else:
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2),
                        np.arange(
                            -encoding["n_tokens"][d], encoding["n_tokens"][d]
                        ),
                    )
                plt.ylabel("Attention\nhead")
                plt.yticks(
                    np.arange(n_heads + 1),
                    [str(i) for i in range(1, n_heads + 1)] + ["*"]
                )
                plt.xlabel(f"{key.capitalize()} difference")
                plt.tight_layout()
                plt.savefig(
                    sample_dir / f"{key}_rel2.png",
                    bbox_inches="tight"
                )
                plt.savefig(
                    sample_dir / f"{key}_rel2.pdf",
                    bbox_inches="tight"
                )
                plt.close()


            for d, key in enumerate(encoding["dimensions"]):
                if key not in ("beat", "position", "pitch"):
                    continue
                plt.figure(figsize=(6, 2))
                plt.imshow(
                    np.nan_to_num(
                        rel_event_attn[d]
                        / rel_event_attn[d].sum(-1, keepdims=True)
                    )
                    - np.nan_to_num(
                        rel_random_attn[d]
                        / rel_random_attn[d].sum(-1, keepdims=True)
                    ),
                    norm=matplotlib.colors.CenteredNorm(),
                    cmap="bwr",
                    aspect="auto",
                    interpolation="none",
                )
                if key == "beat":
                    s = encoding["n_tokens"][d] % 4
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2)[s::4],
                        np.arange(
                            -encoding["n_tokens"][d],
                            encoding["n_tokens"][d],
                        )[s::4],
                    )
                    plt.xlim(
                        encoding["n_tokens"][d] - 40.5,
                        encoding["n_tokens"][d] + 4.5,
                    )
                elif key == "position":
                    s = encoding["n_tokens"][d] % 4
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2)[s::4],
                        np.arange(
                            -encoding["n_tokens"][d],
                            encoding["n_tokens"][d],
                        )[s::4],
                    )
                    plt.xlim(
                        encoding["n_tokens"][d] - 12,
                        encoding["n_tokens"][d] + 12,
                    )
                elif key == "pitch":
                    s = encoding["n_tokens"][d] % 5
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2)[s::5],
                        np.arange(
                            -encoding["n_tokens"][d],
                            encoding["n_tokens"][d],
                        )[s::5],
                    )
                    plt.xlim(
                        encoding["n_tokens"][d] - 25.5,
                        encoding["n_tokens"][d] + 25.5,
                    )
                else:
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2),
                        np.arange(
                            -encoding["n_tokens"][d], encoding["n_tokens"][d]
                        ),
                    )
                plt.ylabel("Attention\nhead")
                plt.yticks(np.arange(n_heads), np.arange(n_heads) + 1)
                plt.xlabel(f"{key.capitalize()} difference")
                plt.tight_layout()
                plt.savefig(
                    sample_dir / f"{key}_rel3.png",
                    bbox_inches="tight"
                )
                plt.savefig(
                    sample_dir / f"{key}_rel3.pdf",
                    bbox_inches="tight"
                )
                plt.close()

            for d, key in enumerate(encoding["dimensions"]):
                if key not in ("beat", "position", "pitch"):
                    continue
                plt.figure(figsize=(6, 2))
                plt.imshow(
                    np.nan_to_num(
                        rel_event_attn[d]
                        / rel_event_attn[d].sum(-1, keepdims=True)
                    )
                    / np.nan_to_num(
                        rel_random_attn[d]
                        / rel_random_attn[d].sum(-1, keepdims=True)
                    ),
                    norm=matplotlib.colors.CenteredNorm(1),
                    cmap="bwr",
                    aspect="auto",
                    interpolation="none",
                )
                if key == "beat":
                    s = encoding["n_tokens"][d] % 4
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2)[s::4],
                        np.arange(
                            -encoding["n_tokens"][d],
                            encoding["n_tokens"][d],
                        )[s::4],
                    )
                    plt.xlim(
                        encoding["n_tokens"][d] - 40.5,
                        encoding["n_tokens"][d] + 4.5,
                    )
                elif key == "position":
                    s = encoding["n_tokens"][d] % 4
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2)[s::4],
                        np.arange(
                            -encoding["n_tokens"][d],
                            encoding["n_tokens"][d],
                        )[s::4],
                    )
                    plt.xlim(
                        encoding["n_tokens"][d] - 12,
                        encoding["n_tokens"][d] + 12,
                    )
                elif key == "pitch":
                    s = encoding["n_tokens"][d] % 5
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2)[s::5],
                        np.arange(
                            -encoding["n_tokens"][d],
                            encoding["n_tokens"][d],
                        )[s::5],
                    )
                    plt.xlim(
                        encoding["n_tokens"][d] - 25.5,
                        encoding["n_tokens"][d] + 25.5,
                    )
                else:
                    plt.xticks(
                        np.arange(encoding["n_tokens"][d] * 2),
                        np.arange(
                            -encoding["n_tokens"][d], encoding["n_tokens"][d]
                        ),
                    )
                plt.ylabel("Attention\nhead")
                plt.yticks(np.arange(n_heads), np.arange(n_heads) + 1)
                plt.xlabel(f"{key.capitalize()} difference")
                plt.tight_layout()
                plt.savefig(
                    sample_dir / f"{key}_rel4.png",
                    bbox_inches="tight"
                )
                plt.savefig(
                    sample_dir / f"{key}_rel4.pdf",
                    bbox_inches="tight"
                )
                plt.close()

if __name__ == "__main__":
    main()
