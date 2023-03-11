import argparse
import logging
import pathlib
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics.pairwise
import torch
import torch.utils.data

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
    # Model
    parser.add_argument(
        "--model_steps",
        type=int,
        help="step of the trained model to load (default to the best model)",
    )
    # Others
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
            logging.FileHandler(args.out_dir / "visualize-embedding.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(
        f"Saved arguments to {args.out_dir / 'visualize-embedding-args.json'}"
    )
    utils.save_args(args.out_dir / "visualize-embedding-args.json", args)

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
    device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Load the encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

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

    if train_args["abs_pos_emb"]:
        cos_sim = sklearn.metrics.pairwise.cosine_similarity(
            model.decoder.net.pos_emb.emb.weight.detach().numpy()
        )
        plt.figure(figsize=(4, 4))
        plt.imshow(cos_sim, cmap="inferno", interpolation="none")
        im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
        plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
        plt.xlabel("Position")
        plt.ylabel("Position")
        plt.tight_layout()
        plt.savefig(sample_dir / f"sim-pos-emb.png", bbox_inches="tight")
        plt.savefig(sample_dir / f"sim-pos-emb.pdf", bbox_inches="tight")
        plt.close()

    for idx, key in enumerate(encoding["dimensions"]):
        weight = model.decoder.net.token_emb[idx].emb.weight.detach().numpy()
        if key != "type":
            weight = weight[1:]
        cos_sim = sklearn.metrics.pairwise.cosine_similarity(weight)
        plt.figure(figsize=(4, 4))
        plt.imshow(cos_sim, cmap="inferno", interpolation="none")
        im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
        plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
        plt.xlabel(key.capitalize())
        plt.ylabel(key.capitalize())
        plt.tight_layout()
        plt.savefig(sample_dir / f"sim-emb-{key}.png", bbox_inches="tight")
        plt.savefig(sample_dir / f"sim-emb-{key}.pdf", bbox_inches="tight")
        plt.close()

        if key == "beat":
            plt.figure(figsize=(4, 4))
            plt.imshow(cos_sim[:50, :50], cmap="inferno", interpolation="none")
            im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
            plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
            plt.xlabel(key.capitalize())
            plt.ylabel(key.capitalize())
            plt.tight_layout()
            plt.savefig(
                sample_dir / f"sim-emb-{key}-cropped.png", bbox_inches="tight"
            )
            plt.savefig(
                sample_dir / f"sim-emb-{key}-cropped.pdf", bbox_inches="tight"
            )
            plt.close()

        weight = model.decoder.net.to_logits[idx].weight.detach().numpy()
        if key != "type":
            weight = weight[1:]
        cos_sim = sklearn.metrics.pairwise.cosine_similarity(weight)
        plt.figure(figsize=(4, 4))
        plt.imshow(cos_sim, cmap="inferno", interpolation="none")
        im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
        plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
        plt.xlabel(key.capitalize())
        plt.ylabel(key.capitalize())
        plt.tight_layout()
        plt.savefig(sample_dir / f"sim-out-{key}.png", bbox_inches="tight")
        plt.savefig(sample_dir / f"sim-out-{key}.pdf", bbox_inches="tight")
        plt.close()

        if key == "beat":
            plt.figure(figsize=(4, 4))
            plt.imshow(cos_sim[:50, :50], cmap="inferno", interpolation="none")
            im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
            plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
            plt.xlabel(key.capitalize())
            plt.ylabel(key.capitalize())
            plt.tight_layout()
            plt.savefig(
                sample_dir / f"sim-out-{key}-cropped.png", bbox_inches="tight"
            )
            plt.savefig(
                sample_dir / f"sim-out-{key}-cropped.pdf", bbox_inches="tight"
            )
            plt.close()


if __name__ == "__main__":
    main()
