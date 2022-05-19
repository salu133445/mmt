import argparse
import logging
import pathlib
import pprint
import sys
import time

import torch
import torch.utils.data
import tqdm
import x_transformers
import x_transformers.autoregressive_wrapper

import representation_mmm
import representation_remi
import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "snd"),
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
        "-ns",
        "--n_samples",
        default=100,
        type=int,
        help="number of samples to generate",
    )
    parser.add_argument(
        "-m",
        "--model_steps",
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
        "-f",
        "--filter",
        default="top_k",
        type=str,
        help="sampling filter (default: 'top_k')",
    )
    parser.add_argument(
        "-ft",
        "--filter_threshold",
        default=0.9,
        type=float,
        help="sampling filter threshold (default: 0.9)",
    )
    parser.add_argument("-g", "--gpu", type=int, help="gpu number")
    parser.add_argument(
        "-j", "--jobs", default=1, type=int, help="number of jobs"
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
            logging.FileHandler(args.out_dir / "speed-test.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(f"Saved arguments to {args.out_dir / 'speed-test-args.json'}")
    utils.save_args(args.out_dir / "speed-test-args.json", args)

    # Load training configurations
    logging.info(
        f"Loading training arguments from: {args.out_dir / 'train-args.json'}"
    )
    train_args = utils.load_json(args.out_dir / "train-args.json")
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")

    # Get the specified device
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Get representation
    if train_args["representation"] == "mmm":
        representation = representation_mmm
    elif train_args["representation"] == "remi":
        representation = representation_remi
    else:
        raise ValueError(
            f"Unknown representation: {train_args['representation']}"
        )

    # Load the encoding
    encoding = representation.get_encoding()

    # Load the indexer
    indexer = representation.Indexer(encoding["event_code_map"])

    # Get the vocabulary
    vocabulary = encoding["code_event_map"]

    # Create the model
    logging.info(f"Creating the model...")
    model = x_transformers.TransformerWrapper(
        num_tokens=len(indexer),
        max_seq_len=train_args["max_seq_len"],
        attn_layers=x_transformers.Decoder(
            dim=train_args["dim"],
            depth=train_args["layers"],
            heads=train_args["heads"],
            rotary_pos_emb=train_args["rel_pos_emb"],
            emb_dropout=train_args["dropout"],
            attn_dropout=train_args["dropout"],
            ff_dropout=train_args["dropout"],
        ),
        use_abs_pos_emb=train_args["abs_pos_emb"],
    ).to(device)
    model = x_transformers.AutoregressiveWrapper(model)

    # Load the checkpoint
    checkpoint_dir = args.out_dir / "checkpoints"
    if args.model_steps is None:
        checkpoint_filename = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_filename = checkpoint_dir / f"model_{args.model_steps}.pt"
    model.load_state_dict(torch.load(checkpoint_filename, map_location=device))
    logging.info(f"Loaded the model weights from: {checkpoint_filename}")
    model.eval()

    # Get special tokens
    sos = indexer["start-of-song"]
    eos = indexer["end-of-song"]

    # Get the logits filter function
    if args.filter == "top_k":
        filter_logits_fn = x_transformers.autoregressive_wrapper.top_k
    elif args.filter == "top_p":
        filter_logits_fn = x_transformers.autoregressive_wrapper.top_p
    elif args.filter == "top_a":
        filter_logits_fn = x_transformers.autoregressive_wrapper.top_a
    else:
        raise ValueError("Unknown logits filter.")

    # Iterate over
    total_time = 0
    total_notes = 0
    total_length = 0
    with torch.no_grad():

        for _ in tqdm.tqdm(range(args.n_samples), ncols=80):

            # Get output start tokens
            tgt_start = torch.zeros((1, 1), dtype=torch.long, device=device)
            tgt_start[:, 0] = sos

            # Generate new samples
            start = time.time()
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token=eos,
                temperature=args.temperature,
                filter_logits_fn=filter_logits_fn,
                filter_thres=args.filter_threshold,
            )
            total_time += time.time() - start

            # Decode the generated codes
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()
            notes = representation.decode_notes(
                generated_np[0], encoding, vocabulary
            )
            total_notes += len(notes)

            # Reconstruct the music
            music = representation.reconstruct(notes, encoding["resolution"])
            total_length += music.get_real_end_time()

        logging.info(
            f"Computing time per sample: {total_time  / args.n_samples:.6f} s"
        )
        logging.info(
            f"Computing time per note: {total_time  / total_notes:.6f} s"
        )
        logging.info(
            f"Length per sample: {total_length  / args.n_samples:.6f} s"
        )
        logging.info(
            f"Number of notes per sample: {total_notes  / args.n_samples:.6f}"
        )


if __name__ == "__main__":
    main()
