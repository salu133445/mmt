import argparse
import logging
import pathlib
import pprint
import shutil
import sys

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
    parser.add_argument(
        "-t", "--train_names", type=pathlib.Path, help="training names"
    )
    parser.add_argument(
        "-v", "--valid_names", type=pathlib.Path, help="validation names"
    )
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    # Data
    parser.add_argument(
        "-c",
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=8,
        type=int,
        help="batch size",
    )
    # Model
    parser.add_argument(
        "-m",
        "--max_seq_len",
        default=1024,
        type=int,
        help="maximum sequence length",
    )
    parser.add_argument(
        "-mb",
        "--max_beat",
        default=256,
        type=int,
        help="maximum number of beats",
    )
    # Training
    parser.add_argument(
        "-s",
        "--steps",
        default=100000,
        type=int,
        help="number of steps",
    )
    parser.add_argument(
        "-vs",
        "--valid_steps",
        default=1000,
        type=int,
        help="validation frequency",
    )
    parser.add_argument(
        "-ne",
        "--disable_early_stopping",
        action="store_true",
        help="whether to disable early stopping",
    )
    parser.add_argument(
        "-e",
        "--early_stopping_tolerance",
        default=10,
        type=int,
        help="number of extra validation rounds before early stopping",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.001,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "-lrw",
        "--learning_rate_warmup_steps",
        default=100,
        type=int,
        help="learning rate warmup steps",
    )
    parser.add_argument(
        "-lrm",
        "--learning_rate_decay_end_multiplier",
        default=0.1,
        type=float,
        help="learning rate multiplier at the end",
    )
    parser.add_argument(
        "-lrs",
        "--learning_rate_decay_end_steps",
        default=100000,
        type=int,
        help="learning rate decay end steps",
    )
    parser.add_argument(
        "-dim", "--dim", default=512, type=int, help="model dimension"
    )
    parser.add_argument(
        "-l", "--layers", default=8, type=int, help="number of layers"
    )
    parser.add_argument(
        "-ah", "--heads", default=4, type=int, help="number of attention heads"
    )
    parser.add_argument(
        "-do", "--dropout", default=0.1, type=float, help="dropout rate"
    )
    parser.add_argument(
        "-na",
        "--disable_augmentation",
        action="store_true",
        help="whether to use augmentation",
    )
    parser.add_argument(
        "-nr",
        "--disable_relative_positional_embedding",
        action="store_true",
        help="whether to disable relative positional embedding",
    )
    parser.add_argument("-g", "--gpu", type=int, help="gpu number")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="number of jobs (deafult to `min(batch_size, 8)`)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def get_lr_multiplier(
    step, warmup_steps, decay_end_steps, decay_end_multiplier
):
    """Return the learning rate multiplier with a warmup and decay schedule.

    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.

    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.train_names is None:
            args.train_names = pathlib.Path(
                f"data/{args.dataset}/processed/train-names.txt"
            )
        if args.valid_names is None:
            args.valid_names = pathlib.Path(
                f"data/{args.dataset}/processed/valid-names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes/")
        if args.out_dir is None:
            args.out_dir = pathlib.Path(f"exp/test_{args.dataset}")
    if args.jobs is None:
        args.jobs = min(args.batch_size, 8)

    # Make sure the output directory exists
    args.out_dir.mkdir(exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)

    # Save command-line arguments
    utils.save_args(args.out_dir / "train-args.json", args)

    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "train.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Get the specified device
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Load the encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Create the dataset and data loader
    logging.info(f"Creating the data loader...")
    train_dataset = dataset.SODDataset(
        args.train_names,
        args.in_dir,
        encoding,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_augmentation=not args.disable_augmentation,
        use_csv=args.use_csv,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.jobs,
        collate_fn=dataset.SODDataset.collate,
    )
    valid_dataset = dataset.SODDataset(
        args.valid_names,
        args.in_dir,
        encoding,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_csv=args.use_csv,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        args.batch_size,
        num_workers=args.jobs,
        collate_fn=dataset.SODDataset.collate,
    )

    # Create the model
    logging.info(f"Creating model...")
    model = music_x_transformer.MusicXTransformer(
        dim=args.dim,
        encoding=encoding,
        depth=args.layers,
        heads=args.heads,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        rel_pos_bias=not args.disable_relative_positional_embedding,
        rotary_pos_emb=not args.disable_relative_positional_embedding,
        emb_dropout=args.dropout,
        attn_dropout=args.dropout,
        ff_dropout=args.dropout,
    ).to(device)

    # Summarize the model
    n_parameters = sum(p.numel() for p in model.parameters())
    n_trainables = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logging.info(f"Number of parameters: {n_parameters}")
    logging.info(f"Number of trainable parameters: {n_trainables}")

    # Create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_multiplier(
            step,
            args.learning_rate_warmup_steps,
            args.learning_rate_decay_end_steps,
            args.learning_rate_decay_end_multiplier,
        ),
    )

    # Create a file to record losses
    loss_csv = open(args.out_dir / "loss.csv", "w")
    loss_csv.write(
        "step,train_loss,valid_loss,type_loss,beat_loss,position_loss,"
        "pitch_loss,duration_loss,instrument_loss\n"
    )

    # Initialize variables
    step = 0
    min_val_loss = float("inf")
    if not args.disable_early_stopping:
        count_early_stopping = 0

    # Iterate for the specified number of steps
    train_iterator = iter(train_loader)
    while step < args.steps:

        # Training
        logging.info(f"Training...")
        model.train()
        recent_losses = []

        for batch in (pbar := tqdm.tqdm(range(args.valid_steps), ncols=80)):
            # Get next batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Reinitialize dataset iterator
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            # Get input and output pair
            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)

            # Update the model parameters
            optimizer.zero_grad()
            loss = model(seq, mask=mask)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Compute the moving average of the loss
            recent_losses.append(float(loss))
            if len(recent_losses) > 10:
                del recent_losses[0]
            train_loss = sum(recent_losses) / len(recent_losses)
            pbar.set_postfix(loss=f"{train_loss:8.4f}")

            step += 1

        # Release GPU memory right away
        del seq, mask

        # Validation
        logging.info(f"Validating...")
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_losses = [0] * 6
            count = 0
            for batch in valid_loader:
                # Get input and output pair
                seq = batch["seq"].to(device)
                mask = batch["mask"].to(device)

                # Pass through the model
                loss, losses = model(seq, return_list=True, mask=mask)

                # Accumulate validation loss
                count += len(batch)
                total_loss += len(batch) * float(loss)
                for idx in range(6):
                    total_losses[idx] += float(losses[idx])
        val_loss = total_loss / count
        logging.info(f"Validation loss: {val_loss:.4f}")
        logging.info(
            f"Individual losses: type={total_losses[0] / count:.4f}, "
            f"beat: {total_losses[1] / count:.4f}, "
            f"position: {total_losses[2] / count:.4f}, "
            f"pitch: {total_losses[3] / count:.4f}, "
            f"duration: {total_losses[4] / count:.4f}, "
            f"instrument: {total_losses[5] / count:.4f}"
        )

        # Release GPU memory right away
        del seq, mask

        # Write losses to file
        loss_csv.write(
            f"{step},{train_loss},{val_loss},{total_losses[0]},"
            f"{total_losses[1]},{total_losses[2]},{total_losses[3]},"
            f"{total_losses[4]},{total_losses[5]}\n"
        )

        # Save the model
        checkpoint_filename = args.out_dir / "checkpoints" / f"model_{step}.pt"
        torch.save(model.state_dict(), checkpoint_filename)
        logging.info(f"Saved the model to: {checkpoint_filename}")

        # Copy the model if it is the best model so far
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            shutil.copyfile(
                checkpoint_filename,
                args.out_dir / "checkpoints" / "best_model.pt",
            )
            # Reset the early stopping counter if we found a better model
            if not args.disable_early_stopping:
                count_early_stopping = 0
        elif not args.disable_early_stopping:
            # Increment the early stopping counter if no improvement is found
            count_early_stopping += 1

        # Early stopping
        if (
            not args.disable_early_stopping
            and count_early_stopping > args.early_stopping_tolerance
        ):
            logging.info(
                "Stopped the training for no improvements in "
                f"{args.early_stopping_tolerance} rounds."
            )
            break

    # Save the optimizer states
    optimizer_filename = args.out_dir / "checkpoints" / f"optimizer_{step}.pt"
    torch.save(optimizer.state_dict(), optimizer_filename)
    logging.info(f"Saved the optimizer state to: {optimizer_filename}")

    # Save the scheduler states
    scheduler_filename = args.out_dir / "checkpoints" / f"scheduler_{step}.pt"
    torch.save(scheduler.state_dict(), scheduler_filename)
    logging.info(f"Saved the scheduler state to: {scheduler_filename}")

    # Close the file
    loss_csv.close()


if __name__ == "__main__":
    main()
