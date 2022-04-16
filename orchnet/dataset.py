"""Data loader."""
import argparse
import logging
import pathlib
import pprint
import sys

import numpy as np
import torch
import torch.utils.data

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
        "-b",
        "--batch_size",
        default=8,
        type=int,
        help="batch size",
    )
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
    parser.add_argument(
        "-c",
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "-na",
        "--disable_augmentation",
        action="store_true",
        help="whether to disable data augmentation",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def pad(data, maxlen=None):
    if maxlen is None:
        max_len = max(len(x) for x in data)
    else:
        for x in data:
            assert len(x) <= max_len
    if data[0].ndim == 1:
        padded = [np.pad(x, (0, max_len - len(x))) for x in data]
    elif data[0].ndim == 2:
        padded = [np.pad(x, ((0, max_len - len(x)), (0, 0))) for x in data]
    else:
        raise ValueError("Got 3D data.")
    return np.stack(padded)


def get_mask(data):
    max_seq_len = max(len(sample) for sample in data)
    mask = torch.zeros((len(data), max_seq_len), dtype=torch.bool)
    for i, seq in enumerate(data):
        mask[i, : len(seq)] = 1
    return mask


class SODDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        data_dir,
        encoding,
        max_seq_len=None,
        max_beat=None,
        use_csv=False,
        use_augmentation=False,
    ):
        # TODO: Use notes as inputs rather codes
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        with open(filename) as f:
            self.names = [line.strip() for line in f if line]
        self.encoding = encoding
        self.max_seq_len = max_seq_len
        self.max_beat = max_beat
        self.use_csv = use_csv
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # Get the name
        name = self.names[idx]

        # Load data
        if self.use_csv:
            notes = utils.load_csv(self.data_dir / f"{name}.csv")
        else:
            notes = np.load(self.data_dir / f"{name}.npy")

        # Check the shape of the loaded notes
        assert notes.shape[1] == 5

        # Data augmentation
        if self.use_augmentation:
            # Shift all the pitches for k semitones (k~Uniform(-5, 6))
            pitch_shift = np.random.randint(-5, 7)
            notes[:, 2] = np.clip(notes[:, 2] + pitch_shift, 0, 127)

            # Randomly select a starting beat
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                start_beat = np.random.randint(n_beats - self.max_beat)
                end_beat = start_beat + self.max_beat
                notes = notes[
                    (notes[:, 0] >= start_beat) & (notes[:, 0] < end_beat)
                ]
                notes[:, 0] = notes[:, 0] - start_beat

        # Trim sequence to max_beat
        elif self.max_beat is not None:
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                notes = notes[notes[:, 0] < self.max_beat]

        # Encode the notes
        seq = representation.encode_notes(notes, self.encoding)

        # Trim sequence to max_seq_len
        if self.max_seq_len is not None and len(seq) > self.max_seq_len:
            seq = np.concatenate((seq[: self.max_seq_len - 1], seq[-1:]))

        # if np.any(seq[:, 1] > self.max_beat):
        #     print(notes[:10])
        #     print("...")
        #     print(notes[-10:])
        #     print(seq[:10])
        #     print("...")
        #     print(seq[-10:])
        #     assert False

        return {"name": name, "seq": seq}

    @classmethod
    def collate(cls, data):
        seq = [sample["seq"] for sample in data]
        return {
            "name": [sample["name"] for sample in data],
            "seq": torch.tensor(pad(seq), dtype=torch.long),
            "seq_len": torch.tensor([len(s) for s in seq], dtype=torch.long),
            "mask": get_mask(seq),
        }


def to_device(data, device):
    for key, value in data.items():
        if key in ("name", "programs"):
            continue
        data[key] = value.to(device)


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = pathlib.Path(
                f"data/{args.dataset}/processed/names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes")

    # Load the encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Set up the logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Create the dataset and data loader
    dataset = SODDataset(
        args.names,
        args.in_dir,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_csv=args.use_csv,
        use_augmentation=not args.disable_augmentation,
        encoding=encoding,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, args.batch_size, True, collate_fn=SODDataset.collate
    )

    # Iterate over the loader
    n_batches = 0
    n_samples = 0
    seq_lens = []
    for i, batch in enumerate(data_loader):
        n_batches += 1
        n_samples += len(batch["name"])
        seq_lens.extend(int(l) for l in batch["seq_len"])
        if i == 0:
            logging.info("Example:")
            for key, value in batch.items():
                if key == "name":
                    continue
                logging.info(f"Shape of {key}: {value.shape}")
            logging.info(f"Name: {batch['name'][0]}")
    logging.info(
        f"Successfully loaded {n_batches} batches ({n_samples} samples)."
    )

    # Print sequence length statistics
    logging.info(f"Avg sequence length: {sum(seq_lens) / len(seq_lens):2f}")
    logging.info(f"Min sequence length: {min(seq_lens)}")
    logging.info(f"Max sequence length: {max(seq_lens)}")


if __name__ == "__main__":
    main()
