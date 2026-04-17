# PyTorch dataset from SigMF

import dataclasses
import glob
import json
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import zstandard
from torch.utils.data import Dataset


# SigMF datatype string → numpy dtype for a single component (I or Q).
# For complex types (prefix 'c') the raw array is interleaved [I, Q, I, Q, …]
# and sample_size = itemsize * 2; for real types it is itemsize * 1.
SIGMF_DTYPE_MAP = {
    "cf32_le": np.dtype("<f4"),
    "cf64_le": np.dtype("<f8"),
    "ci32_le": np.dtype("<i4"),
    "ci16_le": np.dtype("<i2"),
    "ci8":     np.dtype("int8"),
    "cu32_le": np.dtype("<u4"),
    "cu16_le": np.dtype("<u2"),
    "cu8":     np.dtype("uint8"),
    "rf32_le": np.dtype("<f4"),
    "rf64_le": np.dtype("<f8"),
    "ri32_le": np.dtype("<i4"),
    "ri16_le": np.dtype("<i2"),
    "ri8":     np.dtype("int8"),
    "ru32_le": np.dtype("<u4"),
    "ru16_le": np.dtype("<u2"),
    "ru8":     np.dtype("uint8"),
}


@dataclasses.dataclass
class SignalDescription:
    sample_rate: float = 0.0
    upper_frequency: float = 0.0
    lower_frequency: float = 0.0


@dataclasses.dataclass
class SignalCapture:
    absolute_path: str
    num_bytes: int
    byte_offset: int
    item_type: np.dtype
    is_complex: bool
    signal_description: SignalDescription = dataclasses.field(
        default_factory=SignalDescription
    )


def _bytes_to_iq(raw: bytes, item_type: np.dtype, is_complex: bool) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=item_type)
    if is_complex:
        return (arr[0::2] + 1j * arr[1::2]).astype(np.complex64)
    return arr.astype(np.float32)


def reader_from_sigmf(signal_capture: SignalCapture) -> np.ndarray:
    with open(signal_capture.absolute_path, "rb") as f:
        f.seek(signal_capture.byte_offset)
        raw = f.read(signal_capture.num_bytes)
    return _bytes_to_iq(raw, signal_capture.item_type, signal_capture.is_complex)


def _reader_from_zst(signal_capture: SignalCapture) -> np.ndarray:
    with zstandard.ZstdDecompressor().stream_reader(
        open(signal_capture.absolute_path, "rb"), read_across_frames=True
    ) as f:
        f.seek(signal_capture.byte_offset)
        raw = f.read(signal_capture.num_bytes)
    return _bytes_to_iq(raw, signal_capture.item_type, signal_capture.is_complex)


class SigMFDataset(Dataset):
    """Mappable PyTorch dataset built from annotated SigMF files.

    Args:
        root: Root path(s) to search recursively for SigMF files.
        sample_count: Number of IQ samples per example.
        index_filter: Optional predicate to drop index entries.
        class_list: Ordered list of class names; extended automatically.
        allowed_filetypes: File extensions to scan for.
        only_first_samples: Use only the first ``sample_count`` samples of
            each annotation rather than splitting into multiple examples.
        transform: Callable applied to the raw complex numpy array.
        target_transform: Callable applied to the integer label.
    """

    def __init__(
        self,
        root: Union[str, List[str]],
        sample_count: int = 2048,
        index_filter: Optional[Callable[[Tuple[Any, SignalCapture]], bool]] = None,
        class_list: Optional[List[str]] = None,
        allowed_filetypes: Optional[List[str]] = None,
        only_first_samples: bool = True,
        transform=None,
        target_transform=None,
        **_kwargs,
    ):
        super().__init__()
        if allowed_filetypes is None:
            allowed_filetypes = [".sigmf-data", ".sigmf-meta"]
        self.sample_count = sample_count
        self.allowed_classes = class_list.copy() if class_list else []
        self.class_list = class_list if class_list else []
        self.allowed_filetypes = allowed_filetypes
        self.only_first_samples = only_first_samples
        self.transform = transform
        self.target_transform = target_transform
        if isinstance(root, str):
            root = [root]
        self.index_files: List[str] = []
        self.index = self.indexer_from_sigmf_annotations(root)
        if index_filter:
            self.index = list(filter(index_filter, self.index))

    def get_indices(self, indices=None):
        if not indices:
            return self.index
        return map(self.index.__getitem__, indices)

    def get_class_counts(self, indices=None) -> dict:
        class_counts = {idx: 0 for idx in range(len(self.class_list))}
        for label_idx, _ in self.get_indices(indices):
            class_counts[label_idx] += 1
        return class_counts

    def get_weighted_sampler(self, indices=None) -> torch.utils.data.WeightedRandomSampler:
        class_counts = self.get_class_counts(indices)
        weight = 1.0 / np.array(list(class_counts.values()))
        samples_weight = np.array([weight[t] for t, _ in self.get_indices(indices)])
        samples_weight = torch.from_numpy(samples_weight)
        return torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    def get_data(self, signal_capture: SignalCapture) -> np.ndarray:
        if signal_capture.absolute_path.endswith(".sigmf-data"):
            return reader_from_sigmf(signal_capture)
        if signal_capture.absolute_path.endswith(".zst"):
            return _reader_from_zst(signal_capture)
        raise ValueError(
            f"Cannot read {signal_capture.absolute_path}: unsupported file type."
        )

    def __getitem__(self, item: int) -> Tuple[np.ndarray, Any]:
        target, signal_capture = self.index[item]
        iq_data = self.get_data(signal_capture)
        if self.transform:
            iq_data = self.transform(iq_data)
        if self.target_transform:
            target = self.target_transform(target)
        return iq_data, target

    def __len__(self) -> int:
        return len(self.index)

    def indexer_from_sigmf_annotations(
        self, root: List[str]
    ) -> List[Tuple[Any, SignalCapture]]:
        index = []
        for file_type in self.allowed_filetypes:
            for r in root:
                if os.path.isfile(r):
                    file_list = [f"{os.path.splitext(r)[0]}.sigmf-data"]
                elif os.path.isdir(r):
                    file_list = glob.glob(
                        os.path.join(r, "**", "*" + file_type), recursive=True
                    )
                else:
                    raise ValueError(f"Path does not exist: {r}")
                for f in file_list:
                    meta_path = f"{os.path.splitext(f)[0]}.sigmf-meta"
                    if os.path.isfile(meta_path):
                        data_path = f"{os.path.splitext(f)[0]}.sigmf-data"
                        signals = self._parse_sigmf_annotations(data_path)
                        if signals:
                            index.extend(signals)
        self.index_files = list(set(self.index_files))
        return index

    def _get_name_to_idx(self, name: str) -> int:
        try:
            return self.class_list.index(name)
        except ValueError:
            print(f"Adding {name} to class list")
            self.class_list.append(name)
            return self.class_list.index(name)

    def _parse_sigmf_annotations(
        self, absolute_file_path: str
    ) -> List[Tuple[Any, SignalCapture]]:
        meta_path = f"{os.path.splitext(absolute_file_path)[0]}.sigmf-meta"
        meta = json.load(open(meta_path, "r"))
        item_type = SIGMF_DTYPE_MAP[meta["global"]["core:datatype"]]
        sample_size = item_type.itemsize * (
            2 if "c" in meta["global"]["core:datatype"] else 1
        )
        total_num_samples = os.path.getsize(absolute_file_path) // sample_size  # noqa: F841

        index: List[Tuple[Any, SignalCapture]] = []
        if len(meta["captures"]) == 1:
            for annotation in meta["annotations"]:
                label = annotation.get("core:label")
                if self.allowed_classes and (label not in self.allowed_classes):
                    continue
                if annotation["core:sample_count"] < self.sample_count:
                    continue

                signal_description = SignalDescription(
                    sample_rate=meta["global"].get("core:sample_rate", 0.0),
                )
                signal_description.upper_frequency = annotation.get(
                    "core:freq_upper_edge", 0.0
                )
                signal_description.lower_frequency = annotation.get(
                    "core:freq_lower_edge", 0.0
                )

                subparts = int(annotation["core:sample_count"] / self.sample_count)
                if self.only_first_samples:
                    subparts = 1

                for i in range(subparts):
                    sample_start = annotation["core:sample_start"] + i * self.sample_count
                    capture = SignalCapture(
                        absolute_path=absolute_file_path,
                        num_bytes=sample_size * self.sample_count,
                        byte_offset=sample_size * sample_start,
                        item_type=item_type,
                        is_complex="c" in meta["global"]["core:datatype"],
                        signal_description=signal_description,
                    )
                    index.append((self._get_name_to_idx(label), capture))

                self.index_files.append(absolute_file_path)
        else:
            print(
                "Not clear how to handle annotations when there is more than one capture"
            )
        return index
