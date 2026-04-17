"""Visualization utilities replacing removed torchsig.utils.visualize and
torchsig.utils.cm_plotter APIs."""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def two_channel_to_complex(data: np.ndarray) -> np.ndarray:
    """Convert a (2, N) real array back to a complex (N,) array."""
    return data[0].astype(np.float64) + 1j * data[1].astype(np.float64)


def _to_complex(item: np.ndarray) -> np.ndarray:
    """Coerce a dataset item to a 1-D complex array."""
    if np.iscomplexobj(item):
        return item.ravel()
    if item.ndim == 2 and item.shape[0] == 2:
        return two_channel_to_complex(item)
    return item.ravel().astype(complex)


class IQVisualizer:
    """Iterates a DataLoader and yields matplotlib figures of IQ constellations."""

    def __init__(self, data_loader, num_samples: int = 36):
        self.data_loader = data_loader
        self.num_samples = num_samples

    def __iter__(self):
        for batch_data, _batch_labels in self.data_loader:
            if hasattr(batch_data, "numpy"):
                batch_data = batch_data.numpy()
            batch_data = np.asarray(batch_data)
            n = min(self.num_samples, len(batch_data))
            cols = max(1, int(np.ceil(np.sqrt(n))))
            rows = max(1, int(np.ceil(n / cols)))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            axes = np.array(axes).flatten()
            for i in range(n):
                iq = _to_complex(batch_data[i])
                axes[i].plot(iq.real, iq.imag, ".", markersize=1, alpha=0.5)
                axes[i].set_aspect("equal")
                axes[i].axis("off")
            for j in range(n, len(axes)):
                axes[j].axis("off")
            plt.tight_layout()
            yield fig
            break  # one figure per call to iter()


class SpectrogramVisualizer:
    """Iterates a DataLoader and yields matplotlib figures of spectrograms."""

    def __init__(
        self,
        data_loader,
        sample_rate: float = 1.0,
        window=None,
        nperseg: int = 256,
        nfft: Optional[int] = None,
        num_samples: int = 9,
    ):
        self.data_loader = data_loader
        self.sample_rate = sample_rate
        self.window = window
        self.nperseg = nperseg
        self.nfft = nfft
        self.num_samples = num_samples

    def __iter__(self):
        from scipy import signal as sp_signal

        for batch_data, _batch_labels in self.data_loader:
            if hasattr(batch_data, "numpy"):
                batch_data = batch_data.numpy()
            batch_data = np.asarray(batch_data)
            n = min(self.num_samples, len(batch_data))
            cols = max(1, int(np.ceil(np.sqrt(n))))
            rows = max(1, int(np.ceil(n / cols)))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
            axes = np.array(axes).flatten()
            for i in range(n):
                iq = _to_complex(batch_data[i])
                f, t, Sxx = sp_signal.spectrogram(
                    iq,
                    fs=self.sample_rate,
                    window=self.window if self.window is not None else "hann",
                    nperseg=self.nperseg,
                    nfft=self.nfft,
                    return_onesided=False,
                )
                Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
                axes[i].pcolormesh(
                    t,
                    np.fft.fftshift(f),
                    np.fft.fftshift(Sxx_db, axes=0),
                    shading="gouraud",
                )
                axes[i].axis("off")
            for j in range(n, len(axes)):
                axes[j].axis("off")
            plt.tight_layout()
            yield fig
            break


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes=None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap=None,
    text: bool = True,
    rotate_x_text: int = 0,
    figsize=(10, 8),
) -> Figure:
    from sklearn.metrics import confusion_matrix

    if cmap is None:
        cmap = plt.cm.Blues

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)

    if classes is not None:
        ticks = range(len(classes))
        ax.set_xticks(list(ticks))
        ax.set_yticks(list(ticks))
        ax.set_xticklabels(classes, rotation=rotate_x_text, ha="right")
        ax.set_yticklabels(classes)

    if text:
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
                ax.text(
                    j,
                    i,
                    val,
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=6,
                )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    fig.tight_layout()
    return fig
