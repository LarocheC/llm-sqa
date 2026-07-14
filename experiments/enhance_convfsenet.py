"""
Speech enhancement with Clement's ConvFSENet (claroche1/sparse-nsnet2), driven
from the exported FP32 streaming ONNX — self-contained (no TF/Keras/model code).

Recipe mirrors convfsenet-hgq/stream_infer.py exactly:
  STFT(n_fft=512, hop=256, sqrt-Hann, center) -> compressed magnitude |S|^0.3
  -> ConvFSENet (frame-by-frame, 9 recurrent conv states) -> mask in [0,1]
  -> spec * mask -> ISTFT.
"""

import os

import numpy as np
import onnxruntime as ort
import torch
from experiments import config as cfg  # noqa: E402

NFFT, HOP, SR, COMPRESS = 512, 256, 16000, 0.3
DEFAULT_ONNX = os.environ.get(
    "SQA_CONVFSENET_ONNX",
    str(cfg.DATA_ROOT / "convfsenet" / "g_best_fp32.onnx"),
)


def _window():
    return torch.sqrt(torch.hann_window(NFFT, periodic=True))


class ConvFSENet:
    def __init__(self, onnx_path: str = DEFAULT_ONNX):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.state_inputs = [i for i in self.sess.get_inputs() if i.name.startswith("state_")]
        self.win = _window()

    def _init_states(self):
        return {i.name: np.zeros([1] + list(i.shape[1:]), dtype=np.float32) for i in self.state_inputs}

    def enhance(self, wav: np.ndarray) -> np.ndarray:
        # Recipe from convfsenet/inference_onnx.py (== eco8-neaixt):
        # RMS-normalize -> STFT (plain Hann, normalized=False) -> PLAIN magnitude
        # (graph compresses internally) -> frame loop -> spec*mask -> ISTFT -> /norm.
        x = torch.as_tensor(np.asarray(wav, dtype=np.float32))
        n = x.shape[-1]
        norm = torch.sqrt(len(x) / (torch.sum(x**2) + 1e-8))
        xn = (x * norm).unsqueeze(0)  # [1, L]
        win = torch.hann_window(NFFT)
        spec = torch.stft(xn, NFFT, HOP, NFFT, win, center=True, normalized=False, return_complex=True)  # [1,F,T]
        mag = spec.abs().numpy().astype(np.float32)  # [1,F,T]
        states = self._init_states()
        masks = []
        for t in range(mag.shape[2]):
            inp = {"noisy_mag": mag[:, :, t]}  # [1, 257]
            inp.update(states)
            out = self.sess.run(None, inp)
            masks.append(out[0])  # [1, 257]
            states = {f"state_{i}_in": out[i + 1] for i in range(len(self.state_inputs))}
        mask = torch.as_tensor(np.stack(masks, axis=2))  # [1,F,T]
        enhanced = torch.istft(spec * mask, NFFT, HOP, NFFT, win, center=True, normalized=False, length=n)
        return (enhanced / norm).squeeze(0).numpy()


if __name__ == "__main__":
    # quick verification: PESQ should improve noisy -> enhanced
    import glob, io, os
    import soundfile as sf
    import pyarrow.parquet as pq
    from pesq import pesq

    p = str(cfg.voicebank_parquet())
    rows = pq.read_table(p).slice(0, 6).to_pylist()
    enh = ConvFSENet()
    for row in rows:
        clean, _ = sf.read(io.BytesIO(row["clean"]["bytes"]))
        noisy, _ = sf.read(io.BytesIO(row["noisy"]["bytes"]))
        e = enh.enhance(noisy)
        L = min(len(clean), len(noisy), len(e))
        pn = pesq(SR, clean[:L].astype(np.float32), noisy[:L].astype(np.float32), "wb")
        pe = pesq(SR, clean[:L].astype(np.float32), e[:L].astype(np.float32), "wb")
        print(f"{row['id']}: PESQ noisy={pn:.2f} -> enhanced={pe:.2f}  (Δ{pe-pn:+.2f})")
