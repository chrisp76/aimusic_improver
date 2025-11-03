import tempfile
import os
import zipfile
import importlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import gradio as gr
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt
import noisereduce as nr
import pyloudnorm as pyln
from pydub import AudioSegment


# ----------------------
# Audio DSP utilities
# ----------------------

def _butter_sos(filter_type: str, cutoff_hz: float, sample_rate: int, order: int = 4):
    nyq = 0.5 * sample_rate
    normalized = cutoff_hz / nyq
    return butter(order, normalized, btype=filter_type, output="sos")


def _butter_band_sos(low_hz: float, high_hz: float, sample_rate: int, order: int = 4):
    nyq = 0.5 * sample_rate
    low = low_hz / nyq
    high = high_hz / nyq
    return butter(order, [low, high], btype="band", output="sos")


def highpass(signal: np.ndarray, sample_rate: int, cutoff_hz: float = 40.0) -> np.ndarray:
    sos = _butter_sos("highpass", cutoff_hz, sample_rate, order=2)
    return sosfilt(sos, signal)


def lowpass(signal: np.ndarray, sample_rate: int, cutoff_hz: float = 19000.0) -> np.ndarray:
    sos = _butter_sos("lowpass", cutoff_hz, sample_rate, order=2)
    return sosfilt(sos, signal)


def bandpass(signal: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    sos = _butter_band_sos(low_hz, high_hz, sample_rate, order=2)
    return sosfilt(sos, signal)


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    mean = np.mean(signal, axis=-1, keepdims=True)
    return signal - mean


def soft_limiter(signal: np.ndarray, ceiling_db: float = -0.1) -> np.ndarray:
    ceiling_linear = 10 ** (ceiling_db / 20.0)
    return np.clip(signal, -ceiling_linear, ceiling_linear)


def soft_saturate(signal: np.ndarray, drive: float = 1.5) -> np.ndarray:
    driven = drive * signal
    saturated = np.tanh(driven)
    return saturated / np.max(np.abs(saturated) + 1e-9)


def simple_compressor(signal: np.ndarray, sample_rate: int, threshold_db: float = -18.0, ratio: float = 2.0, attack_ms: float = 10.0, release_ms: float = 80.0) -> np.ndarray:
    # Envelope follower
    attack_coeff = np.exp(-1.0 / (attack_ms * 0.001 * sample_rate))
    release_coeff = np.exp(-1.0 / (release_ms * 0.001 * sample_rate))

    abs_signal = np.abs(signal) + 1e-12
    env = np.zeros_like(signal)
    for i in range(signal.shape[-1]):
        if i == 0:
            env[..., i] = abs_signal[..., i]
        else:
            coeff = attack_coeff if abs_signal[..., i] > env[..., i - 1] else release_coeff
            env[..., i] = (1 - coeff) * abs_signal[..., i] + coeff * env[..., i - 1]

    env_db = 20.0 * np.log10(env + 1e-12)
    over_db = np.maximum(0.0, env_db - threshold_db)
    gain_reduction_db = over_db - over_db / ratio
    gain = 10 ** (-gain_reduction_db / 20.0)
    return signal * gain


def de_esser(signal: np.ndarray, sample_rate: int, s_band: Tuple[float, float] = (5000.0, 12000.0), threshold_db: float = -24.0, ratio: float = 4.0) -> np.ndarray:
    # Extract sibilant band
    s_band_signal = bandpass(signal, sample_rate, s_band[0], s_band[1])
    # Level detect on band
    env = np.maximum(1e-12, np.abs(s_band_signal))
    env_db = 20.0 * np.log10(env)
    over_db = np.maximum(0.0, env_db - threshold_db)
    reduction_db = over_db - over_db / ratio
    gain = 10 ** (-reduction_db / 20.0)
    # Only reduce the band portion; keep fundamentals intact
    reduced_band = s_band_signal * gain
    # Recombine by replacing a portion of the band with reduced version
    return signal - s_band_signal + reduced_band


def reduce_noise(signal: np.ndarray, sample_rate: int, strength: float = 0.6) -> np.ndarray:
    # strength in [0, 1], map to prop_decrease ~ [0.2, 1.0]
    prop = 0.2 + 0.8 * np.clip(strength, 0.0, 1.0)
    return nr.reduce_noise(y=signal, sr=sample_rate, prop_decrease=prop, stationary=False, use_tensorflow=False)


def lufs_normalize(signal: np.ndarray, sample_rate: int, target_lufs: float = -14.0) -> np.ndarray:
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(signal)
    if np.isfinite(loudness):
        gain_db = target_lufs - loudness
        gain = 10 ** (gain_db / 20.0)
        return signal * gain
    return signal


def dereverb_simple(signal: np.ndarray, sample_rate: int, strength: float = 0.0) -> np.ndarray:
    """Very lightweight dereverb via sustain suppression using dual-window envelope ratio.
    strength in [0,1].
    """
    if strength <= 0.0:
        return signal
    eps = 1e-8
    win_short = max(1, int(0.015 * sample_rate))
    win_long = max(win_short + 1, int(0.200 * sample_rate))

    kernel_short = np.ones(win_short, dtype=np.float32) / float(win_short)
    kernel_long = np.ones(win_long, dtype=np.float32) / float(win_long)

    abs_sig = np.abs(signal)
    short_env = np.convolve(abs_sig, kernel_short, mode="same")
    long_env = np.convolve(abs_sig, kernel_long, mode="same")

    tail_ratio = np.clip((long_env + eps) / (short_env + eps), 1.0, 10.0)
    # Map ratio to gain reduction controlled by strength
    gain = 1.0 / (1.0 + strength * (tail_ratio - 1.0))
    return signal * gain


# ----------------------
# Optional: Demucs stems
# ----------------------

def _has_demucs() -> bool:
    return importlib.util.find_spec("demucs") is not None


def separate_stems_demucs(input_path: str, model: str = "htdemucs") -> Tuple[str, str]:
    if not _has_demucs():
        return "", "Demucs nicht installiert. Optional mit: pip install demucs torch torchaudio"
    out_root = tempfile.mkdtemp(prefix="demucs_out_")
    try:
        cmd = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            model,
            "-o",
            out_root,
            input_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return "", f"Demucs Fehler: {proc.stderr or proc.stdout}"

        stems_dir = None
        for root, dirs, files in os.walk(out_root):
            wavs = [f for f in files if f.lower().endswith(".wav")]
            if wavs:
                stems_dir = root
        if stems_dir is None:
            return "", "Keine Stems gefunden."

        zip_tmp = tempfile.NamedTemporaryFile(suffix="_stems.zip", delete=False)
        with zipfile.ZipFile(zip_tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(os.listdir(stems_dir)):
                if f.lower().endswith(".wav"):
                    zf.write(os.path.join(stems_dir, f), arcname=f)
        return zip_tmp.name, "Stems ZIP erstellt"
    except Exception as e:
        return "", f"Demucs Ausnahme: {e}"


def ui_split_stems(file):
    if not file:
        return None, "Bitte zuerst eine Datei hochladen."
    zip_path, msg = separate_stems_demucs(file)
    return zip_path, msg

# ----------------------
# Presets and parameters
# ----------------------

@dataclass
class Preset:
    name: str
    denoise_strength: float
    deesser_threshold_db: float
    deesser_ratio: float
    compressor_threshold_db: float
    compressor_ratio: float
    saturation_drive: float
    target_lufs: float


PRESETS: Dict[str, Preset] = {
    "Vocal": Preset(
        name="Vocal",
        denoise_strength=0.55,
        deesser_threshold_db=-28.0,
        deesser_ratio=5.0,
        compressor_threshold_db=-22.0,
        compressor_ratio=2.5,
        saturation_drive=1.4,
        target_lufs=-16.0,
    ),
    "Music": Preset(
        name="Music",
        denoise_strength=0.4,
        deesser_threshold_db=-24.0,
        deesser_ratio=3.0,
        compressor_threshold_db=-20.0,
        compressor_ratio=1.8,
        saturation_drive=1.25,
        target_lufs=-14.0,
    ),
    "Podcast": Preset(
        name="Podcast",
        denoise_strength=0.65,
        deesser_threshold_db=-30.0,
        deesser_ratio=6.0,
        compressor_threshold_db=-24.0,
        compressor_ratio=3.0,
        saturation_drive=1.5,
        target_lufs=-16.0,
    ),
    "Custom": Preset(
        name="Custom",
        denoise_strength=0.5,
        deesser_threshold_db=-26.0,
        deesser_ratio=4.0,
        compressor_threshold_db=-20.0,
        compressor_ratio=2.0,
        saturation_drive=1.3,
        target_lufs=-14.0,
    ),
}


# ----------------------
# Core processing
# ----------------------

def _ensure_2d(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio[np.newaxis, :]
    return audio


def process_pipeline(
    input_path: str,
    preset_name: str,
    denoise_strength: float,
    deesser_threshold_db: float,
    deesser_ratio: float,
    comp_threshold_db: float,
    comp_ratio: float,
    saturation_drive: float,
    target_lufs: float,
    dereverb_strength: float,
    output_format: str,
) -> Tuple[str, str]:
    # Load
    audio, sr = librosa.load(input_path, sr=48000, mono=False)
    audio = _ensure_2d(audio)

    # Pre-filtering and cleanup per channel
    processed: List[np.ndarray] = []
    for ch in audio:
        x = ch.astype(np.float32)
        x = remove_dc_offset(x)
        x = highpass(x, sr, 40.0)
        x = lowpass(x, sr, 19500.0)
        x = reduce_noise(x, sr, strength=denoise_strength)
        x = dereverb_simple(x, sr, strength=dereverb_strength)
        x = de_esser(x, sr, threshold_db=deesser_threshold_db, ratio=deesser_ratio)
        x = simple_compressor(x, sr, threshold_db=comp_threshold_db, ratio=comp_ratio)
        x = soft_saturate(x, drive=saturation_drive)
        processed.append(x)

    y = np.vstack(processed)

    # Loudness normalize operates on mono or stereo; librosa expects (n,) for mono. Provide (n,) or (n, channels) for measurement
    # Convert to shape (n, channels) for pyloudnorm
    y_for_lufs = y.T
    y_for_lufs = lufs_normalize(y_for_lufs, sr, target_lufs=target_lufs)
    y = y_for_lufs.T

    # Final limiter
    y = soft_limiter(y, ceiling_db=-0.1)

    # Write to temp wav or convert to mp3 for browser playback
    with tempfile.TemporaryDirectory() as td:
        wav_path = f"{td}/processed.wav"
        # soundfile expects shape (n, channels)
        sf.write(wav_path, y.T, sr, subtype="PCM_16")
        if output_format.lower() == "wav":
            final_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with open(wav_path, "rb") as fsrc, open(final_tmp.name, "wb") as fdst:
                fdst.write(fsrc.read())
            return final_tmp.name, preset_name
        else:
            mp3_path = f"{td}/processed.mp3"
            seg = AudioSegment.from_wav(wav_path)
            seg.export(mp3_path, format="mp3", bitrate="320k")
            final_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            with open(mp3_path, "rb") as fsrc, open(final_tmp.name, "wb") as fdst:
                fdst.write(fsrc.read())
            return final_tmp.name, preset_name


# ----------------------
# Gradio Interface
# ----------------------

def _apply_preset(preset_name: str,
                  denoise_strength: float,
                  deesser_threshold_db: float,
                  deesser_ratio: float,
                  comp_threshold_db: float,
                  comp_ratio: float,
                  saturation_drive: float,
                  target_lufs: float):
    if preset_name in PRESETS and preset_name != "Custom":
        p = PRESETS[preset_name]
        return (
            p.denoise_strength,
            p.deesser_threshold_db,
            p.deesser_ratio,
            p.compressor_threshold_db,
            p.compressor_ratio,
            p.saturation_drive,
            p.target_lufs,
        )
    return (
        denoise_strength,
        deesser_threshold_db,
        deesser_ratio,
        comp_threshold_db,
        comp_ratio,
        saturation_drive,
        target_lufs,
    )


# ----------------------
# Artifact analysis
# ----------------------

def analyze_artifacts(file_path: str) -> str:
    audio, sr = librosa.load(file_path, sr=48000, mono=False)
    audio = _ensure_2d(audio)
    mono = np.mean(audio, axis=0)
    eps = 1e-9

    # Clipping
    clip_ratio = float(np.mean(np.abs(mono) > 0.999))

    # DC offset
    dc = float(np.mean(mono))

    # High frequency ratios
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512)) + eps
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    def band_energy(f_lo, f_hi):
        mask = (freqs >= f_lo) & (freqs < f_hi)
        return float(np.mean(S[mask, :])) if np.any(mask) else 0.0

    e_1_5k = band_energy(1000.0, 5000.0)
    e_5_12k = band_energy(5000.0, 12000.0)
    e_12_20k = band_energy(12000.0, 20000.0)
    sibilance_idx = (e_5_12k / (e_1_5k + eps)) if e_1_5k > 0 else 0.0
    harshness_idx = (e_12_20k / (e_1_5k + eps)) if e_1_5k > 0 else 0.0

    # Noise floor proxy: 10th percentile absolute amplitude
    noise_floor = float(np.quantile(np.abs(mono), 0.10))

    # Recommendations
    rec = []
    if clip_ratio > 0.001:
        rec.append("Limiter aktiv (Clipping erkannt)")
    if abs(dc) > 1e-3:
        rec.append("DC-Offset entfernen")
    if sibilance_idx > 0.7:
        rec.append("De-Esser stärker")
    if harshness_idx > 0.5:
        rec.append("Lowpass/EQ mildern")
    if noise_floor > 0.02:
        rec.append("Denoise stärker")

    lines = [
        f"Clipping-Anteil: {clip_ratio:.4f}",
        f"DC-Offset: {dc:.6f}",
        f"Sibilance-Index (5-12k / 1-5k): {sibilance_idx:.2f}",
        f"Harshness-Index (12-20k / 1-5k): {harshness_idx:.2f}",
        f"Noise-Floor (|x| p10): {noise_floor:.3f}",
    ]
    if rec:
        lines.append("Empfehlungen: " + ", ".join(rec))
    else:
        lines.append("Keine auffälligen Artefakte erkannt.")
    return "\n".join(lines)


def ui_analyze(file):
    if not file:
        return "Bitte zuerst eine Datei hochladen."
    try:
        return analyze_artifacts(file)
    except Exception as e:
        return f"Analyse fehlgeschlagen: {e}"


def ui_process(file, preset_name, denoise_strength, deesser_threshold_db, deesser_ratio, comp_threshold_db, comp_ratio, saturation_drive, target_lufs, dereverb_strength, output_format):
    # Use preset values when not Custom
    (
        denoise_strength,
        deesser_threshold_db,
        deesser_ratio,
        comp_threshold_db,
        comp_ratio,
        saturation_drive,
        target_lufs,
    ) = _apply_preset(
        preset_name,
        denoise_strength,
        deesser_threshold_db,
        deesser_ratio,
        comp_threshold_db,
        comp_ratio,
        saturation_drive,
        target_lufs,
    )

    mp3_path, used_preset = process_pipeline(
        file,
        preset_name,
        denoise_strength,
        deesser_threshold_db,
        deesser_ratio,
        comp_threshold_db,
        comp_ratio,
        saturation_drive,
        target_lufs,
        dereverb_strength,
        output_format,
    )
    return mp3_path, f"Preset: {used_preset}"


with gr.Blocks(title="AI Music Improver") as demo:
    gr.Markdown("""
    ### AI Music Improver
    Lade dein AI-generiertes MP3 hoch und wende Humanizing-Tools an:
    - Rauschminderung, De-Esser, EQ-Filter, Kompression, Sättigung, LUFS-Normalisierung
    - Wähle ein Preset oder passe die Parameter an
    """)

    with gr.Row():
        input_audio = gr.Audio(label="Eingang (MP3/WAV)", sources=["upload"], type="filepath")
        output_audio = gr.Audio(label="Ergebnis (MP3/WAV)", type="filepath")

    with gr.Row():
        preset = gr.Dropdown(choices=list(PRESETS.keys()), value="Vocal", label="Preset")
        status = gr.Textbox(label="Status", interactive=False)
        analyze_btn = gr.Button("Analysieren")

    with gr.Accordion("Erweiterte Parameter", open=False):
        denoise_strength = gr.Slider(0.0, 1.0, value=PRESETS["Custom"].denoise_strength, step=0.01, label="Denoise Stärke")
        deesser_threshold_db = gr.Slider(-60.0, 0.0, value=PRESETS["Custom"].deesser_threshold_db, step=1.0, label="De-Esser Threshold (dB)")
        deesser_ratio = gr.Slider(1.0, 10.0, value=PRESETS["Custom"].deesser_ratio, step=0.1, label="De-Esser Ratio")
        comp_threshold_db = gr.Slider(-60.0, 0.0, value=PRESETS["Custom"].compressor_threshold_db, step=1.0, label="Compressor Threshold (dB)")
        comp_ratio = gr.Slider(1.0, 6.0, value=PRESETS["Custom"].compressor_ratio, step=0.1, label="Compressor Ratio")
        saturation_drive = gr.Slider(1.0, 3.0, value=PRESETS["Custom"].saturation_drive, step=0.05, label="Saturation Drive")
        target_lufs = gr.Slider(-30.0, -8.0, value=PRESETS["Custom"].target_lufs, step=0.5, label="Ziel-Lautheit (LUFS)")
        dereverb_strength = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="De-Reverb Stärke")
        output_format = gr.Dropdown(choices=["mp3", "wav"], value="mp3", label="Export Format")

    run_btn = gr.Button("Verarbeiten")

    with gr.Row():
        split_btn = gr.Button("Stems aufteilen (Demucs, optional)")
        stems_zip = gr.File(label="Stems ZIP", interactive=False)

    run_btn.click(
        fn=ui_process,
        inputs=[
            input_audio,
            preset,
            denoise_strength,
            deesser_threshold_db,
            deesser_ratio,
            comp_threshold_db,
            comp_ratio,
            saturation_drive,
            target_lufs,
            dereverb_strength,
            output_format,
        ],
        outputs=[output_audio, status],
    )

    analyze_btn.click(
        fn=ui_analyze,
        inputs=[input_audio],
        outputs=[status],
    )

    split_btn.click(
        fn=ui_split_stems,
        inputs=[input_audio],
        outputs=[stems_zip, status],
    )

if __name__ == "__main__":
    demo.launch()


