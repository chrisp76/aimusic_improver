# AI Music Improver

Web-App zum Hochladen von MP3/WAV-Dateien und "Humanizing" der Audios mit:
- Rauschminderung (spectral gating)
- De-Esser (5–12 kHz Bandabhängige Dynamik)
- EQ-Filter (Low-/High-Pass)
- Kompression (einfacher RMS-Kompressor)
- Sättigung (tanh Waveshaper)
- LUFS-Normalisierung (EBU R128 Approx.)
- De-Reverb (leichtgewichtige Nachhall-Reduktion)
- Analyse & Empfehlungen (automatische Artefakt-Erkennung)

Die App ist in Python (Gradio) gebaut und läuft lokal im Browser oder in Docker.

## Voraussetzungen

- Python 3.10+ (empfohlen)
- FFmpeg installiert (für MP3-Export)
  - macOS (Homebrew):
    ```bash
    brew install ffmpeg
    ```

## Setup

```bash
cd /Users/christianpohlmann/aimusic_improver
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Starten

```bash
python app.py
```

Die App startet mit einer lokalen URL (z.B. http://127.0.0.1:7860). Öffne die URL im Browser.

## Docker

### Build & Run (Docker direkt)

```bash
cd /Users/christianpohlmann/aimusic_improver
docker build -t aimusic_improver .
docker run --rm -p 8686:8686 \
  -e GRADIO_SERVER_PORT=8686 \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  -v demucs_cache:/cache \
  aimusic_improver
```

### Docker Compose

```bash
cd /Users/christianpohlmann/aimusic_improver
docker compose up --build
```

Die App ist dann unter `http://localhost:8686` erreichbar. Der Cache (Modelle) wird persistent in `demucs_cache` gehalten.

## Nutzung

1. MP3/WAV hochladen.
2. Preset wählen (Vocal, Music, Podcast) oder "Custom" und Parameter anpassen.
3. Optional: "Analysieren" für Artefakt-Hinweise.
4. "Verarbeiten" klicken – Ergebnis als MP3 oder WAV (Dropdown) anhören/herunterladen.
5. Optional: "Stems aufteilen (Demucs)" erzeugt ein ZIP mit getrennten Spuren.

## Hinweise & Grenzen

- Der Fokus liegt auf generischen Artefakten (Hiss, Sibilanz, Schärfe, Pegel). Starke Hallentfernungen, Click/Clip-Reparatur oder präzise Instrumententrennung sind nur begrenzt möglich.
- Optional: Stems per Demucs (vocals, drums, bass, other). "Gitarre" liegt meist in "other".

## Struktur

- `app.py`: Gradio UI und Signalverarbeitung
- `requirements.txt`: Abhängigkeiten

## Optionale Features

### Demucs (Stems-Trennung)

Demucs ist optional (größere Downloads, benötigt PyTorch). Installation:

```bash
pip install demucs torch torchaudio
```

Danach in der App auf "Stems aufteilen (Demucs)" klicken – ein ZIP mit WAV-Stems wird erzeugt.
In Docker ggf. zuerst "Modelle vorladen (Demucs)" drücken (Internet im Container notwendig).

### Hinweise zu De-Reverb & Analyse

- De-Reverb ist bewusst leichtgewichtig; für starke Räume empfiehlt sich spezialisierte Dereverb-Software.
- Die Analyse zeigt Kennzahlen (Clipping, Sibilanz, Harshness, Noise-Floor) und einfache Empfehlungen.

## Fehlerbehebung

- MP3-Export schlägt fehl: FFmpeg fehlt – installiere via `brew install ffmpeg` und starte neu.
- Verzerrter Klang: Reduziere `Saturation Drive`, erhöhe Thresholds, senke Ratio.
- Zu leise/laut: Passe Ziel-Lautheit (LUFS) an.


