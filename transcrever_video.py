"""Transcreve o vídeo do WhatsApp usando Whisper, com ffmpeg do imageio-ffmpeg."""
import os
import sys

import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

# Faz o Whisper usar o ffmpeg do imageio-ffmpeg (no Windows o exe não se chama "ffmpeg")
import whisper.audio
_original_load_audio = whisper.audio.load_audio

def _load_audio(file: str, sr: int = whisper.audio.SAMPLE_RATE):
    from subprocess import run
    import numpy as np
    cmd = [
        ffmpeg_exe,
        "-nostdin", "-threads", "0",
        "-i", file,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr),
        "-"
    ]
    out = run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

whisper.audio.load_audio = _load_audio

def main():
    video = "WhatsApp Video 2026-03-05 at 22.38.19.mp4"
    if not os.path.isfile(video):
        print(f"Arquivo não encontrado: {video}")
        sys.exit(1)
    print("Carregando modelo Whisper (base)...")
    model = whisper.load_model("base")
    print("Transcrevendo (português)...")
    result = model.transcribe(video, language="pt", fp16=False)
    text = result["text"].strip()
    out_txt = "transcricao_whatsapp_video.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nTranscrição salva em: {out_txt}\n")
    print("--- Transcrição ---")
    print(text)

if __name__ == "__main__":
    main()
