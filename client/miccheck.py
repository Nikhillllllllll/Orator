"""Microphone diagnostic — records 3 seconds and reports the input level.

Run:  uv run python client/miccheck.py

If the level stays near zero while you talk, the OS isn't giving us real audio
(usually: Microphone permission not granted to your terminal app, or the wrong
input device is selected). That is what makes Whisper hallucinate "thank you".
"""

import sys
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
SECONDS = 3


def main() -> None:
    try:
        dev = sd.query_devices(kind="input")
        print(f"Default input device: {dev['name']}")
    except Exception as e:
        print(f"⚠️  Could not query input device: {e}")

    print(f"\n🎙  Recording {SECONDS}s — say something now...\n")
    frames: list[np.ndarray] = []

    def cb(indata, n, t, status):
        if status:
            print(f"  (stream status: {status})")
        frames.append(indata.copy())
        rms = float(np.sqrt(np.mean((indata.astype(np.float32) / 32768.0) ** 2)))
        bar = "█" * min(40, int(rms * 400))
        print(f"\r  level |{bar:<40}| {rms:.4f}", end="", flush=True)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=cb):
        time.sleep(SECONDS)

    print("\n")
    if not frames:
        print("❌ No audio frames captured at all.")
        sys.exit(1)

    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    print(f"Overall RMS:  {rms:.4f}")
    print(f"Peak:         {peak:.4f}")
    print()

    if rms < 0.005:
        print("❌ SILENT — the OS is handing us empty audio.")
        print("   Fix: System Settings → Privacy & Security → Microphone →")
        print("        enable your terminal app (Terminal / iTerm / VS Code).")
        print("   Then fully quit & reopen that terminal and re-run this check.")
        print("   Also check the input device in System Settings → Sound → Input.")
    elif rms < 0.02:
        print("⚠️  Very quiet. Mic works but input gain is low — move closer or")
        print("   raise the input volume in System Settings → Sound → Input.")
    else:
        print("✅ Mic is capturing real audio. Dictation should work.")


if __name__ == "__main__":
    main()
