import os
import zipfile
import subprocess
import gradio as gr
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio
import torchaudio
import torch
from pathlib import Path
import warnings

# ===== FIX 1: Suppress all warnings =====
warnings.filterwarnings("ignore")

# --- Configuration ---
BASE_DIR = Path(__file__).parent.absolute()

# ===== FIX 2: Windows path conversion =====
def win_path(path):
    """Convert Path objects to Windows-style strings"""
    return str(path).replace('/', '\\')

# ===== FIX 3: Environment variables with Windows paths =====
os.environ.update({
    "TORTOISE_MODELS_DIR": win_path(BASE_DIR / "tortoise_models"),
    "TORTOISE_DISABLE_RVQ": "1",  # Disable rvq.pt requirement
    "TORTOISE_DISABLE_CLVP": "1",  # Disable clvp2.pth requirement
    "TORTOISE_CACHE_DIR": win_path(BASE_DIR / "tmp")  # Explicit cache dir
})

# ===== FIX 4: Directory creation with exist_ok =====
os.makedirs(os.environ["TORTOISE_MODELS_DIR"], exist_ok=True)
os.makedirs(os.environ["TORTOISE_CACHE_DIR"], exist_ok=True)

# ===== FIX 5: All paths converted to Windows format =====
ZIP_FILE = win_path(BASE_DIR / "Trump-LipSync.zip")
EXTRACTED_FOLDER = win_path(BASE_DIR / "Trump-LipSync")
VOICE_DIR = win_path(BASE_DIR / "trump_voices")
TRUMP_VIDEO_PATH = win_path(EXTRACTED_FOLDER) + "\\videos\\trump.mp4"
CHECKPOINT_PATH = win_path(EXTRACTED_FOLDER) + "\\Wav2Lip\\checkpoints\\wav2lip_gan.pth"
OUTPUT_WAV = win_path(BASE_DIR / "generated_speech.wav")
OUTPUT_VIDEO = win_path(BASE_DIR / "lip_synced_output.mp4")

# --- Helper Functions ---
def setup_environment():
    """Ensure all required files exist"""
    if not os.path.exists(VOICE_DIR):
        os.makedirs(VOICE_DIR)

    print("\n=== FILE VERIFICATION ===")
    # ===== FIX 6: Using os.listdir() instead of Path.glob() =====
    print(f"Models: {os.listdir(os.environ['TORTOISE_MODELS_DIR'])}")
    print(f"Voice samples: {os.listdir(VOICE_DIR)}")
    print(f"Video exists: {os.path.exists(TRUMP_VIDEO_PATH)}")
    print(f"Checkpoint exists: {os.path.exists(CHECKPOINT_PATH)}")

    if not os.path.exists(EXTRACTED_FOLDER):
        if not os.path.exists(ZIP_FILE):
            raise FileNotFoundError("Missing ZIP file")
        print("Extracting assets...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTED_FOLDER)

    # ===== FIX 7: Voice sample validation =====
    if len(os.listdir(VOICE_DIR)) < 3:
        raise FileNotFoundError("Need 3+ voice samples (trump_sample_1.wav, etc.)")

# --- Core Functions ---
def generate_speech(text):
    """Generate speech with enhanced error handling"""
    try:
        print("\n=== TTS INITIALIZATION ===")
        
        # ===== FIX 8: Explicit TTS initialization =====
        tts = TextToSpeech()
        
        # Load first 3 voice samples
        voice_samples = []
        for i in range(1, 4):
            sample_path = os.path.join(VOICE_DIR, f"trump_sample_{i}.wav")
            if not os.path.exists(sample_path):
                raise FileNotFoundError(f"Missing sample: {sample_path}")
            voice_samples.append(load_audio(sample_path, 22050))
        
        # ===== FIX 9: Force 'fast' preset for reliability =====
        gen = tts.tts_with_preset(
            text,
            voice_samples=voice_samples,
            preset="fast",
            conditioning_latents=None
        )
        
        torchaudio.save(
            OUTPUT_WAV,
            gen.squeeze(0).cpu(),
            22050,
            format="wav"
        )
        return True, OUTPUT_WAV
    except Exception as e:
        print(f"\n!!! TTS FAILURE !!!")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        print(f"Current directory: {os.getcwd()}")
        return False, f"TTS failed: {str(e)}"

def run_wav2lip():
    """Run Wav2Lip with path fixes"""
    try:
        # ===== FIX 10: Subprocess with explicit paths =====
        subprocess.run([
            "python",
            os.path.join(EXTRACTED_FOLDER, "Wav2Lip", "inference.py"),
            "--checkpoint_path", CHECKPOINT_PATH,
            "--face", TRUMP_VIDEO_PATH,
            "--audio", OUTPUT_WAV,
            "--outfile", OUTPUT_VIDEO,
            "--fps", "25",
            "--pads", "0", "10", "0", "0"
        ], check=True)
        return True, OUTPUT_VIDEO
    except subprocess.CalledProcessError as e:
        return False, f"Wav2Lip error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

# --- Main Processing ---
def process_lipsync(text):
    """Full pipeline with cleanup"""
    try:
        # Clean previous outputs
        for f in [OUTPUT_WAV, OUTPUT_VIDEO]:
            if os.path.exists(f):
                os.remove(f)
        
        # Generate speech
        success, msg = generate_speech(text)
        if not success:
            return msg, None
        
        # Run Wav2Lip
        success, output_path = run_wav2lip()
        return ("Success!", output_path) if success else (msg, None)
    except Exception as e:
        return f"Processing failed: {str(e)}", None

# --- Gradio Interface ---
def main():
    print("\n" + "="*40)
    print("=== TRUMP VOICE CLONING SYSTEM ===")
    print("="*40 + "\n")
    
    try:
        setup_environment()
        print("✅ Setup completed successfully")
    except Exception as e:
        print(f"\n❌ SETUP FAILED: {str(e)}")
        print("Please verify:")
        print("- 3+ voice samples in trump_voices/")
        print("- Trump-LipSync.zip exists")
        print("- Enough disk space")
        return

    iface = gr.Interface(
        fn=process_lipsync,
        inputs=gr.Textbox(
            label="Enter Text",
            placeholder="What would Trump say? (Keep under 20 words)",
            lines=3
        ),
        outputs=[
            gr.Textbox(label="Status"),
            gr.Video(label="Lip-Synced Video")
        ],
        title="Trump Voice Cloning & Lip Sync",
        examples=[
            ["The wall just got 10 feet higher!"],
            ["Nobody knows more about technology than me"],
            ["We're gonna win so much, you'll get tired of winning"]
        ],
        allow_flagging="never"
    )
    
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main()