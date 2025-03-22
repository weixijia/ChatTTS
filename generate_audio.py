import argparse
import time
import logging
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from ChatTTS.core import Chat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_chattts_model(device='auto'):
    """Initialize and load ChatTTS model with optimized settings"""
    try:
        chat = Chat(logger=logger)
        device = select_device(device)
        logger.info(f"Loading model on device: {device}")
        
        chat.load(
            source='local',
            device=device,
            compile=True,
            use_flash_attn=True
        )
        return chat
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

def select_device(device):
    """Automatically select available device"""
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device

def extract_voice_features(model, wav_path, target_sr=24000):
    """Extract voice features from audio file with preprocessing"""
    try:
        from librosa import load, resample
    except ImportError:
        raise ImportError("librosa is required for audio processing. Install with 'pip install librosa'")
    
    # Load and preprocess audio
    wav, sr = load(wav_path, sr=target_sr)
    if len(wav) == 0:
        raise ValueError("Empty audio file")
        
    # Convert to tensor and extract features
    audio_tensor = torch.FloatTensor(wav).unsqueeze(0)
    spk_emb = model.sample_audio_speaker(audio_tensor)
    return spk_emb

def generate_speech(model, text, spk_emb=None, **kwargs):
    """Generate speech with enhanced parameters and error handling"""
    params = model.InferCodeParams(
        spk_emb=spk_emb,
        temperature=kwargs.get('temperature', 0.3),
        top_P=kwargs.get('top_p', 0.7),
        top_K=kwargs.get('top_k', 20),
        repetition_penalty=kwargs.get('repetition_penalty', 1.05),
        prompt=f"[speed_{kwargs.get('speed',5)}][pitch_{kwargs.get('pitch',1.0)}]",
        stream_speed=kwargs.get('stream_speed', 12000)
    )
    
    try:
        texts = split_text(text, max_length=100)
        wavs = []
        
        for batch in batch_texts(texts, batch_size=4):
            wav = model.infer(
                batch,
                params_infer_code=params,
                use_decoder=True
            )
            wavs.extend(wav)
            
        return np.concatenate(wavs)
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise

def split_text(text, max_length=100):
    """Smart text splitting with punctuation awareness"""
    sentences = []
    current = []
    length = 0
    
    for char in text:
        current.append(char)
        length += 1
        if char in {'。', '！', '？', '.', '!', '?'} and length >= max_length//2:
            sentences.append(''.join(current))
            current = []
            length = 0
        elif length >= max_length:
            sentences.append(''.join(current))
            current = []
            length = 0
            
    if current:
        sentences.append(''.join(current))
    return sentences

def batch_texts(texts, batch_size=4):
    """Create batches with dynamic size based on text length"""
    batches = []
    current_batch = []
    current_length = 0
    
    for text in texts:
        text_length = len(text)
        if current_length + text_length > 500:  # Max characters per batch
            batches.append(current_batch)
            current_batch = []
            current_length = 0
        current_batch.append(text)
        current_length += text_length
    
    if current_batch:
        batches.append(current_batch)
    return batches

def main():
    parser = argparse.ArgumentParser(
        description='ChatTTS Voice Generation Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--voice', type=str, default='weixijia_audio.wav',
                      help='Path to reference voice WAV file')
    parser.add_argument('--text', type=str, default='Hello, world!',
                      help='Text to generate (enclose in quotes)')
    parser.add_argument('--output', type=str, default='output',
                      help='Output directory')
    parser.add_argument('--temperature', type=float, default=0.3,
                      help='Control randomness (0.1-1.0)', metavar='[0.1-1.0]')
    parser.add_argument('--top_p', type=float, default=0.7,
                      help='Nucleus sampling (0.1-1.0)', metavar='[0.1-1.0]')
    parser.add_argument('--speed', type=int, default=5,
                      help='Speaking speed (1-10)', metavar='[1-10]')
    parser.add_argument('--pitch', type=float, default=1.0,
                      help='Pitch adjustment (0.5-2.0)', metavar='[0.5-2.0]')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cpu', 'cuda', 'mps'],
                      help='Computation device')

    args = parser.parse_args()
    
    # Validate parameters
    if not (0.1 <= args.temperature <= 1.0):
        raise ValueError("Temperature must be between 0.1 and 1.0")
    if not (1 <= args.speed <= 10):
        raise ValueError("Speed must be between 1 and 10")
    if not (0.5 <= args.pitch <= 2.0):
        raise ValueError("Pitch must be between 0.5 and 2.0")
    
    try:
        # Initialize model
        logger.info("Loading ChatTTS model...")
        tts = load_chattts_model(args.device)
        
        # Process voice sample
        spk_emb = None
        if args.voice:
            logger.info(f"Extracting features from {args.voice}...")
            spk_emb = extract_voice_features(tts, args.voice)
        
        # Generate speech
        logger.info("Generating audio...")
        start_time = time.time()
        audio = generate_speech(
            tts,
            args.text,
            spk_emb=spk_emb,
            temperature=args.temperature,
            top_p=args.top_p,
            speed=args.speed,
            pitch=args.pitch
        )
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)
        timestamp = int(time.time())
        
        # Save audio
        audio_path = output_dir / f"output_{timestamp}.wav"
        sf.write(str(audio_path), audio, 24000)
        logger.info(f"Audio saved to: {audio_path}")
        
        # Save voice features if extracted
        if spk_emb is not None:
            emb_path = output_dir / f"voice_features_{timestamp}.npy"
            np.save(str(emb_path), spk_emb)
            logger.info(f"Voice features saved to: {emb_path}")
        
        logger.info(f"Generation completed in {time.time()-start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'tts' in locals():
            tts.unload()
            del tts

if __name__ == "__main__":
    main()