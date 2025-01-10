# Insanely Fast Whisper Transcription Tool

A powerful audio transcription tool that combines the speed of Faster Whisper with an easy-to-use Gradio interface. This tool provides both command-line and GUI options for transcribing audio files with timestamped output.

## Features

- 🎯 High-accuracy transcription using Faster Whisper
- 🖥️ Interactive GUI interface with Gradio
- 📊 Real-time progress tracking
- ⏱️ Timestamp generation for each segment
- 🌐 Public URL sharing option
- 💻 GPU support with CPU fallback
- 💾 Save transcriptions to file
- 🔧 Configurable model settings

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (optional, but recommended for faster processing)

### Setup

1. Clone this repository:
```bash
git clone [repository-url]
cd whisper-transcription-tool
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install required packages:
```bash
pip install torch faster-whisper gradio tqdm
```

## Usage

### GUI Mode

1. Launch the graphical interface:
```bash
python script.py --gui
```

This will:
- Open the interface in your default browser
- Generate a public URL for sharing (e.g., https://xxxx.gradio.live)
- Display both local and public URLs in the terminal

### Command Line Mode

Basic usage:
```bash
python script.py --audio_path path/to/audio.mp3 --output transcript.txt
```

Available command-line options:
```bash
--gui            Launch GUI interface
--audio_path     Path to the audio file
--model          Model size (default: large-v3)
--device         Device to use (cuda/cpu, default: cuda)
--compute_type   Compute type (default: float16)
--output         Output file path (optional)
```

## Model Options

### Model Sizes
- `tiny`: Fastest, lowest accuracy
- `base`: Fast, basic accuracy
- `small`: Balanced speed/accuracy
- `medium`: Good accuracy, moderate speed
- `large-v3`: Best accuracy, slowest speed

### Compute Types
- `float16`: Default, good for GPU
- `int8`: Better for CPU, lower memory usage

### Devices
- `cuda`: GPU processing (recommended)
- `cpu`: CPU processing (fallback)

## Output Format

The transcription output includes timestamps in HH:MM:SS format:
```
(00:00:00) First segment of transcription
(00:00:05) Second segment of transcription
...
```

## GUI Interface Features

1. **Input Section**
   - Audio file upload area
   - Model size selection
   - Device selection
   - Compute type selection

2. **Control Section**
   - Transcribe button
   - Save button
   - Output path setting

3. **Output Section**
   - Transcription display
   - Status messages
   - Progress tracking

4. **Progress Tracking**
   - Model loading status
   - Segment processing progress
   - Overall completion percentage

## Performance Tips

1. **For Faster Processing**
   - Use a CUDA-compatible GPU
   - Choose smaller model sizes (tiny/base) for speed
   - Use float16 compute type with GPU

2. **For Better Accuracy**
   - Use larger model sizes (large-v3)
   - Ensure clean audio input
   - Use GPU for processing

3. **For Limited Resources**
   - Use CPU with int8 compute type
   - Choose smaller model sizes
   - Process shorter audio files

## Common Issues and Solutions

1. **CUDA Out of Memory**
   - Try a smaller model size
   - Switch to int8 compute type
   - Free up GPU memory

2. **Slow Processing**
   - Verify GPU is being used
   - Check model size selection
   - Consider input audio length

3. **Connection Issues**
   - Check internet connection for shared URLs
   - Verify port availability
   - Check firewall settings

## Acknowledgments

This project uses the following major components:
- Faster Whisper
- Gradio
- PyTorch
