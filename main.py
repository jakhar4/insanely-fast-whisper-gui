import torch
from faster_whisper import WhisperModel
import datetime
import argparse
import os
import gradio as gr
import time
from tqdm.auto import tqdm

class TranscriptionProgress:
    def __init__(self):
        self.current_segment = 0
        self.total_segments = 0
        self.progress_text = ""
        
    def update(self, message):
        self.progress_text = message
        return self.progress_text

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(datetime.timedelta(seconds=int(seconds)))

def transcribe_audio(audio_path, model_size="large-v3", device="cuda", compute_type="float16", progress=gr.Progress()):
    """
    Transcribe audio using Insanely Fast Whisper with progress tracking
    """
    try:
        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            progress(0, desc="CUDA not available. Falling back to CPU...")
            device = "cpu"
            compute_type = "int8"

        progress(0.1, desc="Loading model...")
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        
        progress(0.2, desc="Starting transcription...")
        
        # First pass to get segments for progress tracking
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Convert generator to list for counting
        segments_list = list(segments)
        total_segments = len(segments_list)
        
        # Process segments with progress tracking
        transcription = []
        for idx, segment in enumerate(segments_list, 1):
            timestamp = format_timestamp(segment.start)
            text = segment.text.strip()
            formatted_segment = f"({timestamp}) {text}"
            transcription.append(formatted_segment)
            
            # Update progress
            progress((idx / total_segments) * 0.8 + 0.2, 
                    desc=f"Processing segment {idx}/{total_segments}")
            
        progress(1.0, desc="Transcription completed!")
        return "\n".join(transcription)

    except Exception as e:
        return f"An error occurred: {str(e)}"

def save_transcription(text, output_path):
    """Save transcription to a text file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return f"Transcription saved to: {output_path}"
    except Exception as e:
        return f"Error saving transcription: {str(e)}"

def create_gui():
    """Create Gradio interface"""
    with gr.Blocks(title="Audio Transcription with Insanely Fast Whisper") as interface:
        gr.Markdown("# Audio Transcription with Insanely Fast Whisper")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Upload Audio",
                    type="filepath"
                )
                
                with gr.Row():
                    model_size = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        value="large-v3",
                        label="Model Size"
                    )
                    device = gr.Dropdown(
                        choices=["cuda", "cpu"],
                        value="cuda" if torch.cuda.is_available() else "cpu",
                        label="Device"
                    )
                    compute_type = gr.Dropdown(
                        choices=["float16", "int8"],
                        value="float16",
                        label="Compute Type"
                    )
                
                transcribe_btn = gr.Button("Transcribe")
                save_btn = gr.Button("Save Transcription")
                output_path = gr.Textbox(
                    label="Save Path (optional)",
                    placeholder="Enter path to save transcription (e.g., output.txt)"
                )
                
            with gr.Column():
                output_text = gr.TextArea(
                    label="Transcription Output",
                    interactive=False,
                    lines=20
                )
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        # Set up event handlers
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[
                audio_input,
                model_size,
                device,
                compute_type
            ],
            outputs=output_text
        )
        
        save_btn.click(
            fn=save_transcription,
            inputs=[
                output_text,
                output_path
            ],
            outputs=status_text
        )
        
        # Example usage
        gr.Markdown("""
        ## Instructions
        1. Upload an audio file
        2. Select model settings (or use defaults)
        3. Click 'Transcribe' to start transcription
        4. Optionally save the transcription to a file
        
        ## Notes
        - Large models provide better accuracy but require more memory
        - CUDA (GPU) processing is faster when available
        - Progress bar shows transcription status
        """)
    
    return interface

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Insanely Fast Whisper")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    parser.add_argument("--audio_path", help="Path to the audio file")
    parser.add_argument("--model", default="large-v3", help="Model size (default: large-v3)")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu, default: cuda)")
    parser.add_argument("--compute_type", default="float16", help="Compute type (default: float16)")
    parser.add_argument("--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    if args.gui:
        # Launch Gradio interface
        interface = create_gui()
        # Launch with sharing enabled and other configuration options
        interface.launch(
            share=True,  # Enables public URL
            server_name="0.0.0.0",  # Allows external connections
            server_port=7860,  # Default Gradio port
            inbrowser=True  # Opens interface in browser automatically
        )
    else:
        # Command-line mode
        if not args.audio_path:
            print("Error: Please provide an audio path or use --gui for the graphical interface")
            return
            
        if not os.path.exists(args.audio_path):
            print(f"Error: Audio file not found at {args.audio_path}")
            return
        
        # Perform transcription
        transcription = transcribe_audio(
            args.audio_path,
            args.model,
            args.device,
            args.compute_type
        )
        
        # Save transcription if output path is provided
        if transcription and args.output:
            save_transcription(transcription, args.output)

if __name__ == "__main__":
    main()
