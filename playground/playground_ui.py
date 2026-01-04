"""
Playground UI
Gradio-based UI for the ACE-Step Playground.
"""
import os
import sys
import gradio as gr
from typing import Optional, List

# Add project root to sys.path
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Studio Bridge JavaScript code (for frontend API calls)
from studio_bridge import (
    STUDIO_BRIDGE_JS,
    JS_CONNECT_STUDIO,
    JS_GET_AUDIO_FROM_STUDIO,
    JS_SEND_AUDIO_TO_STUDIO,
)


# =============================================================================
# Dynamic UI Configuration
# =============================================================================

TASK_VISIBILITY = {
    "generate": {
        "audio_codes": True,
        "reference_audio": True,
        "source_audio": False,
        "repaint_params": False,
        "cover_params": False,
        "track_params": False,
    },
    "repaint": {
        "audio_codes": False,
        "reference_audio": True,
        "source_audio": True,
        "repaint_params": True,
        "cover_params": False,
        "track_params": False,
    },
    "cover": {
        "audio_codes": False,
        "reference_audio": True,
        "source_audio": True,
        "repaint_params": False,
        "cover_params": True,
        "track_params": False,
    },
    "add": {
        "audio_codes": False,
        "reference_audio": True,
        "source_audio": True,
        "repaint_params": False,
        "cover_params": False,
        "track_params": True,
    },
    "complete": {
        "audio_codes": False,
        "reference_audio": True,
        "source_audio": True,
        "repaint_params": False,
        "cover_params": False,
        "track_params": True,
    },
    "extract": {
        "audio_codes": False,
        "reference_audio": True,
        "source_audio": True,
        "repaint_params": False,
        "cover_params": False,
        "track_params": False,
    },
}


def update_task_visibility(task: str):
    """Update visibility of task-specific components based on selected task."""
    vis = TASK_VISIBILITY.get(task, TASK_VISIBILITY["generate"])
    return (
        gr.update(visible=vis["audio_codes"]),
        gr.update(visible=vis["reference_audio"]),
        gr.update(visible=vis["source_audio"]),
        gr.update(visible=vis["repaint_params"]),
        gr.update(visible=vis["cover_params"]),
        gr.update(visible=vis["track_params"]),
    )


# =============================================================================
# UI Creation
# =============================================================================

def create_ui(handler):
    """
    Create the Gradio UI for the Playground.
    
    Args:
        handler: An instance of PlaygroundHandler
    """
    
    # Get available models
    available_llm_models = handler.get_available_llm_models()
    available_dit_models = handler.get_available_dit_models()
    
    with gr.Blocks(
        title="ACE-Step Playground",
        theme=gr.themes.Soft(),
        js=STUDIO_BRIDGE_JS  # Inject Studio Bridge JS into the page
    ) as demo:
        gr.Markdown("# 🎵 ACE-Step Playground")
        gr.Markdown("Generate music using LLM for audio codes and DiT for audio synthesis.")
        
        # =================================================================
        # Studio Integration (Top Section)
        # =================================================================
        with gr.Accordion("🔗 Studio Integration", open=False):
            gr.Markdown("Connect to ACE Studio for audio exchange")
            with gr.Row():
                studio_token = gr.Textbox(
                    label="Studio Token",
                    placeholder="Enter token from ACE Studio WebView...",
                    type="password",
                    scale=3
                )
                connect_studio_btn = gr.Button(
                    "🔗 Connect",
                    variant="primary",
                    scale=1
                )
            studio_connection_status = gr.Textbox(
                label="Connection Status",
                interactive=False,
                placeholder="Not connected"
            )
        
        with gr.Tabs():
            # =================================================================
            # LLM Section
            # =================================================================
            with gr.TabItem("🤖 LLM Section"):
                gr.Markdown("### Generate audio codes from text descriptions")
                
                # ---------------------------------------------------------
                # 1. Model Loading
                # ---------------------------------------------------------
                with gr.Accordion("1. Model Loading", open=True):
                    with gr.Row():
                        llm_model_dropdown = gr.Dropdown(
                            choices=available_llm_models,
                            value=available_llm_models[0] if available_llm_models else None,
                            label="LLM Model",
                            allow_custom_value=True,
                            scale=3
                        )
                        llm_backend = gr.Dropdown(
                            choices=["vllm", "pt"],
                            value="vllm",
                            label="Backend",
                            scale=1
                        )
                        llm_device = gr.Dropdown(
                            choices=["auto", "cuda", "cpu"],
                            value="auto",
                            label="Device",
                            scale=1
                        )
                        load_llm_btn = gr.Button("Load LLM", variant="primary", scale=1)
                    
                    llm_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        placeholder="LLM not loaded"
                    )
                
                # ---------------------------------------------------------
                # 2. Inputs
                # ---------------------------------------------------------
                with gr.Accordion("2. Inputs", open=True):
                    with gr.Row():
                        # Left Column - Text Inputs
                        with gr.Column(scale=1):
                            caption = gr.Textbox(
                                label="Caption",
                                placeholder="Describe the music style, mood, instruments...",
                                lines=3
                            )
                            lyrics = gr.Textbox(
                                label="Lyrics",
                                placeholder="Enter song lyrics (use [verse], [chorus] tags for structure)...",
                                lines=5
                            )
                            negative_caption = gr.Textbox(
                                label="Negative Caption",
                                value="NO USER INPUT",
                                lines=3
                            )
                            negative_lyrics = gr.Textbox(
                                label="Negative Lyrics",
                                value="NO USER INPUT",
                                lines=5
                            )
                        
                        # Right Column - Meta & Config
                        with gr.Column(scale=1):
                            # Meta Group
                            with gr.Group():
                                gr.Markdown("#### Meta")
                                with gr.Row():
                                    bpm = gr.Number(
                                        label="BPM",
                                        value=None,
                                        precision=0
                                    )
                                    target_duration = gr.Number(
                                        label="Target Duration (s)",
                                        value=None,
                                        precision=0
                                    )
                                with gr.Row():
                                    key_scale = gr.Textbox(
                                        label="Key Scale",
                                        placeholder="e.g., C Major"
                                    )
                                    time_signature = gr.Textbox(
                                        label="Time Signature",
                                        placeholder="e.g., 4/4"
                                    )
                            
                            # Config Group - Accordion (collapsed by default)
                            with gr.Accordion("Config", open=False):
                                with gr.Row():
                                    temperature = gr.Slider(
                                        label="Temperature",
                                        minimum=0.1,
                                        maximum=2.0,
                                        value=0.85,
                                        step=0.05
                                    )
                                    cfg_scale = gr.Slider(
                                        label="CFG Scale",
                                        minimum=1.0,
                                        maximum=5.0,
                                        value=2.0,
                                        step=0.1
                                    )
                                with gr.Row():
                                    top_k = gr.Number(
                                        label="Top K",
                                        value=None,
                                        precision=0
                                    )
                                    top_p = gr.Slider(
                                        label="Top P",
                                        minimum=0.0,
                                        maximum=1.0,
                                        value=0.9,
                                        step=0.05
                                    )
                                repetition_penalty = gr.Slider(
                                    label="Repetition Penalty",
                                    minimum=1.0,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.05
                                )
                                with gr.Row():
                                    metadata_temperature = gr.Slider(
                                        label="Metadata Temperature",
                                        minimum=0.1,
                                        maximum=2.0,
                                        value=0.85,
                                        step=0.05
                                    )
                                    codes_temperature = gr.Slider(
                                        label="Codes Temperature",
                                        minimum=0.1,
                                        maximum=2.0,
                                        value=1.0,
                                        step=0.05
                                    )
                    
                    generate_codes_btn = gr.Button("🎼 Generate Codes", variant="primary", size="lg")
                
                # ---------------------------------------------------------
                # 3. Results
                # ---------------------------------------------------------
                with gr.Accordion("3. Results", open=True):
                    with gr.Row():
                        metadata_output = gr.JSON(
                            label="Generated Metadata",
                            scale=1
                        )
                        audio_codes_output = gr.Textbox(
                            label="Audio Codes",
                            lines=8,
                            scale=2
                        )
                    llm_generation_status = gr.Textbox(
                        label="Generation Status",
                        interactive=False
                    )
            
            # =================================================================
            # ACEStep Section
            # =================================================================
            with gr.TabItem("🎹 ACEStep Section"):
                gr.Markdown("### Generate audio from codes using DiT model")
                
                # ---------------------------------------------------------
                # 1. Model Loading
                # ---------------------------------------------------------
                with gr.Accordion("1. Model Loading", open=True):
                    with gr.Row():
                        dit_config_dropdown = gr.Dropdown(
                            choices=available_dit_models,
                            value=available_dit_models[0] if available_dit_models else None,
                            label="DiT Model Config",
                            allow_custom_value=True,
                            scale=3
                        )
                        dit_device = gr.Dropdown(
                            choices=["auto", "cuda", "cpu"],
                            value="auto",
                            label="Device",
                            scale=1
                        )
                        load_dit_btn = gr.Button("Load DiT", variant="primary", scale=1)
                    
                    dit_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        placeholder="DiT model not loaded"
                    )
                
                # ---------------------------------------------------------
                # 2. Task & Conditions
                # ---------------------------------------------------------
                with gr.Accordion("2. Task & Conditions", open=True):
                    # Task Selection
                    task_type = gr.Dropdown(
                        choices=["generate", "repaint", "cover", "add", "complete", "extract"],
                        value="generate",
                        label="Task Type"
                    )
                    


                    with gr.Row():
                        # Left Column - Common Inputs (always visible)
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("#### Common Inputs")
                                ace_caption = gr.Textbox(
                                    label="Caption",
                                    placeholder="Copy from LLM section or enter new...",
                                    lines=2
                                )
                                ace_lyrics = gr.Textbox(
                                    label="Lyrics",
                                    placeholder="Copy from LLM section or enter new...",
                                    lines=3
                                )
                            
                            # Meta
                            with gr.Group():
                                gr.Markdown("#### Meta")
                                with gr.Row():
                                    ace_bpm = gr.Number(
                                        label="BPM",
                                        value=None,
                                        precision=0
                                    )
                                    ace_target_duration = gr.Number(
                                        label="Target Duration (s)",
                                        value=None,
                                        precision=0
                                    )
                                with gr.Row():
                                    ace_key_scale = gr.Textbox(
                                        label="Key Scale",
                                        placeholder="e.g., C Major"
                                    )
                                    ace_time_signature = gr.Textbox(
                                        label="Time Signature",
                                        placeholder="e.g., 4/4"
                                    )
                            
                            # Logical Conditions
                            with gr.Group():
                                gr.Markdown("#### Logical Conditions")
                                inference_steps = gr.Slider(
                                    label="Inference Steps",
                                    minimum=1,
                                    maximum=100,
                                    value=20,
                                    step=1
                                )
                                guidance_scale = gr.Slider(
                                    label="Guidance Scale",
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=7.0,
                                    step=0.5
                                )
                                with gr.Row():
                                    seed = gr.Number(
                                        label="Seed (-1 for random)",
                                        value=-1,
                                        precision=0
                                    )
                                    use_random_seed = gr.Checkbox(
                                        label="Use Random Seed",
                                        value=True
                                    )
                                vocal_language = gr.Dropdown(
                                    choices=["en", "zh", "ja", "ko"],
                                    value="en",
                                    label="Vocal Language"
                                )
                        
                        # Right Column - Dynamic Task-specific Inputs
                        with gr.Column(scale=1):
                            # Audio Codes Group (only visible for generate task)
                            with gr.Group(visible=True) as audio_codes_group:
                                gr.Markdown("#### Audio Codes")
                                ace_audio_codes = gr.Textbox(
                                    label="Audio Codes",
                                    placeholder="Paste audio codes from LLM section...",
                                    lines=3
                                )
                            
                            # Reference Audio Group (visible for all tasks)
                            with gr.Group(visible=True) as reference_audio_group:
                                gr.Markdown("#### Reference Audio")
                                reference_audio = gr.Audio(
                                    label="Reference Audio (for style guidance)",
                                    type="filepath"
                                )
                                get_ref_from_studio_btn = gr.Button(
                                    "📥 Get from Studio",
                                    variant="secondary"
                                )
                            
                            # Source Audio Group (for repaint, cover, add, complete, extract)
                            with gr.Group(visible=False) as source_audio_group:
                                gr.Markdown("#### Source Audio")
                                source_audio = gr.Audio(
                                    label="Source Audio (to be processed)",
                                    type="filepath"
                                )
                                get_src_from_studio_btn = gr.Button(
                                    "📥 Get from Studio",
                                    variant="secondary"
                                )
                            
                            # Repaint Parameters Group (dynamic visibility)
                            with gr.Group(visible=False) as repaint_params_group:
                                gr.Markdown("#### Repaint Parameters")
                                with gr.Row():
                                    repainting_start = gr.Number(
                                        label="Start (s)",
                                        value=0.0,
                                        precision=2
                                    )
                                    repainting_end = gr.Number(
                                        label="End (s)",
                                        value=10.0,
                                        precision=2
                                    )
                            
                            # Cover Parameters Group (dynamic visibility)
                            with gr.Group(visible=False) as cover_params_group:
                                gr.Markdown("#### Cover Parameters")
                                audio_cover_strength = gr.Slider(
                                    label="Audio Cover Strength",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=1.0,
                                    step=0.05
                                )
                            
                            # Track Parameters Group (dynamic visibility)
                            with gr.Group(visible=False) as track_params_group:
                                gr.Markdown("#### Track Parameters")
                                track_type = gr.Dropdown(
                                    choices=["vocal", "bass", "drums", "guitar", "piano", "other"],
                                    value="vocal",
                                    label="Track Type"
                                )
                            
                            # Advanced Settings (Accordion - collapsed by default)
                            with gr.Accordion("Advanced Settings", open=False):
                                with gr.Row():
                                    cfg_interval_start = gr.Slider(
                                        label="CFG Interval Start",
                                        minimum=0.0,
                                        maximum=1.0,
                                        value=0.0,
                                        step=0.05
                                    )
                                    cfg_interval_end = gr.Slider(
                                        label="CFG Interval End",
                                        minimum=0.0,
                                        maximum=1.0,
                                        value=1.0,
                                        step=0.05
                                    )
                                with gr.Row():
                                    use_adg = gr.Checkbox(
                                        label="Use ADG",
                                        value=False
                                    )
                                    use_tiled_decode = gr.Checkbox(
                                        label="Use Tiled Decode",
                                        value=True
                                    )
                                with gr.Row():
                                    audio_format = gr.Dropdown(
                                        choices=["mp3", "wav", "flac"],
                                        value="mp3",
                                        label="Audio Format"
                                    )
                    
                    generate_audio_btn = gr.Button("🎵 Generate Audio", variant="primary", size="lg")
                
                # ---------------------------------------------------------
                # 3. Results
                # ---------------------------------------------------------
                with gr.Accordion("3. Results", open=True):
                    audio_output_1 = gr.Audio(
                        label="Generated Audio 1",
                        type="filepath"
                    )
                    send_audio1_to_studio_btn = gr.Button(
                        "📤 Send Audio 1 to Studio",
                        variant="secondary"
                    )
                    audio_output_2 = gr.Audio(
                        label="Generated Audio 2",
                        type="filepath"
                    )
                    send_audio2_to_studio_btn = gr.Button(
                        "📤 Send Audio 2 to Studio",
                        variant="secondary"
                    )
                    actual_texts = gr.Textbox(
                        label="Actual Text Input",
                        interactive=False
                    )
                    audio_generation_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )

        
        # =================================================================
        # Event Handlers
        # =================================================================
        
        # -----------------------------------------------------------------
        # LLM Section Events
        # -----------------------------------------------------------------
        
        load_llm_btn.click(
            fn=handler.initialize_llm,
            inputs=[llm_model_dropdown, llm_backend, llm_device],
            outputs=[llm_status]
        )
        
        def generate_codes_wrapper(
            caption, lyrics, negative_caption, negative_lyrics,
            bpm, key_scale, time_signature, target_duration,
            temperature, cfg_scale, top_k, top_p,
            repetition_penalty, metadata_temperature, codes_temperature
        ):
            """Wrapper function to prepare inputs and call handler."""
            # Combine negative caption and lyrics (use caption as primary negative prompt)
            negative_prompt = negative_caption if negative_caption else "NO USER INPUT"
            
            # Prepare user metadata
            user_metadata = {}
            if bpm is not None:
                user_metadata["bpm"] = str(int(bpm))
            if key_scale:
                user_metadata["keyscale"] = key_scale
            if time_signature:
                user_metadata["timesignature"] = time_signature
            
            return handler.generate_llm_codes(
                caption=caption,
                lyrics=lyrics,
                temperature=temperature,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                top_k=int(top_k) if top_k is not None else None,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                metadata_temperature=metadata_temperature,
                codes_temperature=codes_temperature,
                target_duration=target_duration,
                user_metadata=user_metadata if user_metadata else None
            )
        
        generate_codes_btn.click(
            fn=generate_codes_wrapper,
            inputs=[
                caption, lyrics, negative_caption, negative_lyrics,
                bpm, key_scale, time_signature, target_duration,
                temperature, cfg_scale, top_k, top_p,
                repetition_penalty, metadata_temperature, codes_temperature
            ],
            outputs=[metadata_output, audio_codes_output, llm_generation_status]
        )
        
        # -----------------------------------------------------------------
        # ACEStep Section Events
        # -----------------------------------------------------------------
        
        load_dit_btn.click(
            fn=handler.initialize_dit,
            inputs=[dit_config_dropdown, dit_device],
            outputs=[dit_status]
        )
        
        # Dynamic visibility based on task type
        task_type.change(
            fn=update_task_visibility,
            inputs=[task_type],
            outputs=[
                audio_codes_group,
                reference_audio_group,
                source_audio_group,
                repaint_params_group,
                cover_params_group,
                track_params_group,
            ]
        )
        
        def generate_audio_wrapper(
            task, caption, lyrics, codes,
            steps, guidance, seed_val, random_seed,
            ref_audio, src_audio, repaint_start, repaint_end, cover_strength,
            track_type_val,
            bpm_val, key_val, time_sig_val, vocal_lang,
            adg, cfg_start, cfg_end, fmt, tiled,
            progress=gr.Progress(track_tqdm=True)
        ):
            """Wrapper function to call handler."""
            # Determine actual seed
            actual_seed = -1 if random_seed else int(seed_val) if seed_val is not None else -1
            
            return handler.generate_audio(
                task_type=task,
                caption=caption,
                lyrics=lyrics,
                audio_codes=codes,
                inference_steps=int(steps),
                guidance_scale=guidance,
                seed=actual_seed,
                reference_audio_path=ref_audio,
                source_audio_path=src_audio,
                repainting_start=repaint_start,
                repainting_end=repaint_end,
                audio_cover_strength=cover_strength,
                bpm=int(bpm_val) if bpm_val is not None else None,
                key_scale=key_val or "",
                time_signature=time_sig_val or "",
                vocal_language=vocal_lang,
                use_adg=adg,
                cfg_interval_start=cfg_start,
                cfg_interval_end=cfg_end,
                audio_format=fmt,
                use_tiled_decode=tiled,
                track_type=track_type_val,
                progress=progress
            )
        
        generate_audio_btn.click(
            fn=generate_audio_wrapper,
            inputs=[
                task_type, ace_caption, ace_lyrics, ace_audio_codes,
                inference_steps, guidance_scale, seed, use_random_seed,
                reference_audio, source_audio, repainting_start, repainting_end, audio_cover_strength,
                track_type,
                ace_bpm, ace_key_scale, ace_time_signature, vocal_language,
                use_adg, cfg_interval_start, cfg_interval_end, audio_format, use_tiled_decode
            ],
            outputs=[audio_output_1, audio_output_2, audio_generation_status, actual_texts]
        )
        
        # -----------------------------------------------------------------
        # Studio Integration Events (Frontend JavaScript)
        # -----------------------------------------------------------------
        # Note: These handlers use the js= parameter to execute in the browser
        # because Studio runs on the user's localhost and is not accessible
        # from the server-side Python code.
        
        # Connect button - uses frontend JS
        connect_studio_btn.click(
            fn=lambda x: x,  # Pass-through function (JS handles the logic)
            inputs=[studio_token],
            outputs=[studio_connection_status],
            js=JS_CONNECT_STUDIO
        )
        
        # Get from Studio buttons - uses frontend JS
        get_ref_from_studio_btn.click(
            fn=lambda: (None, ""),  # Placeholder (JS handles the logic)
            inputs=[],
            outputs=[reference_audio, studio_connection_status],
            js=JS_GET_AUDIO_FROM_STUDIO
        )
        
        get_src_from_studio_btn.click(
            fn=lambda: (None, ""),  # Placeholder (JS handles the logic)
            inputs=[],
            outputs=[source_audio, studio_connection_status],
            js=JS_GET_AUDIO_FROM_STUDIO
        )
        
        # Send to Studio buttons - uses frontend JS
        send_audio1_to_studio_btn.click(
            fn=lambda x: "",  # Placeholder (JS handles the logic)
            inputs=[audio_output_1],
            outputs=[studio_connection_status],
            js=JS_SEND_AUDIO_TO_STUDIO
        )
        
        send_audio2_to_studio_btn.click(
            fn=lambda x: "",  # Placeholder (JS handles the logic)
            inputs=[audio_output_2],
            outputs=[studio_connection_status],
            js=JS_SEND_AUDIO_TO_STUDIO
        )
    
    return demo