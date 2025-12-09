%%writefile /content/chatterbox-finetuning/src/gradio_local.py
import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
# [QUAN TRỌNG] Import class config để hack size
from chatterbox.models.t3.modules.t3_config import T3Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HACK CẤU HÌNH NGAY TỪ ĐẦU FILE ---
# Dòng này cực kỳ quan trọng, nó ép toàn bộ model khởi tạo với size 2549
print(">>> FORCING VOCAB SIZE TO 2549 <<<")
T3Config.text_tokens_dict_size = 2549
# --------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_model():
    print("Loading model from /content/chatterbox-finetuning/infer ...")
    # Đảm bảo folder infer đã có đủ: t3_cfg.safetensors, tokenizer.json, ve, s3gen
    model = ChatterboxTTS.from_local(ckpt_dir=str("/content/chatterbox-finetuning/infer"), device=DEVICE)
    print(f"Model loaded successfully on: {DEVICE}")
    print(f"Current Vocab Size in Model: {model.t3.text_emb.weight.shape[0]}")
    return model

def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    if model is None:
        T3Config.text_tokens_dict_size = 2549 # Hack lại lần nữa cho chắc
        model = ChatterboxTTS.from_local(ckpt_dir=str("/content/chatterbox-finetuning/infer"), device=DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    # Xử lý nếu không có file mẫu (Reference Audio)
    if audio_prompt_path is None:
        print("Warning: No reference audio provided!")
        # Trả về lỗi hoặc im lặng tùy bạn
        return None

    try:
        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfgw,
        )
        return (model.sr, wav.squeeze(0).cpu().numpy())
    except Exception as e:
        print(f"Error generation: {e}")
        return None

with gr.Blocks() as demo:
    model_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Xin chào, tôi là trí tuệ nhân tạo nói tiếng Việt.",
                label="Text Input",
                max_lines=5
            )
            # Input Audio (Bắt buộc phải có để clone giọng)
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio (Voice Clone)", value=None)

            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration", value=.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[model_state, text, ref_wav, exaggeration, temp, seed_num, cfg_weight],
        outputs=audio_output,
    )

if __name__ == "__main__":
    print("Starting Gradio...")
    demo.queue().launch(share=True, server_name="0.0.0.0", debug=True)
