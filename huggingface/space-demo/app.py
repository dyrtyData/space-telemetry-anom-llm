"""
STAR-Pipeline demo — Hugging Face Space (Gradio).

Loads the fine-tuned Qwen3-8B advice GGUF and, given a telemetry window, returns an
ANOMALY/NOMINAL verdict + structured DIAGNOSIS/ADVICE/ACTION.

Runs on a FREE CPU Space (slow: tens of seconds per generation for an 8B Q4_K_M model on
2 vCPU). For interactive speed, upgrade to a ZeroGPU Space (requires HF PRO) and set
N_GPU_LAYERS=-1. For the *vision* model demo, swap to transformers+peft on ZeroGPU and accept a
PNG plot as input (see the vision model card).
"""
import os
import gradio as gr
from llama_cpp import Llama

REPO = os.environ.get("STAR_GGUF_REPO", "dyrtyData/star-pipeline-qwen3-8b-advice-gguf")
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))  # 0 = CPU; -1 = all layers (GPU Space)

SYSTEM = (
    "You are a spacecraft telemetry analyst. Classify the telemetry window as ANOMALY or NOMINAL. "
    "If ANOMALY, provide DIAGNOSIS, ADVICE, and ACTION lines."
)

llm = Llama.from_pretrained(repo_id=REPO, filename="*Q4_K_M.gguf",
                            n_gpu_layers=N_GPU_LAYERS, n_ctx=2048, verbose=False)

EXAMPLE = ("mission=Mission1 channel=channel_41 "
           "values=[0.10, 0.11, 0.10, 0.12, 0.55, 0.61, 0.58, 0.60, 0.59, 0.62]")


def analyze(window: str) -> str:
    if not window.strip():
        return "Enter a telemetry window first."
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": window.strip()}],
        max_tokens=300, temperature=0.0,
    )
    return out["choices"][0]["message"]["content"].strip()


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=4, label="Telemetry window (normalized)", value=EXAMPLE),
    outputs=gr.Textbox(lines=10, label="Verdict + diagnostic advice"),
    title="STAR-Pipeline — Spacecraft Telemetry Anomaly Advisor",
    description=(
        "QLoRA-fine-tuned Qwen3-8B (GGUF) on ESA satellite telemetry. Returns ANOMALY/NOMINAL "
        "+ structured advice. Best used as the *advisor* behind a high-precision detector (the "
        "hybrid design). Note: on a free CPU Space, each analysis takes ~10-40s."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
