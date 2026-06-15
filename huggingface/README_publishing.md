# Publishing to Hugging Face — step-by-step

Ready-to-upload artifacts in this folder turn "weights uploaded" into a presentable, discoverable
project. Nothing here pushes automatically — run the commands below with your own HF token
(`huggingface-cli login`).

## Files in this folder

| File | Goes to |
|---|---|
| `qwen3-8b-advice-gguf_MODELCARD.md` | the **text** model repo's `README.md` |
| `qwen3-vl-8b-detection_MODELCARD.md` | the **vision** model repo's `README.md` |
| `esa-ad-star-splits_DATASETCARD.md` | a **dataset** repo's `README.md` |
| `space-demo/` | a Gradio **Space** repo (app.py + requirements.txt + README.md) |

## 1. Model cards (immediate, highest ROI)

The YAML frontmatter is what makes HF render eval results, link the base model, and surface the
model in search. Upload each card as the repo `README.md`:

```bash
huggingface-cli login
# text model
huggingface-cli upload dyrtyData/star-pipeline-qwen3-8b-advice-gguf \
  qwen3-8b-advice-gguf_MODELCARD.md README.md
# vision model
huggingface-cli upload dyrtyData/star-pipeline-qwen3-vl-8b-detection \
  qwen3-vl-8b-detection_MODELCARD.md README.md
```

## 2. Dataset repo (derived splits — legal under CC BY 3.0 IGO with attribution)

```bash
huggingface-cli repo create esa-ad-star-splits --type dataset
huggingface-cli upload dyrtyData/esa-ad-star-splits \
  esa-ad-star-splits_DATASETCARD.md README.md --repo-type dataset
# then upload the derived JSONL splits (NOT raw ESA-AD):
huggingface-cli upload dyrtyData/esa-ad-star-splits \
  ../data/splits/ . --repo-type dataset   # train/val/test_with_advice.jsonl
```

Keep the model cards' `datasets: [dyrtyData/esa-ad-star-splits]` field — it cross-links them.

## 3. Gradio demo Space

```bash
huggingface-cli repo create star-pipeline-demo --type space --space_sdk gradio
huggingface-cli upload dyrtyData/star-pipeline-demo space-demo/ . --repo-type space
```
Free CPU tier works (slow); for interactive speed use ZeroGPU (HF PRO) + Space env `N_GPU_LAYERS=-1`.

## 4. Collection (the "project page")

Create a Collection in the HF web UI ("+ New Collection"), e.g.
**"Spacecraft Telemetry Anomaly Detection (STAR-Pipeline)"**, and add, in order:
the text model → the vision model → the dataset → the demo Space. Add a one-line note to each.
This is the single link to share with interviewers.

## 5. (Optional) Paper page link

Add the ESA-AD arXiv URL (`https://arxiv.org/abs/2406.17826`) in each card body — HF auto-creates an
`arxiv:` tag and lists your repos on that paper's page. If you later post your own write-up to arXiv,
submit it at https://hf.co/papers/submit to appear in Daily Papers and link models+dataset+demo.

## Checklist

- [ ] Both model `README.md`s replaced with the cards here (eval-results render on the repo page)
- [ ] Dataset repo created, card uploaded, derived JSONL uploaded (raw ESA-AD NOT uploaded)
- [ ] Demo Space live
- [ ] Collection created and populated
- [ ] `base_model`, `license`, `datasets`, `model-index` present in each model card's YAML
