# Space Telemetry Anomaly Detection & Resolution Pipeline (STAR-Pipeline)

## 1. Project Context & Business Objective

The primary objective of this project is to build an end-to-end, production-ready AI pipeline that detects anomalies in multivariate time-series data and automatically generates human-readable diagnostic advice.

Ultimate Goal: This project serves as a technical showcase for an upcoming AI Engineering interview for a mission-critical infrastructure role. The hiring team needs to see a demonstrated ability to move beyond external API wrappers and successfully fine-tune open-source foundation models for highly specific, localized business use cases (specifically, anomaly detection and end-user resolution advice). The architecture, ETL robustness, and evaluation metrics must reflect senior-level engineering standards.

## 2. Hardware Constraints & Compute Strategy

The pipeline must be cost-effective and intelligently split between local and cloud resources:

* Local Hardware (Inference, ETL & EDA): MacBook Pro (M3 Max, 14-core CPU, 30-core GPU, 36GB unified memory, 1TB SSD, macOS Tahoe 26.2). To be used for all exploratory data analysis, data transformation pipelines, and final local GGUF model inference/testing.
* Cloud Hardware (Model Training): Do not attempt full LoRA fine-tuning on the local machine to save time and optimize GPU usage. Utilize decentralized cloud GPU rentals (e.g., Vast.ai or RunPod) to spin up an Ubuntu/PyTorch container with a single NVIDIA RTX 4090 (24GB VRAM) or A6000 (48GB VRAM) for a few hours.
* Note: if anything can be set up with CLI, API, or MCP so that Claude Code can automate any process, please do so or instruct the user on how to do so. We also have access to chrome-devtools, so if anything can't be done through cli, api, mcp etc. and needs manual maneuvering, please write detailed instructions for Cursor Browser Use or chrome-devtools, or for the user to manually be able to follow step by step as if they have never done any of this before...please explain the concepts and the why behind things as well for the user to be able to understand and learn the process.

## 3. Reference Architectures & Research Directives

While traditional telemetry systems decouple detection and alerting, recent research proves LLMs can handle tokenized numerical sequences directly.

Agent Directive: Review the following implementations where LLMs were successfully trained for anomaly detection. Note: These are baseline examples. You must conduct deeper research to determine if there are newer, more efficient models or patching techniques for time-series data before finalizing the implementation plan.

* AnomLLM:[github.com/rose-stl-lab/anomllm](https://github.com/rose-stl-lab/anomllm)
* Time-LLM: Concepts surrounding time-series patching and prompt prefixing for LLMs.

Agent Directive for Fine-Tuning (Unsloth): The fine-tuning phase will heavily utilize Unsloth for its QLoRA optimization. You must thoroughly research the Unsloth documentation to determine the optimal data formatting (e.g., ChatML vs. ShareGPT), dataset structuring, and hyperparameters (LoRA rank, alpha, target modules) for a 7B-8B parameter model (e.g., Qwen2.5 or Llama-3).

* Unsloth Fine-Tuning Guide: [unsloth.ai/docs/get-started/fine-tuning-llms-guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
* Unsloth General Docs: [unsloth.ai/docs](https://unsloth.ai/docs)

## 4. Datasets

The model needs to learn the rhythm of nominal spacecraft operations to identify deviations. Agent Directive: Evaluate and select the most appropriate dataset from the following options to feed the pipeline.

* ESA Anomaly Dataset (ESA-AD): A ~12 GB real-life satellite telemetry dataset from the European Space Agency with curated anomaly annotations. [d-nb.info/1361113049/34](https://d-nb.info/1361113049/34) (Official benchmark pipeline reference: kplabs-pl/ESA-ADB on GitHub).
* NASA MSL & SMAP Telemetry: Real spacecraft multivariate time-series telemetry from the Curiosity Rover and SMAP satellite. [kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)

note: Validate this approach against your research and propose adjustments in your generated plan. If there are more relevant datasets that would be better for this project, please do deep research to find them, ideally they also satisfy what FDL in /Users/laptop/Developer/fdl_technicalInterview/space-telemetry-anom-llm/thoughts/issues/Nina_FDL.png is looking for -- 'space-based observing and working with satellite/space telescope data'. Explain all of your reasoning for whatever you choose.

## 5. Suggested Pipeline Walkthrough

Note to Agent: This is a suggested path. Validate this approach against your research and propose adjustments in your generated plan. Explain all of your reasoning for whatever you choose.

1. Phase 1: ETL & Data Transformation (Local): Utilize robust data engineering practices to process the raw HDF5/CSV telemetry data. Slice the continuous multivariate time-series into discrete, tokenizable rolling windows (patches) and format them into instruction-response JSONL pairs (e.g., mapping a telemetry sequence to a "Nominal" or "Anomaly + Resolution" string).
2. Phase 2: Traditional ML Baseline (Evaluation Phase): Train a traditional traditional machine learning model (LSTM, Autoencoder, or Isolation Forest) on the exact same dataset to serve as a baseline. This is required to empirically compare the fine-tuned LLM's recall/precision against standard industry practices.
3. Phase 3: LLM Fine-Tuning (Cloud): Set up the Unsloth environment on a Vast.ai/RunPod instance. Load the selected base model in 4-bit precision, apply the LoRA adapters, and execute the training loop using the processed JSONL data.
4. Phase 4: Inference & Integration (Local): Export the fine-tuned model to GGUF format, pull it down to the M3 Max, and run an inference script that feeds it unseen telemetry data to evaluate its diagnostic accuracy against the traditional ML baseline.

## 6. Definition of Done 

Note: these are proposed...Validate this approach against your research and propose adjustments in your generated plan. Explain all of your reasoning for whatever you choose.

* Data pipelines are modular, scalable, and successfully convert raw telemetry into LLM-digestible patches.
* A traditional ML baseline (e.g., LSTM) is trained and evaluated.
* An open-source LLM is fine-tuned via Unsloth and successfully infers both the presence of an anomaly and the correct diagnostic advice from a raw telemetry patch.
* Evaluation metrics (False Positives, Precision, Recall) are documented comparing the LLM approach vs. the traditional baseline.


Reference Doc - original Gemini thread: /Users/laptop/Developer/fdl_technicalInterview/space-telemetry-anom-llm/thoughts/issues/space-anomaly-detection-ai-advice_exploratoryThread.md


Note: always track all changes to git.
