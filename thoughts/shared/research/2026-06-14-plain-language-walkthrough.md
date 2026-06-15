# STAR-Pipeline — Plain-Language Walkthrough

**Purpose:** This is the *learn-everything* companion to the rigorous
[results & analysis doc](2026-06-14-results-analysis.md). It assumes **no machine-learning
background** and explains, from scratch, every concept, every decision, and every result — so you
can present the project confidently and answer follow-up questions. If a term shows up in the
analysis doc and you're not 100% sure what it means, it's defined here.

Read it top to bottom once; after that, §10 (the interview Q&A) and §11 (glossary) are the
quick-reference parts.

---

## 1. The problem, in one breath

A spacecraft streams thousands of sensor readings ("telemetry") — voltages, temperatures,
pressures — second by second. Most of the time everything is **nominal** (normal). Occasionally
something drifts, spikes, or freezes: an **anomaly**. Two things are wanted:

1. **Detection** — automatically notice the anomaly.
2. **Advice** — tell a human operator, in plain English, *what* it means and *what to do*.

A product manager for a mission-critical system asked: *can a fine-tuned, open-source AI model do
both — so we own the model instead of renting an external API?* This project answers that
**empirically** (by building and measuring it) rather than by opinion.

---

## 2. The raw materials: time series, telemetry, and "windows"

**Time series** = a sequence of numbers measured over time (a heart-rate trace, a stock price, a
temperature log). Telemetry is many time series at once (one per sensor, called a **channel**).

The problem: an AI model can't look at a months-long stream all at once. So we chop each channel
into short, overlapping chunks called **windows** (here, 32 consecutive time-steps). Think of
sliding a 32-step ruler along the stream, moving it 16 steps each time (the "stride"). Each window
is one example the model judges: *is this 32-step snippet normal or anomalous?* This chunking is
called **patching** (the AnomLLM/Time-LLM papers use this term).

Two preprocessing steps matter:

- **Resampling to 1-hour cadence:** the raw data is enormous (tens of millions of points per
  channel). We average it onto a regular 1-hour grid to make it tractable. (Trade-off: very brief
  blips shorter than an hour can get smoothed away — noted as a limitation.)
- **RevIN normalization:** different channels live on wildly different scales (a voltage of ~28,
  a temperature of ~300). **Normalizing** rescales each channel to a common footing (roughly,
  "how many standard deviations from this channel's own average is this reading?") so the model
  compares *shapes*, not raw magnitudes. "RevIN" = *reversible instance normalization* — it can be
  undone later if you need the original units.

**Train / validation / test split:** we divide the windows into three piles. **Train** (21,000
windows) is what the model learns from. **Validation** (4,500) is used to check progress during
training. **Test** (4,500) is held back and only used at the very end to score the model on data it
has never seen — the honest exam. We kept the test pile **~25% anomalous** on purpose so the
numbers are interpretable.

**The dataset:** the **ESA Anomaly Dataset (ESA-AD)** — real telemetry from three European Space
Agency missions, with experts having marked which segments are true anomalies. We chose it over the
older NASA MSL/SMAP data because ESA-AD is newer, expert-annotated, and has its own *official
benchmark* (see §9), whereas the NASA set is widely criticized for label quality.

---

## 3. The classical detectors (the baselines we must beat)

Before reaching for an LLM, you build simpler, proven methods as a **baseline** — a yardstick. If
the fancy approach can't beat the simple one, that's important to know.

### Isolation Forest (the simple floor)
A non-deep-learning method. Intuition: anomalies are "few and different," so if you keep randomly
splitting the data, weird points get isolated quickly. It's fast and cheap — **but it treats every
timestamp independently and ignores the order of time entirely.** For telemetry, where *sequence*
is everything (a voltage drop is fine *after* a thruster fires, alarming if it's been off for an
hour), this is a serious handicap. Result: it drowns in false alarms (F1 just 0.188). It exists in
this project to show the floor.

### LSTM (the strong baseline)
An **LSTM** (Long Short-Term Memory network) is a neural network built to *remember* sequence — when
it looks at step 30 it hasn't forgotten step 1. This is why NASA's **Telemanom** system used LSTMs
for exactly this job. Our LSTM is an **autoencoder**: it learns to *reconstruct* normal telemetry.
Trained only on the rhythm of normal data, it reconstructs normal windows well but *fails* to
reconstruct anomalies (it's never seen that pattern). The size of that reconstruction failure (the
"error") is the anomaly signal; if the error exceeds a **threshold** (we use *mean + 3 standard
deviations*), we flag it. This is the single best **detector** in the whole project.

---

## 4. The LLM approach (the new idea being tested)

A **Large Language Model (LLM)** is the technology behind ChatGPT/Claude — trained to predict text.
The research frontier (the **AnomLLM** and **Time-LLM** papers) showed you can feed an LLM the
*numbers* of a telemetry window written out as text and fine-tune it to answer "ANOMALY or NOMINAL"
*and* write advice. One model does detection **and** explanation. That's the "unified LLM" idea this
project puts to the test.

Key terms:

- **Tokenization:** LLMs read **tokens** (chunks of text), not raw floating-point numbers. So we
  write the window's numbers out as text the model can read.
- **Base model:** an off-the-shelf, general-purpose LLM. We use **Qwen3-8B** ("8B" = 8 billion
  parameters — the "knobs" inside the model; 8B is small enough to run on a laptop, big enough to be
  capable). We picked Qwen3 over alternatives for its stronger reasoning/math scores and good
  tooling support.
- **Fine-tuning:** taking that general base model and training it a little more on *your* specific
  data so it specializes — here, on spacecraft anomalies and advice. **This is the core skill the
  whole project is meant to demonstrate.**

### Why fine-tuning is feasible on a budget: QLoRA

Training all 8 billion knobs would need a giant, expensive machine. **LoRA** (Low-Rank Adaptation)
is a trick: freeze the original model and train only a tiny set of small "adapter" matrices bolted
onto it — a fraction of a percent of the weights. **QLoRA** adds **quantization**: it loads the big
frozen model in a compressed 4-bit form (instead of 16-bit) so it fits in far less memory. Together
they let us fine-tune an 8B model on a **single rented GPU for a couple of dollars**. We used a
toolkit called **Unsloth** that makes this fast and handles the fiddly parts.

- **r=16, α=16:** the "size" and "scaling" of the LoRA adapter — sensible standard settings.
- **3 epochs:** the model saw the whole training set 3 times.
- **learning rate 2e-4:** how big a step the model takes each update — a standard, stable value.

### Getting it onto a laptop: GGUF + Metal

After training in the cloud, we **export** the model to a format called **GGUF** and **quantize** it
(`Q4_K_M` — a good 4-bit compression that keeps quality while shrinking the file to ~4.7 GB). GGUF
runs efficiently on a Mac via **Metal** (Apple's GPU framework), through a library called
`llama-cpp-python`. End result: the fine-tuned model runs **locally on the MacBook**, with **no
external API** — exactly the "we own the model" requirement.

### The two LLM flavors we built
- **Text model (Qwen3-8B):** reads the numbers as text, outputs ANOMALY/NOMINAL **plus** structured
  advice. This is the main model.
- **Vision model (Qwen3-VL-8B):** the *AnomSeer* idea — instead of numbers, we draw each window as a
  **PNG plot** (a little line chart) and fine-tune a model that "looks at the picture" to decide
  ANOMALY/NOMINAL. No advice; pure detection. It's a second, independent way to detect.

---

## 5. How we score everything (the metrics, in plain English)

Imagine the model flags some windows as anomalies. Four outcomes:
- **True Positive (TP):** flagged, and it really was an anomaly. ✅
- **False Positive (FP):** flagged, but it was actually normal — a *false alarm*. ❌
- **False Negative (FN):** not flagged, but it really was an anomaly — a *miss*. ❌
- **True Negative (TN):** not flagged, and it really was normal. ✅

From these:

- **Precision** = of everything you flagged, what fraction were real? *"When the alarm rings, how
  often is it right?"* Low precision = alert fatigue.
- **Recall** = of all the real anomalies, what fraction did you catch? *"How many real problems did
  you miss?"*
- **F1** = a single balanced score combining precision and recall (their harmonic mean). One number
  to rank approaches.
- **CEF0.5** = like F1 but it **weights precision more heavily** (β=0.5). In a control room a false
  alarm is more annoying/costly than a near-miss, so this is the *operationally honest* score. It's
  the metric the ESA benchmark favors.
- **Affinity-F1** = an *interval-aware* score for streaming data. Real anomalies last a *stretch* of
  time, not one instant. This metric merges your flagged windows into time intervals and checks
  whether your intervals line up with the true ones. It only makes sense on a **continuous**
  timeline (see the "degenerate" note below).

**The trivial baseline trick (very important).** What score do you get by being *stupid* — flagging
**every** window as anomalous? You catch 100% of anomalies (recall = 1.0) but your precision is just
the anomaly rate (~25%), giving **F1 ≈ 0.40 for free**. This "always-anomaly" line is the bar every
real detector must clear. It's the single most important idea for reading our results honestly:
**an F1 around 0.40 is worthless if it came from over-flagging.** (More on this in §7.)

---

## 6. The journey — what actually happened, phase by phase (incl. the messy parts)

The honest story matters; it's part of what makes this a *senior* showcase. The work ran across ~87
work sessions over three days and hit plenty of real-world walls. Here's the arc.

- **Phase 1 — ETL (getting the data usable).** The plan *assumed* the dataset came as a couple of
  big files per mission. It didn't — ESA-AD ships as hundreds of separate per-channel files in a
  different structure, so the data-loading code had to be rewritten from scratch (deviation **D2**).
  The download was painfully slow from the official source, so we switched to a byte-identical
  Kaggle mirror (**D3**). Naïve windowing would have produced *tens of millions* of windows, so we
  added 1-hour resampling and balanced subsampling to cap it at 30,000 (**D4**). One mission stored
  some channels as words instead of numbers, which crashed things until handled (**D5**). *Lesson:
  real data never matches the spec; budget for it.*
- **Phase 1.5 — Advice labels.** To teach the model to write advice, we first need example advice.
  We generated 7,457 structured advice records (one per anomaly) *in-session* (no paid API), each
  with a diagnosis, recommended action, severity, and pattern type. (These are *synthetic* labels —
  a noted limitation.)
- **Phase 2 — Baselines.** Built the LSTM and Isolation Forest. To keep it quick, we first ran just
  **3 channels** as a smoke test. (This "only 3 channels" shortcut becomes important later.)
- **Phase 3 — Fine-tuning in the cloud.** Rented an NVIDIA RTX 4090 on Vast.ai (~$0.49/hr). Lots of
  friction: an SSH key with a passphrase blocked automation (**D11**); a `pkill` command
  accidentally killed its own shell (**D12**); the training library had changed its API in a new
  version and the code needed rewriting (**D9**). Once running, training was smooth: loss fell from
  2.85 to 0.24 over 3 epochs. Total cost ~**$2.30**.
- **Phase 4 — Bring the model home.** The exported model file was 4.68 GB — just over the **4 GB
  single-file limit of the FAT32-formatted** external drive (**D17**), so it had to live on the
  Mac's internal SSD. Downloading 5 GB over a trans-Atlantic SSH link from Hungary was crawling at
  ~300 KB/s (**D18**); we solved it by pushing the model to Hugging Face and pulling it back through
  their fast CDN (**D20**). *Lesson: "cheapest GPU" isn't "fastest" once you count data transfer.*
- **Phase 5 — Full evaluation.** Ran the model over all 4,500 test windows. This multi-hour job
  **died twice**: once because its progress output was buffered and invisible (it looked alive but
  wasn't), once because the laptop went to sleep. We then hardened it: unbuffered logging,
  checkpoint-every-250-windows with atomic writes, a `--resume` flag, and proper detachment so it
  survives the session ending (**D25**). *Lesson: any job over a few minutes must be made
  resumable and observable before you trust it.* Result: text-LLM F1 0.453, with 99.6% of flags
  carrying structured advice.

At this point the project "worked" — but a careful review found **four gaps** that an interviewer
would immediately poke. **Phases 6–9 exist to close them.**

- **Phase 6 — "Did the fine-tuning actually help?"** The cleanest proof is to run the *same model
  before fine-tuning* through the *exact same test harness*. We did — plus a few-shot version, plus
  a frontier model (Claude) as a sanity check, plus the trivial baseline. This is the heart of the
  result (§7). It revealed the *real* story and even caught a fairness bug we fixed (the frontier
  was first tested zero-shot while the base got few-shot examples — an apples-to-oranges
  comparison; we re-ran the frontier few-shot too, **D31**).
- **Phase 7 — Level the detection field.** The LSTM had only been scored on 3 hand-picked channels.
  We ran it on **all 58** Mission-1 target channels. The honest F1 *dropped* from 0.663 to **0.552**
  — and we kept the lower, honest number. Bonus: scoring continuous channel timelines made
  Affinity-F1 *meaningful* (0.649) for the first time. (The run died once more — and `--resume`
  recovered it for free, vindicating the Phase-5 hardening, **D38**.)
- **Phase 8 — The vision detector.** Trained the Qwen3-VL model on PNG plots (originally planned,
  never run). Rented a bigger A6000 GPU (~$0.40/hr) — and discovered the vision training code,
  written back in Phase 3 but *never executed*, had three latent bugs (**D31** in the log). Once
  fixed, it trained in ~65 min and scored F1 0.457 with high precision (0.769). Cost ~$1.
- **Phase 9 — Is the advice any good?** We'd only proven the advice had the right *shape* (99.6%
  structured). Phase 9 graded a 120-flag sample for *correctness* on a transparent rubric (§7).

Total cloud spend across everything: **~$3.33.**

---

## 7. What the results actually mean (the part to internalize)

### 7a. The detector contest: the LSTM wins
On a fair, full-channel comparison the **LSTM is the best detector** (F1 0.552, precision 0.785,
CEF0.5 0.684). The direct text-LLM detector loses on precision badly (0.360) — it flags 1,898
windows when only 1,123 are truly anomalous, so ~2 of every 3 alarms are wrong. Its *one* edge is
recall (it catches more), but at an unacceptable false-alarm cost for a control room. **This is the
expected result** — it matches the published AnomLLM finding that LLMs trade precision for recall
and don't beat tuned sequence models at detection yet. Reporting this honestly (rather than hiding a
negative result) is exactly the signal the brief wanted.

### 7b. But fine-tuning *did* help — and here's the airtight version
Rank the LLM-family approaches against the **trivial always-anomaly line (F1 0.399)**:

- **Raw base model, zero-shot:** F1 **0.000.** It won't even follow the output format — it rambles
  in "thinking mode" and never commits to ANOMALY/NOMINAL. (Learning to *obey the output contract*
  is itself something fine-tuning taught it.)
- **Base model with 2 examples (few-shot):** F1 **0.420** — *looks* competitive, but it got there
  by flagging ~80% of all windows. That's barely above flag-everything. It's **over-flagging, not
  detecting.** It's also 3× slower and writes proper advice only 13% of the time.
- **Frontier model (Claude), zero-shot and few-shot:** F1 **0.254 / 0.239** — *below* the dumb
  baseline. A much more capable general model, given two different prompts, **sits at chance.** Why?
  Because it's being shown ~10 normalized numbers with no channel history — there's almost no signal
  there to find. Adding examples doesn't help, which proves it's not a prompt-wording problem.
- **Our fine-tuned model:** F1 **0.453** with *balanced* precision/recall (0.360 / 0.609). **It is
  the only one that clears the trivial baseline with a real trade-off** — the lone genuine detector
  in the LLM family.

The takeaway: **fine-tuning bought the mission/channel-specific knowledge that no prompt could
supply.** That's the whole point of fine-tuning, and it's demonstrated, not asserted. Plus
fine-tuning gave near-perfect output compliance, reliable structured advice, and 3× faster
inference.

### 7c. The advice is good — *when the flag is right*
We graded 120 of the model's flags (correctness / actionability / grounding, 0–6):
- On **true positives** (the model correctly flagged a real anomaly): **5.58/6, 95% high-quality** —
  it names the right channel, the right magnitude, a sensible action. Genuinely useful.
- On **false positives** (false alarms): ~**1/6** — it confidently explains an anomaly that isn't
  there. Garbage in, garbage out.

So the advice quality is **gated by detection precision**. The conclusion writes itself: don't use
the LLM to *decide* what's anomalous; use a high-precision detector for that, and use the LLM to
*explain* the things that detector flags.

### 7d. The recommended design: the Hybrid
```
window → [LSTM detector: cheap, precise, instant] → if it flags → [Qwen3-8B: writes the advice] → operator
```
Cheap precise detection + expensive-but-only-when-needed explanation. It inherits the LSTM's strong
detection *and* adds the advice the baselines can't produce. **This is the answer to the friend's
original question:** a localized, open-source, no-external-API system that both detects and advises,
with a cost profile that works in production.

---

## 8. Why each big decision was made (so you can defend it)

| Decision | Why |
|---|---|
| **ESA-AD over NASA MSL/SMAP** | Newer, expert-annotated, has its own official benchmark; NASA set has known label-quality criticism. |
| **Qwen3-8B** | Strong reasoning/math scores, well-supported by Unsloth, 8B fits a laptop after quantization. |
| **QLoRA, not full fine-tuning** | Trains <1% of weights in 4-bit → a few dollars on one rented GPU instead of a cluster. |
| **Cloud for training, laptop for inference** | Training needs NVIDIA/CUDA; inference runs fine on the Mac's Metal GPU — matches the "own the model, no API" goal and the hardware budget. |
| **GGUF + Metal locally** | Lets the fine-tuned model run on the MacBook with no vendor dependency. |
| **Build *all* approaches, not just the LLM** | Makes the recommendation empirical. The brief explicitly asked to validate, not assert. |
| **CEF0.5 + Affinity-F1, not point-adjustment** | These are the ESA benchmark's metrics; point-adjustment is known to inflate scores. |
| **The trivial baseline** | Without it, an over-flagging F1 of 0.42 looks like "detection." With it, the truth is visible. |
| **Checkpoint/resume on long jobs** | A multi-hour job that dies at 90% shouldn't cost the whole run — a real production habit. |

---

## 9. Where this sits in the field (the "contribution" answer)

If asked *"what does this contribute?"*, say:

> It takes the field's open-source **gold-standard pieces** — the **ESA-ADB** benchmark and its
> evaluation metrics, NASA's **Telemanom** LSTM method, and the **AnomLLM/AnomSeer/Time-LLM** line
> of LLM-based detection — and runs them as a **single, honest, like-for-like bake-off on real ESA
> telemetry**, across three input modalities (numbers-as-text, numbers-as-image, classical
> features). It then answers the question most fine-tuning demos dodge — *"did the fine-tuning
> actually help, or could you have just prompted an API?"* — with a comparison framed against a
> trivial baseline so the answer survives a skeptic. The transferable findings: (1) on this data a
> tuned LSTM still out-detects a fine-tuned LLM, matching the literature; (2) a fine-tuned 8B model
> nonetheless beats both a few-shot base and a frontier model, because it learned localized priors
> prompting can't supply; and (3) LLM "explainability" is only as good as the detector feeding it —
> which is why the deployable design is the hybrid.

It's not a new algorithm; it's a **rigorous applied result with an honest headline** — which is
precisely the engineering maturity the role is screening for.

---

## 10. Interview Q&A (anticipate these)

**Q: Did the LLM beat the traditional baseline?**
A: At *detection*, no — and that's the honest, expected result. The LSTM is more precise (0.785 vs
0.360) and wins F1 and CEF0.5. The LLM's value is elsewhere: it's the only LLM-family approach that
beats a trivial baseline with balanced precision/recall, and it adds high-quality advice the
baselines can't. So the right design is the hybrid.

**Q: Then why fine-tune at all — why not just prompt Claude/GPT?**
A: We tested exactly that. A frontier model prompted zero- *and* few-shot scored *below* the dumb
baseline on this input. The signal the fine-tune uses — mission/channel-specific "normal" — isn't in
the prompt; it's learned into the weights. Prompting can't recover it. Fine-tuning also gave us a
model we *own*, that runs locally with no API, at 3× the speed.

**Q: Your few-shot base got F1 0.420, almost matching your 0.453. Doesn't that undermine the
fine-tune?**
A: No — that 0.420 is a mirage. It comes from flagging ~80% of windows (precision 0.282), barely
above flag-everything. It's over-flagging, not detecting. The fine-tune hits a similar F1 with a
*balanced* trade-off, plus reliable advice (99.6% vs 13%) and 3× lower latency.

**Q: Isn't comparing a 3-channel LSTM to a 4,500-window LLM unfair?**
A: It was, in Phase 5 — which is why Phase 7 re-ran the LSTM on all 58 channels. Its honest F1 fell
from 0.663 to 0.552, and we kept the lower number. It still wins. Full like-for-like on one
contiguous stream is the remaining refinement.

**Q: How do you know the advice is actually correct, not just well-formatted?**
A: Phase 9 graded a sample on a transparent rubric. On correct flags it's 5.58/6 (95% high-quality,
right channel and magnitude). On false alarms it's ~1/6 — it explains a non-existent anomaly. So
advice quality is gated by detection precision, which is the argument for the hybrid.

**Q: What was the hardest engineering part?**
A: Making the multi-hour eval survivable. It died to output buffering and to machine sleep before I
added unbuffered logging, atomic checkpoint/resume, and proper daemonization. That paid off when the
58-channel LSTM run later died and resumed for free.

**Q: What would you do next?**
A: Ship the hybrid; level the detection field on one contiguous stream; calibrate the LLM's decision
threshold for a precision/recall curve; widen the advice grade with a human SME; and ensemble the
recall-oriented text model with the precision-oriented vision model.

---

## 11. Glossary (quick reference)

- **Anomaly** — a segment of telemetry that deviates from normal behavior.
- **Channel** — one sensor's time series (e.g. one voltage line).
- **Telemetry** — many channels of sensor data streamed from a spacecraft.
- **Window / patch** — a short fixed-length slice of a channel (here 32 steps) — one example.
- **Stride** — how far the window slides each step (16).
- **Normalization / RevIN** — rescaling each channel to a common footing; RevIN is the reversible
  version.
- **Train / validation / test** — learn-from / tune-on / final-exam data piles.
- **Baseline** — a simpler reference method you must beat to claim progress.
- **LSTM** — a neural network that remembers sequence; good for time series.
- **Autoencoder** — a network that learns to reconstruct its input; reconstruction *error* flags
  anomalies.
- **Isolation Forest** — a fast tree-based detector that ignores time order (the simple floor).
- **LLM** — large language model (ChatGPT/Claude-style), trained to predict text.
- **Token / tokenization** — the text chunks an LLM reads; we render numbers as text tokens.
- **Base model** — the off-the-shelf LLM before any of our training (Qwen3-8B).
- **Frontier model** — a top-tier general model (here Claude), used as a prompting sanity check.
- **Fine-tuning** — extra training of a base model on your specific data to specialize it.
- **LoRA** — fine-tune only small adapter matrices, leaving the big model frozen.
- **QLoRA** — LoRA + loading the frozen model in compressed 4-bit form (saves memory).
- **Quantization** — storing model weights in fewer bits (e.g. 4-bit) to shrink/speed it.
- **Unsloth** — the toolkit that makes QLoRA fine-tuning fast and GGUF export easy.
- **GGUF** — a model file format that runs efficiently on laptops/CPUs/Apple GPUs.
- **Metal** — Apple's GPU framework; runs the GGUF model on the Mac.
- **Zero-shot / few-shot** — prompting a model with no examples / with a handful of examples, *no*
  fine-tuning.
- **Epoch** — one full pass through the training data.
- **TP / FP / FN / TN** — true/false positive/negative (see §5).
- **Precision** — fraction of flags that were real (alarm accuracy).
- **Recall** — fraction of real anomalies caught (miss rate's complement).
- **F1** — balanced precision+recall score.
- **CEF0.5** — precision-weighted F-score; the cost-of-false-alarm-aware metric.
- **Affinity-F1** — interval-aware F-score for continuous streams.
- **Trivial / always-anomaly baseline** — flag everything; the "free" score every detector must
  beat (F1 ≈ 0.40 here).
- **Hybrid** — the recommended design: cheap precise detector triggers an LLM advisor.
- **Telemanom** — NASA JPL's LSTM anomaly-detection method (our LSTM follows it).
- **AnomLLM / AnomSeer / Time-LLM** — the research line on LLMs detecting time-series anomalies
  (text, vision, and patching approaches respectively).
- **ESA-AD / ESA-ADB** — the European Space Agency anomaly dataset and its official benchmark
  pipeline.
- **CEF / point-adjustment** — ESA's preferred metric / a discredited metric we avoided.
