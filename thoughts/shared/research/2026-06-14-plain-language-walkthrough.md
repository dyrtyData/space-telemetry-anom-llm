# STAR-Pipeline — Plain-Language Walkthrough

**Purpose:** This is the *learn-everything* companion to the rigorous
[results & analysis doc](2026-06-14-results-analysis.md). It assumes **no machine-learning
background** and explains, from scratch, every concept, every decision, and every result — so you
can present the project confidently and answer follow-up questions. If a term shows up in the
analysis doc and you're not 100% sure what it means, it's defined here.

Read it top to bottom once; after that, §10 (the FAQ) and §11 (glossary) are the
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
"error") is the anomaly signal; if the error exceeds a **threshold**, we flag it. (How high to set
that threshold is a knob we *tuned* — see §7d; we landed on *mean + 4 standard deviations*.) This is
the strongest **single** detector in the project — though, as it turns out, *combining* all the
detectors does better still (§7e).

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
- **AUC-PR** = a *threshold-free* score. Most detectors don't just say yes/no — under the hood they
  produce a *confidence* (how anomalous is this window?). AUC-PR measures how well that confidence
  *ranks* real anomalies above normal windows, averaged over every possible cut-off. It answers "does
  the model know what's anomalous?" separately from "did we pick a good alarm threshold?" — a
  distinction that turns out to matter a lot here (§7d).

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
- **Phase 1.5 — Advice labels (what they were for, and how they were made).** To *teach* the model
  to write advice, we first need example advice to learn from — you can't fine-tune a model to
  produce something you never show it. So we generated **7,457 advice records, one per anomaly
  window**, each with a `DIAGNOSIS / ADVICE / ACTION`, a severity, and a pattern type. They were
  **templated in-session from each window's own statistics** (e.g. the *shape* and *fraction* of the
  window that deviates → a pattern like "subtle deviation" or "persistent anomaly" → a severity → a
  matching recommended action). No paid API, and — importantly — **not written by human experts**, so
  they are *synthetic* labels (a real limitation; the model's advice can only be as good as these).
  These advice records were then merged into the **training answers**. **One model learns both
  jobs:** the text Qwen3-8B was trained on `(window → verdict + advice)` pairs, so the *same* model
  emits ANOMALY/NOMINAL **and** the advice — detection and explanation come from one checkpoint. (The
  vision model is separate and detection-only.)
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

At this point the project "worked" — but a careful review found **four gaps** a reviewer would
immediately poke. **Phases 6–9 closed them, and Phases 11–14 then hardened the result.**

- **Phase 6 — "Did the fine-tuning actually help?"** The cleanest proof is to run the *same model
  before fine-tuning* through the *exact same test harness*. We did — plus a few-shot version, plus
  a frontier model (Claude) as a sanity check, plus the trivial baseline. This is the heart of the
  result (§7). It even caught a fairness bug we fixed (the frontier was first tested zero-shot while
  the base got few-shot examples — an apples-to-oranges comparison; we re-ran the frontier few-shot
  too, **D31**).
- **Phase 7 — Level the detection field.** The LSTM had only been scored on 3 hand-picked channels.
  We ran it on **all 58** Mission-1 target channels — the honest F1 *dropped* (no more cherry-picking),
  and scoring continuous channel timelines made Affinity-F1 *meaningful* for the first time. (The run
  died once more — and `--resume` recovered it for free, vindicating the Phase-5 hardening, **D38**.)
- **Phase 8 — The vision detector.** Trained the Qwen3-VL model on PNG plots. Rented a bigger A6000
  GPU — and discovered the vision training code, written back in Phase 3 but *never executed*, had
  three latent bugs. Once fixed, it trained in ~65 min and scored F1 0.457 with high precision (0.769).
- **Phase 9 — Is the advice any good?** We'd only proven the advice had the right *shape* (99.6%
  structured). Phase 9 graded a 120-flag sample for *correctness* on a transparent rubric (§7).

Then five more phases turned a good showcase into a genuinely strong result:

- **Phase 11 — Tune the LSTM.** Its alarm threshold had been left at an untuned default. Sweeping that
  one knob (and discovering Telemanom's fancier "dynamic" thresholding actually *hurts* here — a clean
  negative result) lifted the LSTM to its final P 0.837 / F1 0.553 (§7d).
- **Phase 12 — The vision base control.** We ran the *un-fine-tuned* Qwen3-VL too, to finish the
  "did fine-tuning help?" story for the picture model as well as the text one (§7b).
- **Phase 13 — Calibrate the text LLM.** The big surprise: the text model's "over-flagging" was mostly
  a *reading* error, not a model weakness — fixed by reading its confidence properly (§7d).
- **Phase 14 — Combine the models.** Fuse the three detectors' confidences into one — and it beats
  every single model (§7e).
- **Phase 15 — Test RAG as an alternative to fine-tuning.** If the frontier's failure was a *context*
  problem (missing channel history), could we fix it without fine-tuning? We built a retrieval index
  from the training windows and gave models k=5 labeled neighbors per window. Result: **Frontier+RAG
  F1 0.825** (vs 0.254 zero-shot) — even **Base+RAG (0.531) beats the fine-tune (0.453)**. This
  validates *why* fine-tuning works (it's the channel context) and reveals an alternative path for
  API-tolerant deployments (§7g).

Total cloud spend across everything: **~$4.2.**

---

## 7. What the results actually mean (the part to internalize)

### 7a. The detector contest: the LSTM wins (among single models)
On a fair, full-channel comparison the **LSTM is the best *single* detector** (F1 0.553, precision
0.837, CEF0.5 0.705). As deployed, the direct text-LLM detector looks much worse on precision (0.360)
— it raised far more alarms than there were real anomalies. Its *one* edge is recall (it catches
more). **This is the expected result** — it matches the published AnomLLM finding that LLMs trade
precision for recall and don't beat tuned sequence models at detection. Reporting it honestly is the
whole point of an empirical bake-off.

Two twists make this more interesting than "classical beats LLM," though. **First**, the **vision LLM**
(Qwen3-VL, reading the *plots*) is the precision-oriented opposite of the text model: 0.769 precision,
it almost never false-alarms, so it has the best false-alarm-aware score (CEF0.5 0.604) of any single
LLM. **Second** — and this is the headline of the whole detection story — the text LLM's "over-flagging"
turns out to be a fixable *reading* error (§7d), and because the three detectors make *different*
mistakes, **combining them beats every single one of them** (§7e). So the real winner isn't the LSTM
alone — it's the LSTM, the text model, and the vision model *fused together*.

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
  This is worth understanding because it's the core reason fine-tuning matters here. Each window is
  shown to the model as **~10 normalized numbers plus a mission and channel name** — nothing else.
  The information that actually separates "anomaly" from "normal" — the **signal** — is whether that
  little run of numbers is unusual *for that specific channel*. And knowing that requires **channel
  history**: what's normal for, say, `Mission1/channel_41` (its usual level, its rhythm, its noise).
  *Example:* a reading of 0.6 might be completely normal on a battery-voltage channel but a glaring
  fault on a temperature channel — with no history, you simply can't tell which. The frontier model
  has **never seen this spacecraft's channels**, so 10 context-free numbers carry almost no signal —
  especially for the most common anomalies here, which are *subtle deviations* that look nearly normal
  in 10 numbers. The fine-tuned model, by contrast, **learned each channel's "normal" from 21,000
  training windows** — that learned-in knowledge is exactly what the frontier lacks, and adding a
  couple of prompt examples can't substitute for it (which is why few-shot doesn't help). *Could you
  give the frontier that history?* Yes — by retrieving past windows for the channel into the prompt
  (**RAG**), or by **fine-tuning the frontier** itself. Neither was done here; both are in "what next"
  (§10) — with the caveat that doing so brings back the vendor dependency the owned model avoids.
- **Our fine-tuned model:** F1 **0.453** with *balanced* precision/recall (0.360 / 0.609). **It is
  the only one that clears the trivial baseline with a real trade-off** — the lone genuine detector
  in the LLM family.

The takeaway: **fine-tuning bought the mission/channel-specific knowledge that no prompt could
supply.** That's the whole point of fine-tuning, and it's demonstrated, not asserted. Plus
fine-tuning gave near-perfect output compliance, reliable structured advice, and 3× faster inference.

**The picture model tells the same story from the opposite end.** We also ran the *un-fine-tuned*
Qwen3-VL base on the plots. Unlike the text base (which couldn't even produce a clean ANOMALY/NOMINAL
answer), the picture base *answers* fine — it's 100% compliant — but it **can't tell anomalous plots
from normal ones**: F1 0.350, *below* the dumb baseline. Fine-tuning then more than doubled its
precision (0.310 → 0.769). So the two modalities expose the same lesson from opposite ends: for the
**text** model, fine-tuning's headline win was *learning to follow the format*; for the **picture**
model, it was *learning to actually discriminate*. Both wins come from the same thing — teaching the
model what *this spacecraft's* data looks like.

### 7c. The advice is good — *when the flag is right*
We graded 120 of the model's flags (correctness / actionability / grounding, 0–6):
- On **true positives** (the model correctly flagged a real anomaly): **5.58/6, 95% high-quality**,
  with **near-perfect grounding (1.86 out of 2)**. Genuinely useful.
- On **false positives** (false alarms): ~**1/6** — it confidently explains an anomaly that isn't
  there. Garbage in, garbage out.

**What "grounding" / "right channel" / "subsystem" mean (you asked).** A **channel** is one single
sensor's data stream — e.g. `Mission1/channel_41`, say a battery-bus voltage. A **subsystem** is the
bigger functional group that channel belongs to: *power, thermal, attitude control, propulsion,
comms.* So channel_41 (a voltage) lives in the *power* subsystem. "Grounding" measures whether the
advice is tied to the actual data: did it name the **right channel**, and is the magnitude it quotes
consistent with the numbers in the window? In our sample: **119 of 120 named the correct channel**,
and only **3 of 120 put it in the wrong subsystem** (right sensor, wrong category — e.g. calling a
power channel "thermal"). So to answer your question directly: **yes, it's still doing a good job** —
even on the rare miss it usually points at the exact right sensor; what occasionally slips is the
higher-level label, not the location. That's why grounding is 1.86/2 (near-perfect), not a flat 100%.

So the advice quality is **gated by detection precision**. The conclusion writes itself: don't use
the LLM to *decide* what's anomalous; use a high-precision detector for that, and use the LLM to
*explain* the things that detector flags.

**Is 95% enough for a "lives depend on it" system? No — and here's how you'd close the gap.** 95%
high-quality-when-correct (graded by an LLM, on 120 examples, against *synthetic* advice labels) is a
strong *showcase* number, not a production guarantee for mission-critical use. The biggest levers,
in order:
1. **Human-expert advice labels.** The current ceiling is the templated Phase-1.5 labels; real
   fault-engineer advice (with true root causes) is the single biggest improvement.
2. **Ground it in real references (RAG):** give the advisor the channel's spec sheet / fault catalog
   so it cites documented causes and fixes instead of generic patterns.
3. **Put it behind a high-precision detector** (the hybrid) so it's rarely asked to explain a false
   alarm in the first place.
4. **Let it say "I'm not sure"** (calibrated confidence / abstention) instead of confidently
   inventing a story on a borderline window.
5. **A bigger advisor model + a larger, human-checked grade** if resources allow.

### 7d. The plot twist: the text model wasn't over-flagging — we were misreading it
The text LLM's ugly precision (0.36) had two innocent causes, neither of them "the model is bad at
detecting":
1. **It was answering with randomness turned on.** We'd been decoding with "temperature" (a setting
   that adds randomness so the model doesn't always pick its single most-likely answer). On a yes/no
   verdict, that randomness flips some NOMINALs to ANOMALY by chance. Just turning it *off* lifted
   precision from 0.36 to **0.53** — for free, same model.
2. **We were reading only its yes/no, not its confidence.** Under the hood the model produces a
   *confidence* that a window is anomalous. Instead of taking its bare yes/no, we read that confidence
   and chose a sensible cut-off (only alarm when it's quite sure). Push the cut-off up and precision
   reaches **0.84** — *more than double* the original, and now competitive with the LSTM.

The lesson is worth internalizing: **the model wasn't a bad detector; we were reading its answer
badly.** "Low precision" was a calibration problem (how we turned its output into an alarm), not a
capability problem. (This is also why a "detection-only" retrain isn't worth it — the fix was the
threshold, not the model.) And reading the *confidence* instead of the yes/no is exactly what makes
the next step possible.

### 7e. Combining the models beats all of them
Three detectors, three *different* ways of being wrong: the text model over-flags, the picture model
under-flags, the LSTM is precise on yet another basis. When detectors make **independent** mistakes,
a smart combination can beat any one of them — the errors partly cancel. So we took each model's
*confidence score* (§7d) and trained a tiny "judge" (a **stacker** — a small model that learns how
much to trust each detector) to fuse them into one number.

It works, and it's the strongest detection result in the project: fusing all three reaches **precision
0.922** (and the best false-alarm-aware score, 0.781) — beating every single model. A simple
**"2-out-of-3 agree"** vote also does well, and a deployable version is **"all agree → alarm; they
disagree → ask a human."** *Honest caveat:* this was measured on the windows where all three models
had a score (a fair head-to-head subset), so its number sits on a slightly different yardstick than
the others — but on *identical* windows, the combination wins. (The three-model combo currently
covers Mission 1 only, because we only ever trained the LSTM on Mission 1 — see "what next.")

### 7f. The recommended design: a fused detector + the advisor
```
window → cheap LSTM screen → if it flags → fuse all 3 detectors (precise) → Qwen3-8B writes the advice → operator
```
Use each part where it's strongest: the **LSTM** is free and instant, so let it screen every window;
on the few it flags, run the **fused ensemble** to cut false alarms to a minimum; then the **text LLM**
turns the confirmed flag into a human-readable diagnosis. Cheap where it can be, precise where it
matters, explainable at the end. **This is the answer to the original question:** a localized,
open-source, no-external-API system that both detects (better than any single model) and advises —
with a cost profile that works in production. (Simpler variants: just `LSTM → advisor` if you want
the cheapest thing that works, or `vision → advisor` where false alarms are especially costly.)

### 7g. The RAG alternative: retrieval instead of training (Phase 15)

We said the frontier failed because it lacked **channel history** — the knowledge of what's normal for
each sensor. Fine-tuning bakes that knowledge into the model's weights. But there's another way: **give
it the history at runtime** by retrieving similar past windows from the training data. This is called
**RAG (retrieval-augmented generation)**.

**How it works:** For each test window, we search a FAISS index for the k=5 most similar training
windows *from that channel*, along with their labels ("ANOMALY" or "NOMINAL"). We paste these into the
prompt as context: "here are 5 similar past windows and what they turned out to be — now classify this
one."

**The results are striking:**

| Model | Without context | With RAG (k=5) |
|-------|-----------------|----------------|
| Frontier (Claude) | F1 0.254 (chance) | **F1 0.825, precision 1.000** |
| Base Qwen3-8B | F1 0.000 (won't answer) | **F1 0.531** |
| Fine-tuned Qwen3-8B | F1 0.453 | — |

- The frontier goes from chance to **the best detector in the entire project** once given context.
  Perfect precision — it made zero false alarms on the 150-window sample.
- Even the un-fine-tuned base with RAG (0.531) **beats the fine-tuned model** (0.453).

**What this means:**

1. **The diagnosis was correct.** The frontier wasn't incapable — it was missing information. Give it
   the channel's history and it knows exactly what to do.
2. **Retrieval substitutes for training** for this task. Both approaches use the same 21k training
   windows; one encodes them in weights, the other retrieves them at runtime. For detection, retrieval
   works as well or better.
3. **Trade-offs remain.** RAG requires a retrieval index and (for the frontier) an API call per window.
   If you need to own the model, run offline, or avoid per-call costs, fine-tuning is still the right
   choice. If you're okay with an API and want to skip training, RAG works.

**So which should you use?**
- **Fine-tuning** if: you need a sovereign/offline model, want to avoid vendor lock-in, or are
  processing high-volume streams where API costs add up.
- **RAG** if: you're okay with API calls, want to avoid training entirely, or want the best possible
  detection (Frontier+RAG is the top performer).

Both beat the naive prompt; they're just different ways to give the model the context it needs.

---

## 8. Why each big decision was made (so you can defend it)

| Decision | Why |
|---|---|
| **ESA-AD over NASA MSL/SMAP** | Newer, expert-annotated, has its own official benchmark; NASA set has known label-quality criticism. |
| **Qwen3-8B** | Strong reasoning/math scores, well-supported by Unsloth, 8B fits a laptop after quantization. |
| **QLoRA, not full fine-tuning** | Trains <1% of weights in 4-bit → a few dollars on one rented GPU instead of a cluster. |
| **Cloud for training, laptop for inference** | Training needs NVIDIA/CUDA; inference runs fine on the Mac's Metal GPU — matches the "own the model, no API" goal and the hardware budget. |
| **GGUF + Metal locally** | Lets the fine-tuned model run on the MacBook with no vendor dependency. |
| **Build *all* approaches, not just the LLM** | Makes the recommendation empirical rather than asserted. |
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
> trivial baseline so the answer survives a skeptic. The transferable findings: (1) no *single* LLM
> out-detects a tuned LSTM, matching the literature; (2) but a fine-tuned 8B beats both a few-shot
> base and a frontier model, because it learned localized priors prompting can't supply; (3) an LLM
> detector's "over-flagging" can be a *calibration* artifact — reading its confidence properly, not
> changing the model, recovers most of the gap; (4) fusing detectors that make *independent* errors
> beats every one of them; and (5) LLM "explainability" is only as good as the detector feeding it —
> which is why the deployable design is a fused detector plus an LLM advisor.

It's not a new algorithm; it's a **rigorous applied result with an honest headline** — the kind of
engineering maturity that matters in mission-critical work.

---

## 10. FAQ

**Q: Did the LLM beat the traditional baseline?**
A: No *single* LLM out-detects the tuned LSTM (LSTM F1 0.553 / precision 0.837) — the expected result,
matching the literature. But two caveats matter: (a) the text LLM's precision looks far worse than it
is — read its confidence properly and it jumps from 0.36 to 0.84 (§7d); and (b) the actual best
detector is neither the LSTM nor any LLM alone, it's the three of them **fused** (precision 0.922,
§7e). And the LLM uniquely adds high-quality advice the baselines can't. So the right design is a
fused detector feeding the LLM advisor.

**Q: A precision of 0.36 sounds hopeless for the text detector. Is it?**
A: That 0.36 is an artifact of how we read the model, not the model's ability. It was decoded with
randomness on (which flips some answers) and we took its bare yes/no instead of its confidence. Turn
off the randomness and read the confidence with a sensible cut-off, and precision more than doubles to
0.84 — competitive with the LSTM. The model knew what was anomalous; we were reading its answer badly
(§7d).

**Q: So what's the single best detector overall?**
A: The **ensemble** — a small "judge" model that fuses the confidence scores of the text LLM, the
vision LLM, and the LSTM. Because the three make independent mistakes, the fusion beats all of them:
precision 0.922, the best false-alarm-aware score in the project (§7e). Measured on the windows where
all three have a score (a fair head-to-head).

**Q: Then why fine-tune at all — why not just prompt Claude/GPT?**
A: We tested exactly that. A frontier model prompted zero- *and* few-shot scored *below* the dumb
baseline on this input — **but not because it's incapable**. The problem was missing *context*: the
model had no idea what's normal for each channel. When we gave it that context via RAG (k=5 retrieved
training neighbors), it jumped to F1 0.825 — the best detector in the project. So the choice is:
- **Fine-tune** if you need to own the model, run offline, or avoid per-call API costs.
- **RAG** if you're okay with an API and want to skip training.
Both work; fine-tuning still gave us a model we *own*, that runs locally with no API, at 3× the speed —
which was the project's stated goal.

**Q: Your few-shot base got F1 0.420, almost matching your 0.453. Doesn't that undermine the
fine-tune?**
A: No — that 0.420 is a mirage. It comes from flagging ~80% of windows (precision 0.282), barely
above flag-everything. It's over-flagging, not detecting. The fine-tune hits a similar F1 with a
*balanced* trade-off, plus reliable advice (99.6% vs 13%) and 3× lower latency.

**Q: Isn't comparing a 3-channel LSTM to a 4,500-window LLM unfair?**
A: It was early on — which is why we re-ran the LSTM on all 58 channels (its honest F1 fell, and we
kept the lower number) and later tuned its threshold to its final precision of 0.837. It still leads
the single detectors. Scoring everything on one identical continuous stream is the remaining
refinement (the ensemble already does this on a fair shared subset).

**Q: How do you know the advice is actually correct, not just well-formatted?**
A: Phase 9 graded a sample on a transparent rubric. On correct flags it's 5.58/6 (95% high-quality,
right channel and magnitude). On false alarms it's ~1/6 — it explains a non-existent anomaly. So
advice quality is gated by detection precision, which is the argument for the hybrid.

**Q: Wait — the grader was an LLM (Claude). Isn't that circular? And if Claude can grade it, why not
just use Claude instead of the fine-tuned model?**
A: Two different jobs. The grader did **not** use its own knowledge as the answer key, and did **not**
just compare against the synthetic Phase-1.5 labels (it couldn't — the saved predictions don't carry
the keys to join to them). It scored against **the window's actual numbers and the known ground-truth
label**: does the named channel match the window, is the stated magnitude consistent, and — since we
*know* which windows were truly anomalous — advice about a truly-normal window scores zero by
construction. So grading is *checking finished advice with the answer key in hand*, which is a far
easier task than **producing** detections from scratch. As a *detector*, the same frontier model sat
at chance (§7b) — it can't do the actual job. And even if a frontier could write the advice, you'd
still be sending your telemetry to an outside vendor, paying per call, and waiting on its latency —
the very things owning the model avoids.

**Q: What was the hardest / most important engineering part?**
A: Designing the comparison so the "did fine-tuning help?" claim actually holds up. It's easy to show
a fine-tuned model scoring some number; it's much harder to *prove the fine-tuning is what helped*.
That meant: running the un-fine-tuned base **through the identical harness** (same prompt, decoding,
parser — so only the weights differ); adding a **few-shot** base and a **frontier** model so
"couldn't you just prompt instead?" is answered, not hand-waved; and — the key insight — anchoring
everything to a **trivial always-anomaly baseline** so a high-F1-by-over-flagging result can't
masquerade as real detection. That control design is what turns the result from a demo into evidence,
and it's the part a domain expert will scrutinize hardest. (The supporting infra — making the
multi-hour eval resumable/observable so a crash costs one checkpoint, not the whole run — mattered
too, but was quick to add.)

**Q: What would you do next, and why?**
A: The detection-side hardening (tuning the LSTM, calibrating the text LLM, the vision control, the
fused ensemble, and the RAG comparison) is already **done** — it's woven into §7. What remains, in
priority order:
- **Ship the hybrid (fused detector → LLM advisor).** It's the design the numbers support; everything
  else is refinement. *Low effort.*
- **Improve the advice toward production-grade.** Replace the synthetic advice labels with
  **human-expert-written** advice and add **retrieval grounding** so the model cites real fault
  catalogs, not templates — the biggest lever on whether the advice is trustworthy for mission-critical
  use. *High effort, high impact.*
- **Cover all missions and one common yardstick.** We only trained the LSTM on Mission 1, so the
  three-model fusion is Mission-1-only; training the other missions' LSTMs and scoring everything on
  one identical continuous stream would make every number directly comparable. *~1–2 days + a few
  hours of training.*
- **Fine-tune a frontier model** (GPT-4o/Gemini tuning APIs) — the remaining untested cell in the
  "own vs. rent" comparison. RAG showed the frontier *can* detect with context; fine-tuning it would
  test whether you can have both frontier capability *and* offline/sovereign deployment. Rare use case,
  but completeness. *Medium.*
- **Robustness sweeps:** does it generalize to a spacecraft it never trained on, and does the 1-hour
  averaging hide short anomalies? Both are untested. *Medium.*

---

## 11. Glossary (quick reference)

- **Anomaly** — a segment of telemetry that deviates from normal behavior.
- **Channel** — one sensor's time series (e.g. one voltage line).
- **Subsystem** — the functional group a channel belongs to (power, thermal, attitude, propulsion,
  comms); e.g. a battery-voltage channel is in the *power* subsystem.
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
- **SFT (Supervised Fine-Tuning)** — fine-tuning on input→output example pairs (what we did:
  window → verdict + advice).
- **RAG (retrieval-augmented generation)** — fetching relevant reference material (e.g. a channel's
  history or fault catalog) into the prompt at run time so the model has context it wasn't trained on.
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
- **TP:FP ratio** — true-positive flags to false-positive flags; preserving it in a graded sample
  keeps the real false-alarm rate (here ~43:77, reflecting ~0.36 precision).
- **Precision** — fraction of flags that were real (alarm accuracy).
- **Recall** — fraction of real anomalies caught (miss rate's complement).
- **Calibrated decision threshold / P–R curve** — turning the model's score into a tunable cutoff to
  trade recall for precision; the P–R curve plots that trade-off across all cutoffs (one model = many
  operating points).
- **F1** — balanced precision+recall score.
- **CEF0.5** — precision-weighted F-score; the cost-of-false-alarm-aware metric.
- **Affinity-F1** — interval-aware F-score for continuous streams.
- **AUC-PR** — a threshold-free score of how well a model's *confidence* ranks anomalies above normal
  windows (area under the precision-recall curve); separates "does it know?" from "did we pick a good
  alarm cut-off?".
- **Calibration** — turning a model's raw output into an alarm well: e.g. reading its confidence and
  choosing a sensible cut-off, instead of taking its bare yes/no. Fixed the text LLM's apparent
  over-flagging (§7d).
- **Trivial / always-anomaly baseline** — flag everything; the "free" score every detector must
  beat (F1 ≈ 0.40 here).
- **Hybrid** — the recommended design: a high-precision detector triggers an LLM advisor.
- **Ensemble / score fusion** — combining several detectors' *confidence scores* into one. Hard rules
  ("both must agree", "either fires") are the crude version; a learned **stacker** does better.
- **Stacker** — a small model (here a logistic regression) that learns how much to trust each
  detector's score and fuses them into one — the engine of the winning ensemble (§7e).
- **Telemanom** — NASA JPL's LSTM anomaly-detection method (our LSTM follows it).
- **AnomLLM / AnomSeer / Time-LLM** — the research line on LLMs detecting time-series anomalies
  (text, vision, and patching approaches respectively).
- **ESA-AD / ESA-ADB** — the European Space Agency anomaly dataset and its official benchmark
  pipeline.
- **CEF / point-adjustment** — ESA's preferred metric / a discredited metric we avoided.
