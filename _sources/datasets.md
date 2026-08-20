# Datasets

Two single-speaker Brazilian Portuguese speech corpora, recorded in a professional studio
under one protocol: same sentences, same room, same distance, one male voice and one
female voice. They were built to be *base voices* — enough neutral, high-quality speech to
learn the language from, so that adapting to a new speaker only has to learn that
speaker.

Both are free to download from Kaggle.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🎙️ Neutral Speech — Male
{bdg-success}`~20 hours` {bdg-light}`10,333 segments` {bdg-light}`1 speaker`
^^^
Introduced at **PROPOR 2022**. Recorded with a Neumann TLM 102.

```{button-link} https://www.kaggle.com/datasets/mediatechlab/gneutralspeech
:color: success
:expand:
💾 Download on Kaggle
```
```{button-link} https://www.kaggle.com/code/pedrohlopes/portuguese-tts
:color: secondary
:outline:
:expand:
▶️ Live TTS demo
```
:::

:::{grid-item-card} 🎙️ Neutral Speech — Female
{bdg-success}`~20 hours` {bdg-light}`10,333 segments` {bdg-light}`1 speaker`
^^^
Introduced at **SBrT 2023**. Recorded with a Neumann TLM 103.

```{button-link} https://www.kaggle.com/datasets/mediatechlab/g-neutral-speech-female
:color: success
:expand:
💾 Download on Kaggle
```
```{button-link} https://gpa-smt-ufrj.github.io/sbrt2023
:color: secondary
:outline:
:expand:
🔊 Companion page
```
:::

::::

## Recording specification

Both corpora share the same specification, which is what makes them usable as a matched
pair for gender-aware experiments.

| | |
|---|---|
| **Speakers** | One male, one female (separate datasets) |
| **Duration** | ~20 hours each |
| **Segments** | 10,333 utterances each, matched sentence for sentence |
| **Style** | Neutral emotion; regionalisms and prosody bias deliberately avoided |
| **Microphone** | Neumann TLM 102 (male) / TLM 103 (female), cardioid, 20 cm from the speaker |
| **Captured** | Broadcast `.wav`, 24-bit / 96 kHz |
| **Distributed** | `.wav`, 16-bit / 44.1 kHz, each with a matching `.txt` transcript |
| **Environment** | Professional studio; negligible noise floor, no audible accidental noise |

The 20 cm distance and cardioid pattern were chosen to avoid proximity effect and room
colouration respectively. 96 kHz is well beyond what TTS training needs today — the point
was to capture the material once, at a resolution that would not become the limiting
factor for later research.

## How the corpora were built

**Text.** Short sentences scraped from the website of *Jornal Nacional*, one of Brazil's
main TV news programmes, covering roughly two months of articles from May–June 2021.
Advertisements, figures and footnotes were stripped.

**Normalization by hand.** Scraped news text is full of things that break the
correspondence between spelling and pronunciation, so numerals, abbreviations and foreign
names were rewritten as they would be read aloud: `2020` → *dois mil e vinte*, `TV` →
*tê vê*, `Biden` → *Baiden*.

**Segmentation.** Text was split at commas and periods; audio was cut to match using
[Aeneas](https://github.com/readbeyond/aeneas) forced alignment, targeting 5–10 second
segments and avoiding anything over 20 seconds. Shorter segments let more examples fit in
a batch without allocation spikes, which makes batch statistics more reliable.

**Cleanup.** [py-webrtcvad](https://github.com/wiseman/py-webrtcvad) voice activity
detection removed breaths and corrected bad cuts. Uniform leading silence matters more
than it sounds: non-uniform silence was what prevented the first models from learning
text-audio alignment at all.

## How they compare

Where these corpora sit among the datasets commonly used for Brazilian Portuguese speech,
alongside the English reference points:

| Dataset | Lang. | Hours | Speakers | kHz | Recording |
|---|---|---|---|---|---|
| **Ours (male)** | BR | 20 | 1 m | 96 | Professional, controlled |
| **Ours (female)** | BR | 20 | 1 f | 96 | Professional, controlled |
| TTS-Portuguese | BR | 10.5 | 1 m | 48 | Professional, controlled |
| CETUC | BR | 144 | 50 m, 50 f | 16 | Professional, controlled |
| CML-TTS | BR | 70 | 31 m, 17 f | 24 | Professional, heterogeneous |
| Multilingual LibriSpeech | BR | 200 | 50 | 16 | Professional, heterogeneous |
| Mozilla Common Voice | BR | 200 | 2038 | 48 | Uncontrolled, heterogeneous |
| LJSpeech | US | 20 | 1 f | 22.05 | Professional, controlled |
| VCTK | US | 44 | 109 | 96 | Professional, controlled |

The gap these fill is narrow but specific: **single-speaker, professionally recorded,
long enough to pre-train on, and gender-balanced.** Larger Brazilian Portuguese corpora
exist, but they spread their hours across many speakers or many recording conditions —
neither of which gives you a clean base voice.

## Citing

If you use either corpus, please cite the corresponding paper — see the
[papers page](papers.md) for full references:

- **Male voice** — Leite et al., *A Corpus of Neutral Voice Speech in Brazilian
  Portuguese*, PROPOR 2022.
- **Female voice** — Leite et al., *Neutral TTS Female Voice Corpus in Brazilian
  Portuguese*, SBrT 2023.
