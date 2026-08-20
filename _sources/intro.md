# Pedro Leite — Research Hub

MSc in Electrical Engineering (COPPE/UFRJ), Audio Processing Group (GPA) — speech
synthesis, voice datasets and accent modelling for Brazilian Portuguese.

Brazilian Portuguese is among the most widely spoken languages in the world and one of
the worst served by open speech data. Most of the work collected here follows from that:
record the data properly, release it, then write down what it actually takes to train on
it. This portal gathers the dissertation, the papers, the corpora we published, and the
audio examples that go with them — because for speech work, listening is the argument.

## Dissertation

::::{card} Increasing the Robustness of Brazilian Portuguese Voice Synthesis
:link: papers
:link-type: doc
:class-card: sd-shadow-sm

**COPPE/UFRJ · October 2024 · Advisor: Luiz Wagner Pereira Biscainho**
^^^
What it takes to make Brazilian Portuguese TTS good enough to go on air — from recording
40 hours of studio speech, through the phoneme-level fixes for the pronunciations the
models keep getting wrong, to prosody, emotion and speaker-identity controls on top.
+++
{bdg-primary}`102 pages` {bdg-secondary}`TTS` {bdg-secondary}`pt-BR` — read the TL;DR →
::::

```{button-link} https://w1files.solucaoatrio.net.br/atrio/ufrj-pee_upl//THESIS/10004572/122018208_20250213151626802.pdf
:color: primary
:expand:
📄 Read the full dissertation (PDF)
```

## Papers

Five publications, most recent first. Each card on the [papers page](papers.md) opens with
a one-line hook and hides the numbers one click away.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Extracting accent features without sociolinguistic labels
:link: papers
:link-type: doc

{bdg-info}`SBrT 2026`
^^^
You don't need a big speaker embedding to hear an accent — you need to look at the right
few milliseconds.
:::

:::{grid-item-card} Broadcast-quality synthetic narration
:link: papers
:link-type: doc

{bdg-info}`AES AIMLA 2025`
^^^
Synthetic narration that listeners sometimes prefer to a professionally recorded
voice-over.
:::

:::{grid-item-card} Neutral TTS Female Voice Corpus in Brazilian Portuguese
:link: papers
:link-type: doc

{bdg-info}`SBrT 2023`
^^^
A matching 20-hour female corpus — and evidence that transferring a voice across genders
costs you.
:::

:::{grid-item-card} A Corpus of Neutral Voice Speech in Brazilian Portuguese
:link: papers
:link-type: doc

{bdg-info}`PROPOR 2022`
^^^
The 20-hour male neutral corpus, and proof that minutes of target audio are enough on top
of it.
:::

::::

## Datasets

Two professionally recorded, gender-balanced single-speaker corpora for Brazilian
Portuguese TTS — same sentences, same protocol, one male and one female voice. Both are
free to download.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🎙️ Neutral Speech — Male
:link: https://www.kaggle.com/datasets/mediatechlab/gneutralspeech

{bdg-success}`~20 h` {bdg-light}`10,333 segments`
^^^
Single male speaker, neutral emotion, studio-recorded at 24-bit/96 kHz. Introduced at
PROPOR 2022.
+++
Download on Kaggle →
:::

:::{grid-item-card} 🎙️ Neutral Speech — Female
:link: https://www.kaggle.com/datasets/mediatechlab/g-neutral-speech-female

{bdg-success}`~20 h` {bdg-light}`10,333 segments`
^^^
Single female speaker, same sentences and same protocol as the male corpus. Introduced at
SBrT 2023.
+++
Download on Kaggle →
:::

::::

See the [datasets page](datasets.md) for the full recording spec, how the corpora were
built, and how they compare with the other Brazilian Portuguese speech datasets.

## Listen

The [complementary material](master-thesis.md) walks through every experiment in the
dissertation with the audio to match — the base neutral voice, voice transfer from 8
minutes of target speech, the gender-aware transfer comparison, and grapheme-vs-phoneme
pronunciations of ambiguous words like *gosto*, *forma* and *colher*.
