# Pedro Leite — Research Hub

MSc in Electrical Engineering (COPPE/UFRJ), Audio Processing Group (GPA) — speech
synthesis, voice datasets and accent modelling for Brazilian Portuguese.

Brazilian Portuguese is among the most widely spoken languages in the world and one of
the worst served by open speech data. Most of the work collected here follows from that:
record the data properly, release it, then write down what it actually takes to train on
it. This portal gathers the dissertation, the papers, the corpora we published, and the
audio examples that go with them — because for speech work, listening is the argument.

## Dissertation

**Increasing the Robustness of Brazilian Portuguese Voice Synthesis.**
COPPE/UFRJ, October 2024. Advisor: Luiz Wagner Pereira Biscainho.

What it takes to make Brazilian Portuguese TTS good enough to go on air — from recording
40 hours of studio speech, through the phoneme-level fixes for the pronunciations the
models keep getting wrong, to prosody, emotion and speaker-identity controls on top.

[Read the dissertation (PDF)](https://w1files.solucaoatrio.net.br/atrio/ufrj-pee_upl//THESIS/10004572/122018208_20250213151626802.pdf)
· [Summary](papers.md)

## Papers

Five publications, most recent first. The [papers page](papers.md) gives each one a
one-line summary with the numbers a click away.

- **Extracting accent features in spoken Brazilian Portuguese without sociolinguistic
  labels** — SBrT 2026. You don't need a big speaker embedding to hear an accent; you
  need to look at the right few milliseconds.
- **Broadcast-quality synthetic narration** — AES AIMLA 2025. Synthetic narration that
  listeners sometimes prefer to a professionally recorded voice-over.
- **Neutral TTS Female Voice Corpus in Brazilian Portuguese** — SBrT 2023. A matching
  20-hour female corpus, and evidence that transferring a voice across genders costs you.
- **A Corpus of Neutral Voice Speech in Brazilian Portuguese** — PROPOR 2022. The
  20-hour male neutral corpus, and proof that minutes of target audio are enough on top
  of it.

## Datasets

Two professionally recorded single-speaker corpora for Brazilian Portuguese TTS — same
sentences, same protocol, one male voice and one female voice, about 20 hours and 10,333
segments each. Both are free to download.

- [Neutral Speech — Male](https://www.kaggle.com/datasets/mediatechlab/gneutralspeech),
  introduced at PROPOR 2022.
- [Neutral Speech — Female](https://www.kaggle.com/datasets/mediatechlab/g-neutral-speech-female),
  introduced at SBrT 2023.

The [datasets page](datasets.md) has the full recording specification, how the corpora
were built, and how they compare with the other Brazilian Portuguese speech datasets.

## Complementary material

The [complementary material](master-thesis.md) walks through every experiment in the
dissertation with the audio to match — the base neutral voice, voice transfer from eight
minutes of target speech, the gender-aware transfer comparison, grapheme-versus-phoneme
pronunciations of ambiguous words like *gosto*, *forma* and *colher*, and the voice
conversion, multispeaker and prosody-control samples.
