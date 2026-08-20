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

Most recent first. The [papers page](papers.md) gives each one a one-line summary with
the numbers a click away.

- [Extracting accent features in spoken Brazilian Portuguese without sociolinguistic
  labels](https://arxiv.org/abs/2605.30457) (SBrT 2026)
- [Broadcast-quality synthetic narration](https://aes2.org/publications/elibrary-page/?id=23020)
  (AES AIMLA 2025)
- [Neutral TTS Female Voice Corpus in Brazilian
  Portuguese](https://biblioteca.sbrt.org.br/articlefile/4464.pdf) (SBrT 2023)
- [A Corpus of Neutral Voice Speech in Brazilian
  Portuguese](https://link.springer.com/chapter/10.1007/978-3-030-98305-5_32) (PROPOR 2022)

Earlier, and on a different problem: [Blind Source Separation from Music Recordings
through Deep Neural Networks](http://www.repositorio.poli.ufrj.br/monografias/projpoli10034948.pdf),
my graduation project (Escola Politécnica/UFRJ, 2021).

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
