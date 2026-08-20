# Pedro Leite Research Hub

I have a Master's in Electrical Engineering and I'm currently doing my PhD at COPPE/UFRJ. I work with the Audio Processing Group (GPA), focusing on speech synthesis and accent modelling for Brazilian Portuguese.

This portal is a collection of my research, for easy and public access.

## Dissertation

**Increasing the Robustness of Brazilian Portuguese Voice Synthesis.**
COPPE/UFRJ, October 2024. Advisor: Luiz Wagner Pereira Biscainho.

About what it takes to make Brazilian Portuguese TTS good enough to be production-ready/actually useful.

[Read the dissertation (PDF)](https://w1files.solucaoatrio.net.br/atrio/ufrj-pee_upl//THESIS/10004572/122018208_20250213151626802.pdf)
· [Summary](research.md)

## Papers

Most recent first. The [research page](research.md) gives each one a one-line summary
with the numbers a click away.

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

Two professionally recorded single-speaker corpora for Brazilian Portuguese TTS: same
sentences, same protocol, one male voice and one female voice, about 20 hours and 10,333
segments each. Both are free to download.

- [Neutral Speech, male voice](https://www.kaggle.com/datasets/mediatechlab/gneutralspeech),
  introduced at PROPOR 2022.
- [Neutral Speech, female voice](https://www.kaggle.com/datasets/mediatechlab/g-neutral-speech-female),
  introduced at SBrT 2023.

The [datasets page](datasets.md) has the full recording specification, how the corpora
were built, and how they compare with the other Brazilian Portuguese speech datasets.

## Complementary material

The [material for the master thesis](master-thesis.md) walks through every experiment in
the dissertation with the audio to match: the base neutral voice, voice transfer from
eight minutes of target speech, the gender-aware transfer comparison,
grapheme-versus-phoneme pronunciations of ambiguous words like *gosto*, *forma* and
*colher*, and the voice conversion, multispeaker and prosody-control samples.

The [material for SBrT 2026](accent-features.md) covers the accent-features paper: audio
clips and spectrograms for each accent class, and the full ablation grid.
