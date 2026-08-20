# Papers

Publications and the dissertation, most recent first. Each entry gives the one-line
version; expand **What we found** for the numbers.

## Increasing the Robustness of Brazilian Portuguese Voice Synthesis

*Pedro H. L. Leite.* MSc dissertation, COPPE/UFRJ, October 2024.
Advisor: Luiz Wagner Pereira Biscainho.

What it takes to make Brazilian Portuguese TTS good enough to go on air. The short
answer: record 40 hours of studio speech, release it, and then fix — one at a time —
everything the models get wrong about the language.

[Dissertation (PDF)](https://w1files.solucaoatrio.net.br/atrio/ufrj-pee_upl//THESIS/10004572/122018208_20250213151626802.pdf)
· [Audio examples](master-thesis.md)

```{dropdown} What we found
**Data first.** Two studio corpora, one male and one female, about 20 hours each,
recorded at 24-bit/96 kHz with Neumann TLM 102/103 cardioid microphones at 20 cm, then
segmented into 10,333 matched utterances apiece. Both are [free to
download](datasets.md).

**Phonemes beat graphemes.** Training on phonemizer transcriptions, with stress marks
folded into single tokens, removed the open/closed vowel errors that plague homograph
heterophones — *gosto*, *forma*, *colher* — which grapheme models simply cannot
disambiguate.

**Let the vocoder do the upsampling.** Training text-to-mel at 22.05 kHz and letting the
vocoder rebuild 44.1/48 kHz was not just faster: prosody converged *better*, because the
acoustic model stopped spending capacity on noisy high-frequency content.

**Then control.** A multispeaker VITS over roughly 80 hours and 43 speakers, RVC-based
voice conversion for timbre transfer, emotion and style embeddings, and a web interface
for phoneme-level pitch and duration edits — the work that ultimately put synthetic
voices on air.
```

## Extracting accent features in spoken Brazilian Portuguese without sociolinguistic labels

*Pedro H. L. Leite, Pedro Benevenuto Valadares, Luiz W. P. Biscainho.*
XLIV Brazilian Symposium on Telecommunications and Signal Processing (SBrT 2026),
accepted. arXiv:2605.30457.

You don't need a big speaker embedding to hear an accent — you need to look at the right
few milliseconds. Localized phonological features, 14 to 20 dimensions wide, beat
768- and 1024-dimensional whole-utterance self-supervised embeddings at telling Brazilian
accents apart.

[arXiv](https://arxiv.org/abs/2605.30457)
· [Interactive companion page](https://gpa-smt-ufrj.github.io/accent-features)

```{dropdown} What we found
Speaker accuracy on three phonological contrasts, under balanced stratified
cross-validation over 40, 51 and 20 annotated speakers respectively — the proposed
features against the best whole-utterance embedding of any model tested:

| Task | Contrast | Localized features | Best SSL embedding |
|---|---|---|---|
| /s/ coda | *chiado* [ʃ] vs. sibilant [s] | 1.00 | 0.85 |
| /r/ coda | *carioca* vs. tap vs. *caipira* | 0.85 | 0.57 |
| /d/, /t/ | palatalized vs. not | 0.88 | 0.73 |

The gap is the point. HuBERT, wav2vec 2.0, XLSR-53, XLSR-PT, ECAPA-TDNN and Resemblyzer
all pool across the whole utterance, which averages the accent away. Gate the features to
the phones where the contrast actually lives and 20 dimensions are enough.

The /r/ coda is the hard case: alveolar taps sit acoustically between the uvular and
retroflex extremes, so the tap class carries the lowest recall, 0.71.

None of this requires sociolinguistic labels on the speakers.
```

## Broadcast-quality synthetic narration: a workflow for fine-grained text-to-speech intonation and emotion control

*Luiz Fernando Kruszielski, Pedro H. L. Leite, Myllene P. Fernandes, Andre Pereira,
Luiz W. P. Biscainho.* AES International Conference on Artificial Intelligence and
Machine Learning for Audio (AIMLA 2025), London, September 2025.

Synthetic narration that listeners sometimes prefer to a professionally recorded
voice-over — a production workflow where the director still gets to direct, with emotion
chosen per passage and pitch and duration adjustable word by word.

[AES e-Library](https://aes2.org/publications/elibrary-page/?id=23020)

```{dropdown} What we found
**The method.** One-hot emotional embeddings over application-aware, manually labelled
emotion classes, on a fine-tuned VITS2 synthesizer. The pipeline combines manual
annotation, automatic emotion classification, and phoneme- and word-level pitch and
duration adjustment.

**The result.** The synthetic narration reached parity with professionally recorded
voice-overs, with some listener preference for the synthetic take — precisely because it
can be tuned after the fact.

**Why it matters.** Fine-grained control is what moves TTS from intelligible to
broadcastable. A voice-over artist can be asked for another read; until you can edit
intonation at the word level, a synthetic voice cannot.
```

## Neutral TTS Female Voice Corpus in Brazilian Portuguese

*Pedro H. L. Leite, Edmundo Hoyle, Álvaro Antelo, Luiz F. Kruszielski,
Luiz W. P. Biscainho.* XLI Brazilian Symposium on Telecommunications and Signal
Processing (SBrT 2023), São José dos Campos, October 2023.

A matching 20-hour female corpus, and evidence that transferring a voice across genders
costs you. Speech production is anatomy, and the models learn that implicitly — so we
built the female counterpart to our male corpus and measured what happens when you cross
the two.

[Paper (PDF)](https://biblioteca.sbrt.org.br/articlefile/4464.pdf)
· [Download the corpus](https://www.kaggle.com/datasets/mediatechlab/g-neutral-speech-female)
· [Companion page](https://gpa-smt-ufrj.github.io/sbrt2023)

```{dropdown} What we found
Two base models (Tacotron 2 with Multiband-MelGAN) trained from about 20 hours each, then
fine-tuned onto roughly 75-minute target voices from CETUC, in every male/female
combination. Intelligibility, as word error rate against Whisper transcripts:

| Target voice | From male base | From female base | Ground truth |
|---|---|---|---|
| Male | 12.5 % | 29.8 % | 4.8 % |
| Female | 8.0 % | 8.1 % | 1.8 % |

**Same-gender transfer wins, asymmetrically.** The male target degrades badly when warm
started from a female model — more than double the error rate. The female target is
roughly tied either way on WER, but same-gender transfer still yields visibly sharper F0
contours and energy distribution in the mel spectrograms.

**Speaker similarity agrees.** d-vector cosine similarity against the mean ground-truth
embedding is consistently higher for same-gender transfers.

The practical consequence: a gender-balanced pool of base models is not a nicety but a
prerequisite, which is why this corpus exists.
```

## A Corpus of Neutral Voice Speech in Brazilian Portuguese

*Pedro H. L. Leite, Edmundo Hoyle, Álvaro Antelo, Luiz F. Kruszielski,
Luiz W. P. Biscainho.* 15th International Conference on Computational Processing of
Portuguese (PROPOR 2022), Fortaleza, March 2022. LNCS, pp. 344–352.

The 20-hour male neutral corpus, and proof that minutes of target audio are enough on top
of it. A neutral, professionally recorded base voice is a warm start: learn the language
once, then spend your scarce target-speaker data on timbre alone.

[Paper (Springer)](https://link.springer.com/chapter/10.1007/978-3-030-98305-5_32)
· [Download the corpus](https://www.kaggle.com/datasets/mediatechlab/gneutralspeech)
· [Live TTS demo](https://www.kaggle.com/code/pedrohlopes/portuguese-tts)

```{dropdown} What we found
**The base voice.** Tacotron 2 with WaveGlow, 102k iterations over about six days on an
NVIDIA Quadro RTX 8000. Prosody and accent came out natural but neutral — exactly the
intended starting point.

**Then the payoff.** From that checkpoint, under eight minutes of a target speaker's
audio converged in hours to a voice clearly identifiable as theirs. Seventy-five minutes
from CETUC did it noticeably better: fewer artifacts, better pronunciations.

**The lesson that made it work at all.** The first attempts produced fluent nonsense —
the model never learned text-audio alignment. The cause was leading silence, non-uniform
across the corpus, so the model could not predict when speech begins. Trimming it with
voice activity detection turned a failing pipeline into a working TTS system. It is the
least glamorous and most reusable finding in the paper.

Audio for all three experiments is in the [complementary material](master-thesis.md).
```
