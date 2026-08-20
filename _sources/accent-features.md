# SBrT 2026

Companion material for *Extracting accent features in spoken Brazilian Portuguese without
sociolinguistic labels* (Leite, Valadares and Biscainho, SBrT 2026, arXiv:2605.30457).

Where the thesis work was about generating speech, this one is about characterising it:
telling Brazilian accents apart from the acoustics alone, without any sociolinguistic
label attached to the speaker.

The material lives on its own site, built alongside the paper:

**[gpa-smt-ufrj.github.io/accent-features](https://gpa-smt-ufrj.github.io/accent-features)**

## The accent markers

Three phonological contrasts, chosen because each one splits Brazilian varieties along a
well-documented line:

- **/s/ in coda**: *chiado* [ʃ], as in much of Rio, against the sibilant [s].
- **/r/ in coda**: the *carioca* fricative, the alveolar tap, and the *caipira*
  retroflex.
- **/d/ and /t/ before [i]**: palatalized [dʒi] against non-palatalized [di].

## What is on the companion page

- Audio clips for every accent class, so each contrast can be heard rather than taken on
  trust.
- The matching spectrograms for each clip.
- The full ablation grid rendered interactively: every feature set against every model,
  per task.

## Results

Localized features 14 to 20 dimensions wide beat whole-utterance self-supervised
embeddings of 768 and 1024 dimensions on all three tasks. The summary table and the
discussion are on the [research page](research.md); the paper itself is on
[arXiv](https://arxiv.org/abs/2605.30457).
