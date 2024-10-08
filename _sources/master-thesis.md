---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Complementary material for Master Thesis

During my research over TTS systems, some experiments were conducted when training and developing the AI systems for syntesizing robust and natural speech. Here, we provide a detailed log with audio examples for such experiments, hoping to clarify even more the documentation provided in my master thesis. 

## Experiments @ PROPOR 2022
All experiments were conducted using the tacotron2 and waveglow models, with few modifications to adopt portuguese characters.

To start training the neutral voice, the audio samples in our dataset were converted to 16-bit / 22.05 kHz PCM in order to speed up the processing stages. After that, the initial silence in each audio file was trimmed according to a 40 dB threshold below the maximum signal magnitude. The input sentences used were the raw normalized text transcripts.

+++

### Experiment #1
The first experiment consisted in training a neutral voice directly for our data, aiming to develop a robust voice over structure understanding and natural prosody. The Tacotron 2 model was then trained in an NVIDIA QUADRO RTX 8000 with 48 GB RAM. With this GPU, the text-to-mel model training stage took approximately 6 full days in 102 k iterations with 32 batch size. For Waveglow, we used the pre-trained model from the English language available from the original repository, speeding up training to about two days of processing in 38 k iterations with 80 batch size on the same GPU. Here are the examples for the base neural voice:

**Text**: A operação desta quarta nasceu de uma cooperação entre autoridades brasileiras e americanas.

**Original audio signal**:

```{code-cell} ipython3
:tags: ["remove_input"]
import IPython.display as ipd
ipd.Audio('audios/orig_base.wav')
```

**Synthetized audio signal**:

```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/synth_base.wav')
#print('Synthetized Audio')
```



### Experiment #2
With the neutral voice already trained, two other experiments were conducted, setting aside a small percentage of the data for comparison. In the first one, we use a proprietary dataset with less than 8 minutes of audio from a male speaker to perform a voice transfer experiment. In this case, both the Tacotron 2 and the Waveglow models started from the pre-trained checkpoints and converged in a few hours. Here are some examples:

**Text**: Nossa filha é a primeira aluna da classe.

**Original audio signal**:

```{code-cell} ipython3
:tags: ["remove_input"]
import IPython.display as ipd
ipd.Audio('audios/orig_lw.wav')
```

**Synthetized audio signal**:

```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/synth_lw.wav')
#print('Synthetized Audio')
```

### Experiment #3
The third experiment resorted to a larger dataset (around 75 minutes) collected from the CETUC
dataset to perform a similar voice transfer procedure. The models took around a day to reach convergence, but the voice obtained seems more natural, with reduced synthesis artifacts and better pronunciations. Listen:

**Text**: É um dos eventos mais importantes do país este ano.

**Original audio signal**:

```{code-cell} ipython3
:tags: ["remove_input"]
import IPython.display as ipd
ipd.Audio('audios/orig_alcaim.wav')
```

**Synthetized audio signal**:

```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/synth_alcaim.wav')
#print('Synthetized Audio')
```

The dataset made available in this work can be downloaded [here](https://www.kaggle.com/datasets/mediatechlab/gneutralspeech), and a [live demo](https://www.kaggle.com/code/pedrohlopes/portuguese-tts) for TTS with the base voice is also available.

## Tensorflow's TensorSpeech
The first models we developed, as seen here, were already capable of speaking naturally the sentences and to not sound like robots, but the final quality of the samples was not great. This was improved with the change of implementation for tacotron2, using the Tensorspeech open-source code on github. Here are some examples for the trained voices with this new codebase:



**Synthetized audio signal**:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/tensorspeech1.wav')
```

**Text**: A operação desta quarta nasceu de uma cooperação entre autoridades brasileiras e americanas.



A live demo for syntesizing speech with this trained voice can be accessed [here](https://www.kaggle.com/code/pedrohlopes/portuguese-tts-tensorflowtts-better-quality). We can hear that the overall quality improved a lot for this voice, but the phone ambiguities still confuse some pronunciations. It is also important to note that the significant improvement in quality is also due to the vocoder upsampling stage, which well separated the mel spectrogram training stage from the waveform synthesis.


## SBRT 2023 Gender-aware voice transfer

In this experiment, we aimed to measure the impact of gender-balanced data and the importance of transferring data from the same gender as the target voice. For this, we started by training the same neutral voice from the first experiment and the newly developed female voice dataset, with the exact same 10333 phrases as described in the [article](https://biblioteca.sbrt.org.br/articlefile/4464.pdf). From these pretrained models, we proceeded to a voice transfer experiment, where the female and male models were finetuned with other male and female voices, but with smaller amounts of data. Here are some examples of the resulting voices, from unseen text data.

### Reference audio signals:

**Text**: A melhoria da qualidade não interessa só a técnicos ou empresários.

**Target voice - Male**

```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/sbrt_data/elson-original.wav')
```
**Target voice - Female**

```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/sbrt_data/gabriela-original.wav')
```


### Synthesized audio signals:
**Text**: A melhoria da qualidade não interessa só a técnicos ou empresários.

**Target voice - Male; Base model - Male**

```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/sbrt_data/elson-from-male.wav')
```

**Target voice - Male; Base model - Female**

```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/sbrt_data/elson-from-female.wav')
```

**Target voice - Female; Base model - Male**

```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/sbrt_data/gabriela-from-male.wav')
```

**Target voice - Female; Base model - Female**

```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/sbrt_data/gabriela-from-female.wav')
```

## Phoneme transcriptions

After the gender aware experiment, we proceeded to increase the robustness for pronouncing homograph heterophonic words. To do so, we used phoneme transcriptions to train the models instead of the raw characters. Here are some sentences with ambiguous words, and the audio samples for the phoneme and grapheme transcriptions:

**Text**: Eu **gosto** de coisas diferentes. Meu **gosto** é bem peculiar.

Straight from graphemes:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/grafemas.wav')
```

With phonemes:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/fonemas.wav')
```

---

**Text**: O pão de **forma** é bonito. Ele tem a **forma** de um quadrado.

Straight from graphemes:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/grafemas2.wav')
```



With phonemes:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/fonemas2.wav')
```

---

**Text**: Vou **colher** todos os frutos nesse verão, e vou comer tudo com uma **colher**.

Straight from graphemes:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/grafemas3.wav')
```



With phonemes:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/fonemas3.wav')
```


## Voice conversion and Multispeaker TTS
We also conducted some experiments regarding voice conversion and multispeaker TTS using speaker identity controls through embeddings generated by a pretrained speaker encoder in a speaker verification task. We show here three reference speakers and the voice converted samples between them:

### Reference audio signals:
Speaker 1:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/voz_03_22.wav')
```
Speaker 2:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/voz_04_22.wav')
```
Speaker 3:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/voz_30_22.wav')
```

### Voice conversion samples:
Speaker 1 to Speaker 2:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/03-04.wav')
```
Speaker 1 to Speaker 3:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/03-30.wav')
```
Speaker 2 to Speaker 1:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/04-03.wav')
```
Speaker 2 to Speaker 3:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/04-30.wav')
```

Speaker 3 to Speaker 1:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/30-03.wav')
```
Speaker 3 to Speaker 2:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/30-04.wav')
```

### Multispeaker TTS samples:
**Text**: Dólar fecha em queda e vai a cinco e quarenta e seis, após ata do Copom.

Speaker 1:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/03-TTS.wav')
```
Speaker 2:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/04-TTS.wav')
```
Speaker 3:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/30-TTS.wav')
```

## Emotion and style control
In this experiment, we aimed to control the emotion and style of the synthesized speech. To do so, we used the same embedding architecture from the previous experiments, but as controls for emotion and style of the speech. Here are some style control examples for the same voice:

### Style control samples:
**Text**: Dólar fecha em queda e vai a cinco e quarenta e seis, após ata do Copom.

No label:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/dirceu-normal.wav')
```
Label - Excited:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/dirceu-empolgado.wav')
```
Label - Sad:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/dirceu-triste.wav')
```
Label - Serious:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/dirceu-serio.wav')
```
Label - Narrative:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/dirceu-narrativo.wav')
```

## Phoneme-level controls
The last experiment in this complementary material shows the phoneme-level controls for the synthesized speech. Here are some examples for the same text being synthesized with different phoneme controls:

**Text**: Este é um exemplo para uma mudança na minha forma de ler o texto.

No controls:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/este_neutro.wav')
```
Flattened pitch curve:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/este_monotono.wav')
```
Increased pitch at the first word and forced pauses:
```{code-cell} ipython3
:tags: ["remove_input"]
ipd.Audio('audios/este_pausado.wav')
```
