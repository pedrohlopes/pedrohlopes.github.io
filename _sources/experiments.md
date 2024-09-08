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

# Experiments @ PROPOR 2022
All experiments were conducted using the tacotron2 and waveglow models, with few modifications to adopt portuguese characters.

To start training the neutral voice, the audio samples in our dataset were converted to 16-bit / 22.05 kHz PCM in order to speed up the processing stages. After that, the initial silence in each audio file was trimmed according to a 40 dB threshold below the maximum signal magnitude. The input sentences used were the raw normalized text transcripts. We also present here the spectrogram comparison between original and synthetized samples, alongwith the corresponding text and audios.

+++

## Experiment #1
The first experiment consisted in training a neutral voice directly for our data, aiming to develop a robust voice over structure understanding and natural prosody. The Tacotron 2 model was then trained in an NVIDIA QUADRO RTX 8000 with 48 GB RAM. With this GPU, the text-to-mel model training stage took approximately 6 full days in 102 k iterations with 32 batch size. For Waveglow, we used the pre-trained model from the English language available from the original repository, speeding up training to about two days of processing in 38 k iterations with 80 batch size on the same GPU. 

**Spectrograms comparison**:

<!-- ![alt-text-1](figures/Orig_base_mod.png "Original sample") ![alt-text-2](figures/Synth_base_mod.png "Synthetized sample") -->
```{code-cell} ipython3
:tags: ["remove_input"]
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

%matplotlib inline

# figure size in inches optional
rcParams['figure.figsize'] = 16 ,10
rcParams['figure.dpi'] = 450

# read images
img_A = mpimg.imread('figures/Orig_base_mod.png')
img_B = mpimg.imread('figures/Synth_base_mod.png')

# display images
fig, ax = plt.subplots(1,2, gridspec_kw = {'wspace':0, 'hspace':0.5,'width_ratios': [1, 1.06]})

ax[0].axis('off')
ax[0].imshow(img_A)
ax[1].axis('off')
ax[1].imshow(img_B)
x = None
x
```

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



## Experiment #2
With the neutral voice already trained, two other experiments were conducted, setting aside a small percentage of the data for comparison. In the first one, we use a proprietary dataset with less than 8 minutes of audio from a male speaker to perform a voice transfer experiment. In this case, both the Tacotron 2 and the Waveglow models started from the pre-trained checkpoints and converged in a few hours.

**Spectrograms comparison**:

```{code-cell} ipython3
:tags: ["remove_input"]
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

%matplotlib inline

# figure size in inches optional
rcParams['figure.figsize'] = 16 ,10
rcParams['figure.dpi'] = 450

# read images
img_A = mpimg.imread('figures/real_orig_lw.png')
img_B = mpimg.imread('figures/real_synth_lw.png')

# display images
fig, ax = plt.subplots(1,2, gridspec_kw = {'wspace':0, 'hspace':0.5,'width_ratios': [1, 1.07]})

ax[0].axis('off')
ax[0].imshow(img_A)
ax[1].axis('off')
ax[1].imshow(img_B)
x = None
x
```

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


## Experiment #3
The third experiment resorted to a larger dataset (around 75 minutes) collected from the CETUC
dataset to perform a similar voice transfer procedure. The models took around a day to reach convergence, but the voice obtained seems more natural, with reduced synthesis artifacts and better pronunciations.

**Spectrograms comparison**:

```{code-cell} ipython3
:tags: ["remove_input"]
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

%matplotlib inline

# figure size in inches optional
rcParams['figure.figsize'] = 16 ,10
rcParams['figure.dpi'] = 450

# read images
img_A = mpimg.imread('figures/real_orig_alcaim.png')
img_B = mpimg.imread('figures/real_synth_alcaim.png')

# display images
fig, ax = plt.subplots(1,2, gridspec_kw = {'wspace':0, 'hspace':0.5,'width_ratios': [1, 1.07]})

ax[0].axis('off')
ax[0].imshow(img_A)
ax[1].axis('off')
ax[1].imshow(img_B)
x = None
x
```

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