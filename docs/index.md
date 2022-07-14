# Audio Samples for Multitrack Music Transformer

## Models

- __MTMT-APE__ (ours): Multitrack Music Transformer with absolute positional embedding
- __MTMT-RPE__ (ours): Multitrack Music Transformer with relative positional encoding
- __MTMT-NPE__ (ours): Multitrack Music Transformer without positional encoding
- __MMM__: Decoder-only transformer using the MultiTrack representation proposed by Ens and Pasquier (2020)
- __REMI+__: Decoder-only transformer using the REMI+ representation proposed by von Rütte (2022)

---

## Best samples

Here are some of the best samples we found.

### Best unconditioned generation samples

|-|-|
| {% include audio_player.html filename="audio/sod/best/3_unconditioned.mp3" %} | {% include audio_player.html filename="audio/sod/best/12_unconditioned.mp3" %} |
| {% include audio_player.html filename="audio/sod/best/16_unconditioned.mp3" %} | {% include audio_player.html filename="audio/sod/best/23_unconditioned.mp3" %} |
| {% include audio_player.html filename="audio/sod/best/31_unconditioned.mp3" %} | {% include audio_player.html filename="audio/sod/best/39_unconditioned.mp3" %} |
| {% include audio_player.html filename="audio/sod/best/43_unconditioned.mp3" %} | {% include audio_player.html filename="audio/sod/best/45_unconditioned.mp3" %} |

### Best instrument-informed generation samples

|-|-|
| piano, church-organ, voices | {% include audio_player.html filename="audio/sod/best/7_instrument-informed.mp3" %} |
| trumpet, trombone | {% include audio_player.html filename="audio/sod/best/33_instrument-informed.mp3" %} |
| contrabass, harp, english-horn, flute | {% include audio_player.html filename="audio/sod/best/40_instrument-informed.mp3" %} |
| church-organ, viola, contrabass, strings, voices, horn, oboe | {% include audio_player.html filename="audio/sod/best/10_instrument-informed.mp3" %} |

### Best 4-beat continuation samples

|-|-|
| {% include audio_player.html filename="audio/sod/best/9_4-beat-continuation.mp3" %} | {% include audio_player.html filename="audio/sod/best/19_4-beat-continuation.mp3" %} |
| {% include audio_player.html filename="audio/sod/best/23_4-beat-continuation.mp3" %} | {% include audio_player.html filename="audio/sod/best/26_4-beat-continuation.mp3" %} |
| {% include audio_player.html filename="audio/sod/best/34_4-beat-continuation.mp3" %} | {% include audio_player.html filename="audio/sod/best/35_4-beat-continuation.mp3" %} |

---

## Orchestral generation (unselected samples)

### Unconditioned generation (SOD)

|                 | Sample result 1 | Sample result 2 |
|:---------------:|:---------------:|:---------------:|
| MTMT-APE (ours) | {% include audio_player.html filename="audio/sod/ape/0_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/ape/1_unconditioned.mp3" style="width:250px;" %} |
| MTMT-RPE (ours) | {% include audio_player.html filename="audio/sod/rpe/0_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/rpe/1_unconditioned.mp3" style="width:250px;" %} |
| MTMT-NPE (ours) | {% include audio_player.html filename="audio/sod/npe/0_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/npe/1_unconditioned.mp3" style="width:250px;" %} |
| MMM             | {% include audio_player.html filename="audio/sod/mmm/0_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/mmm/1_unconditioned.mp3" style="width:250px;" %} |
| REMI+           | {% include audio_player.html filename="audio/sod/remi/0_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/remi/1_unconditioned.mp3" style="width:250px;" %} |

|                 | Sample result 3 | Sample result 4 |
|:---------------:|:---------------:|:---------------:|
| MTMT-APE (ours) | {% include audio_player.html filename="audio/sod/ape/2_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/ape/3_unconditioned.mp3" style="width:250px;" %} |
| MTMT-RPE (ours) | {% include audio_player.html filename="audio/sod/rpe/2_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/rpe/3_unconditioned.mp3" style="width:250px;" %} |
| MTMT-NPE (ours) | {% include audio_player.html filename="audio/sod/npe/2_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/npe/3_unconditioned.mp3" style="width:250px;" %} |
| MMM             | {% include audio_player.html filename="audio/sod/mmm/2_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/mmm/3_unconditioned.mp3" style="width:250px;" %} |
| REMI+           | {% include audio_player.html filename="audio/sod/remi/2_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/remi/3_unconditioned.mp3" style="width:250px;" %} |

### Instrument-informed generation (SOD)

|          | Sample result 1 | Sample result 2 |
|:--------:|:---------------:|:---------------:|
| MTMT-APE | {% include audio_player.html filename="audio/sod/ape/0_instrument-informed.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/ape/1_instrument-informed.mp3" style="width:250px;" %} |
| MTMT-RPE | {% include audio_player.html filename="audio/sod/rpe/0_instrument-informed.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/rpe/1_instrument-informed.mp3" style="width:250px;" %} |
| MTMT-NPE | {% include audio_player.html filename="audio/sod/npe/0_instrument-informed.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/npe/1_instrument-informed.mp3" style="width:250px;" %} |

|          | Sample result 3 | Sample result 4 |
|:--------:|:---------------:|:---------------:|
| MTMT-APE | {% include audio_player.html filename="audio/sod/ape/2_instrument-informed.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/ape/3_instrument-informed.mp3" style="width:250px;" %} |
| MTMT-RPE | {% include audio_player.html filename="audio/sod/rpe/2_instrument-informed.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/rpe/3_instrument-informed.mp3" style="width:250px;" %} |
| MTMT-NPE | {% include audio_player.html filename="audio/sod/npe/2_instrument-informed.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/npe/3_instrument-informed.mp3" style="width:250px;" %} |

### 4-beat continuation (SOD)

|              | Sample result 1 | Sample result 2 |
|:------------:|:---------------:|:---------------:|
| MTMT-APE     | {% include audio_player.html filename="audio/sod/ape/0_4-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/ape/1_4-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-RPE     | {% include audio_player.html filename="audio/sod/rpe/0_4-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/rpe/1_4-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-NPE     | {% include audio_player.html filename="audio/sod/npe/0_4-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/npe/1_4-beat-continuation.mp3" style="width:250px;" %} |
| Ground truth | {% include audio_player.html filename="audio/sod/truth/0_truth.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/truth/1_truth.mp3" style="width:250px;" %} |

|              | Sample result 3 | Sample result 4 |
|:------------:|:---------------:|:---------------:|
| MTMT-APE     | {% include audio_player.html filename="audio/sod/ape/2_4-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/ape/3_4-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-RPE     | {% include audio_player.html filename="audio/sod/rpe/2_4-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/rpe/3_4-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-NPE     | {% include audio_player.html filename="audio/sod/npe/2_4-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/npe/3_4-beat-continuation.mp3" style="width:250px;" %} |
| Ground truth | {% include audio_player.html filename="audio/sod/truth/2_truth.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/truth/3_truth.mp3" style="width:250px;" %} |

### 16-beat continuation (SOD)

|              | Sample result 1 | Sample result 2 |
|:------------:|:---------------:|:---------------:|
| MTMT-APE     | {% include audio_player.html filename="audio/sod/ape/0_16-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/ape/1_16-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-RPE     | {% include audio_player.html filename="audio/sod/rpe/0_16-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/rpe/1_16-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-NPE     | {% include audio_player.html filename="audio/sod/npe/0_16-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/npe/1_16-beat-continuation.mp3" style="width:250px;" %} |
| Ground truth | {% include audio_player.html filename="audio/sod/truth/0_truth.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/truth/1_truth.mp3" style="width:250px;" %} |

|              | Sample result 3 | Sample result 4 |
|:------------:|:---------------:|:---------------:|
| MTMT-APE     | {% include audio_player.html filename="audio/sod/ape/2_16-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/ape/3_16-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-RPE     | {% include audio_player.html filename="audio/sod/rpe/2_16-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/rpe/3_16-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-NPE     | {% include audio_player.html filename="audio/sod/npe/2_16-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/npe/3_16-beat-continuation.mp3" style="width:250px;" %} |
| Ground truth | {% include audio_player.html filename="audio/sod/truth/2_truth.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/sod/truth/3_truth.mp3" style="width:250px;" %} |

---

## Pop music generation (unselected samples)

### Unconditioned generation (LMD)

|                 | Sample result 1 | Sample result 2 |
|:---------------:|:---------------:|:---------------:|
| MTMT-APE (ours) | {% include audio_player.html filename="audio/lmd/ape/0_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/ape/1_unconditioned.mp3" style="width:250px;" %} |
| MTMT-RPE (ours) | {% include audio_player.html filename="audio/lmd/rpe/0_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/rpe/1_unconditioned.mp3" style="width:250px;" %} |
| MTMT-NPE (ours) | {% include audio_player.html filename="audio/lmd/npe/0_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/npe/1_unconditioned.mp3" style="width:250px;" %} |
| MMM             | {% include audio_player.html filename="audio/lmd/mmm/0_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/mmm/1_unconditioned.mp3" style="width:250px;" %} |
| REMI+           | {% include audio_player.html filename="audio/lmd/remi/0_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/remi/1_unconditioned.mp3" style="width:250px;" %} |

|                 | Sample result 3 | Sample result 4 |
|:---------------:|:---------------:|:---------------:|
| MTMT-APE (ours) | {% include audio_player.html filename="audio/lmd/ape/2_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/ape/3_unconditioned.mp3" style="width:250px;" %} |
| MTMT-RPE (ours) | {% include audio_player.html filename="audio/lmd/rpe/2_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/rpe/3_unconditioned.mp3" style="width:250px;" %} |
| MTMT-NPE (ours) | {% include audio_player.html filename="audio/lmd/npe/2_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/npe/3_unconditioned.mp3" style="width:250px;" %} |
| MMM             | {% include audio_player.html filename="audio/lmd/mmm/2_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/mmm/3_unconditioned.mp3" style="width:250px;" %} |
| REMI+           | {% include audio_player.html filename="audio/lmd/remi/2_unconditioned.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/remi/3_unconditioned.mp3" style="width:250px;" %} |

### Instrument-informed generation (LMD)

|          | Sample result 1 | Sample result 2 |
|:--------:|:---------------:|:---------------:|
| MTMT-APE | {% include audio_player.html filename="audio/lmd/ape/0_instrument-informed.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/ape/1_instrument-informed.mp3" style="width:250px;" %} |
| MTMT-RPE | {% include audio_player.html filename="audio/lmd/rpe/0_instrument-informed.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/rpe/1_instrument-informed.mp3" style="width:250px;" %} |
| MTMT-NPE | {% include audio_player.html filename="audio/lmd/npe/0_instrument-informed.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/npe/1_instrument-informed.mp3" style="width:250px;" %} |

|          | Sample result 3 | Sample result 4 |
|:--------:|:---------------:|:---------------:|
| MTMT-APE | {% include audio_player.html filename="audio/lmd/ape/2_instrument-informed.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/ape/3_instrument-informed.mp3" style="width:250px;" %} |
| MTMT-RPE | {% include audio_player.html filename="audio/lmd/rpe/2_instrument-informed.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/rpe/3_instrument-informed.mp3" style="width:250px;" %} |
| MTMT-NPE | {% include audio_player.html filename="audio/lmd/npe/2_instrument-informed.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/npe/3_instrument-informed.mp3" style="width:250px;" %} |

### 4-beat continuation (LMD)

|              | Sample result 1 | Sample result 2 |
|:------------:|:---------------:|:---------------:|
| MTMT-APE     | {% include audio_player.html filename="audio/lmd/ape/0_4-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/ape/1_4-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-RPE     | {% include audio_player.html filename="audio/lmd/rpe/0_4-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/rpe/1_4-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-NPE     | {% include audio_player.html filename="audio/lmd/npe/0_4-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/npe/1_4-beat-continuation.mp3" style="width:250px;" %} |
| Ground truth | {% include audio_player.html filename="audio/lmd/truth/0_truth.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/truth/1_truth.mp3" style="width:250px;" %} |

|              | Sample result 3 | Sample result 4 |
|:------------:|:---------------:|:---------------:|
| MTMT-APE     | {% include audio_player.html filename="audio/lmd/ape/3_4-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/ape/4_4-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-RPE     | {% include audio_player.html filename="audio/lmd/rpe/3_4-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/rpe/4_4-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-NPE     | {% include audio_player.html filename="audio/lmd/npe/3_4-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/npe/4_4-beat-continuation.mp3" style="width:250px;" %} |
| Ground truth | {% include audio_player.html filename="audio/lmd/truth/3_truth.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/truth/4_truth.mp3" style="width:250px;" %} |

### 16-beat continuation (LMD)

|              | Sample result 1 | Sample result 2 |
|:------------:|:---------------:|:---------------:|
| MTMT-APE     | {% include audio_player.html filename="audio/lmd/ape/0_16-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/ape/1_16-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-RPE     | {% include audio_player.html filename="audio/lmd/rpe/0_16-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/rpe/1_16-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-NPE     | {% include audio_player.html filename="audio/lmd/npe/0_16-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/npe/1_16-beat-continuation.mp3" style="width:250px;" %} |
| Ground truth | {% include audio_player.html filename="audio/lmd/truth/0_truth.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/truth/1_truth.mp3" style="width:250px;" %} |

|              | Sample result 3 | Sample result 4 |
|:------------:|:---------------:|:---------------:|
| MTMT-APE     | {% include audio_player.html filename="audio/lmd/ape/3_16-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/ape/4_16-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-RPE     | {% include audio_player.html filename="audio/lmd/rpe/3_16-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/rpe/4_16-beat-continuation.mp3" style="width:250px;" %} |
| MTMT-NPE     | {% include audio_player.html filename="audio/lmd/npe/3_16-beat-continuation.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/npe/4_16-beat-continuation.mp3" style="width:250px;" %} |
| Ground truth | {% include audio_player.html filename="audio/lmd/truth/3_truth.mp3" style="width:250px;" %} | {% include audio_player.html filename="audio/lmd/truth/4_truth.mp3" style="width:250px;" %} |
