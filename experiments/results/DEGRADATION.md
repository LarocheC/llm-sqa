# Controlled degradation sweep — SALMONN descriptive SQA

- 16 utterances (p232, p257), 288 clips total
- clean baseline: SALMONN MOS 4.77, DNSMOS 3.35, NISQA 4.61, PESQ 4.64

## Monotonicity: Spearman ρ(severity, score) per family

(severity increases with degradation; a strong NEGATIVE ρ = score falls as quality drops = good)

| family | SALMONN | DNSMOS | NISQA | PESQ |
|---|---|---|---|---|
| noise | -0.78 | -0.95 | -0.95 | -0.93 |
| lowpass | -0.39 | -0.16 | -0.79 | -0.85 |
| clip | -0.33 | -0.66 | -0.86 | -0.92 |
| reverb | -0.11 | -0.72 | -0.70 | -0.75 |

## noise

| level | SALMONN MOS | DNSMOS | NISQA | PESQ | names it? |
|---|---|---|---|---|---|
| clean | 4.77 | 3.35 | 4.61 | 4.64 | 69% |
| snr20 | 4.07 | 2.58 | 2.08 | 1.42 | 88% |
| snr15 | 3.80 | 2.33 | 1.54 | 1.21 | 94% |
| snr10 | 3.19 | 2.14 | 1.14 | 1.10 | 94% |
| snr5 | 2.90 | 1.90 | 0.94 | 1.06 | 100% |
| snr0 | 2.50 | 1.62 | 0.88 | 1.04 | 100% |

*example @ snr0 (MOS 2.5):* The audio contains very intrusive background noise from 0 to 1.5 seconds, making it very difficult to hear the speech clearly. The voice sounds distorted, with a buzzing texture from 0.5 to 1.5 seconds, which further detracts from clarity and naturalness. The speech lacks coherence, with breaks and 

## lowpass

| level | SALMONN MOS | DNSMOS | NISQA | PESQ | names it? |
|---|---|---|---|---|---|
| clean | 4.77 | 3.35 | 4.61 | 4.64 | 6% |
| lp6000 | 4.37 | 3.35 | 4.40 | 4.40 | 25% |
| lp4000 | 4.28 | 3.34 | 3.92 | 4.10 | 25% |
| lp3000 | 4.09 | 3.32 | 3.81 | 3.74 | 38% |
| lp2000 | 3.73 | 3.30 | 3.18 | 3.40 | 50% |

*example @ lp2000 (MOS 5.0):* The audio is in good quality, with no noticeable background noise, distortion, or artifacts, allowing for a clear listening experience. The speaker's voice, a young man's deep tone, is mostly natural, contributing to a pleasant auditory experience. There are no breaks or stutters in the speech, maki

## clip

| level | SALMONN MOS | DNSMOS | NISQA | PESQ | names it? |
|---|---|---|---|---|---|
| clean | 4.77 | 3.35 | 4.61 | 4.64 | 69% |
| clip0.5 | 4.12 | 3.28 | 4.22 | 3.46 | 69% |
| clip0.3 | 4.09 | 3.16 | 3.36 | 2.43 | 94% |
| clip0.2 | 3.67 | 3.04 | 2.70 | 1.87 | 100% |
| clip0.1 | 3.50 | 2.72 | 1.87 | 1.39 | 100% |

*example @ clip0.1 (MOS nan):* {
"sentence\_type": "statement",
"sentence\_content": "Please call Stealer.",
"tone": "serious",
"emotion": "calm",
"clarity": 4.5,
"naturalness": 4.0,
"intelligibility": 4.0,
"listening\_effort": 3.5,
"background noise": 0,
"distortion": 0,
"naturalness of distortion": 4.0,
"overall quality": 4.0
}

## reverb

| level | SALMONN MOS | DNSMOS | NISQA | PESQ | names it? |
|---|---|---|---|---|---|
| clean | 4.77 | 3.35 | 4.61 | 4.64 | 6% |
| rt0.3 | 4.08 | 2.63 | 2.78 | 1.33 | 31% |
| rt0.6 | 4.19 | 1.84 | 2.12 | 1.18 | 12% |
| rt0.9 | 4.20 | 1.71 | 1.85 | 1.15 | 0% |
| rt1.2 | 3.85 | 1.50 | 1.63 | 1.13 | 31% |

*example @ rt1.2 (MOS nan):* {
"sentence\_type": "statement",
"sentence\_content": "Please don't touch.",
"sentence\_confidence": 0.9,
"sentence\_clarity": 0.9,
"sentence\_naturalness": 0.9,
"sentence\_intelligibility": 0.9,
"listening\_effort": 0.5,
"background noise": false,
"distortion": false,
"naturalness": 0.9,
"intelligi
