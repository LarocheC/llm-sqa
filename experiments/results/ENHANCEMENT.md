# Enhancement experiment — does SALMONN track ConvFSENet enhancement?

- 48 utterances x (clean, noisy, enhanced)

## Mean score by condition

| condition | SALMONN MOS | PESQ | DNSMOS | NISQA |
|---|---|---|---|---|
| noisy | 4.20 | 2.23 | 2.86 | 3.24 |
| enhanced | 4.23 | 3.09 | 3.23 | 4.47 |
| clean | 4.61 | 4.64 | 3.37 | 4.57 |

## Did SALMONN notice the enhancement? (per-file noisy -> enhanced)

- SALMONN MOS Δ: mean +0.03; improved 27%, unchanged 52%, worsened 21%
- PESQ Δ: mean +0.86 (improved 100%)
- NISQA Δ: mean +1.23 (improved 100%)
- Spearman ρ(SALMONN MOS Δ, PESQ Δ) = +0.03

## Does SALMONN flag enhancement artifacts? (enhanced vs clean descriptions)

| vocabulary | enhanced | clean | noisy |
|---|---|---|---|
| any artifact term | 67% | 75% | 81% |
| muffl | 0% | 0% | 0% |
| distort | 62% | 73% | 77% |
| unnatural | 21% | 27% | 15% |
| robotic | 2% | 2% | 2% |
| artifact | 44% | 42% | 33% |

## Example enhanced descriptions

- **p232_001** (MOS 4.5, PESQ 3.35): The audio begins with a man's voice speaking at a moderate pace, accompanied by background noise resembling music from 0 to 1.8 seconds. The speech is mostly clear and natural, with no significant distortion, and requires only a moderate amount of listening ef
- **p232_002** (MOS 5.0, PESQ 3.8): The audio quality is excellent, with no noticeable background noise, distortion, or artifacts, allowing for a clear and uninterrupted listening experience. The voice, that of a young man, is very natural, making it easy for listeners to engage with the content
- **p232_003** (MOS 4.5, PESQ 3.54): 4.5
