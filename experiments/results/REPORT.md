# VoiceBank-DEMAND-16k — SALMONN descriptive SQA analysis

- rows: 1648 (824 noisy, 824 clean)
- genuine (non-degenerate) descriptions: 797 noisy, 756 clean
- terse/score-only replies: noisy 27, clean 68

## SNR distribution (computed per pair: 10·log10(Σclean²/Σ(noisy−clean)²))

- min/median/mean/max: -3.1 / 7.3 / 8.4 / 17.1 dB
- quartiles: [2.4, 7.3, 12.0]

## MOS and description length vs SNR (noisy files)

- Spearman corr(SNR, MOS) = 0.372
- Spearman corr(SNR, n_words) = -0.046

| SNR bucket (dB) | n | mean MOS | mean #words | terse-reply rate |
|---|---|---|---|---|
| <5 | 218 | 3.60 | 98 | 2% |
| 5-10 | 209 | 3.91 | 100 | 4% |
| 10-15 | 206 | 4.08 | 98 | 5% |
| >=15 | 191 | 4.36 | 99 | 3% |

- clean files: mean MOS 4.43 (n=741), mean #words 92, terse-reply rate 8%

## Noise/quality keyword prevalence by SNR bucket (% of genuine noisy descriptions)

| keyword | <5 | 5-10 | 10-15 | >=15 | clean |
|---|---|---|---|---|---|
| noise | 89% | 90% | 89% | 82% | 84% |
| background | 89% | 91% | 91% | 88% | 85% |
| distortion | 78% | 85% | 84% | 87% | 84% |
| muffled | 0% | 0% | 0% | 0% | 0% |
| clear | 45% | 50% | 45% | 58% | 66% |
| clean | 0% | 0% | 0% | 0% | 0% |
| smooth | 52% | 68% | 67% | 78% | 78% |
| natural | 93% | 96% | 93% | 96% | 96% |
| effort | 91% | 93% | 92% | 90% | 92% |

## Quality verdict words (% of genuine descriptions containing)

| verdict | noisy | clean |
|---|---|---|
| excellent | 33% | 65% |
| good | 31% | 29% |
| fair | 34% | 13% |
| poor | 9% | 6% |
| bad | 0% | 0% |
| moderate | 85% | 82% |

## Words most over-represented in NOISY vs clean descriptions

| word | noisy % | clean % | log-odds |
|---|---|---|---|
| outdoor | 11% | 0% | +4.47 |
| chatter | 11% | 0% | +4.45 |
| music | 8% | 0% | +4.15 |
| specifically | 8% | 0% | +4.07 |
| people | 10% | 0% | +3.28 |
| contains | 26% | 1% | +3.07 |
| talking | 5% | 0% | +2.27 |
| sentences | 5% | 0% | +2.25 |
| overly | 33% | 4% | +2.14 |
| intrusive | 36% | 6% | +1.75 |
| amount | 16% | 3% | +1.70 |
| itself | 6% | 1% | +1.64 |
| impacting | 4% | 1% | +1.45 |
| issues | 16% | 4% | +1.41 |
| positively | 8% | 2% | +1.26 |

## Words most over-represented in CLEAN vs noisy descriptions

| word | clean % | noisy % | log-odds |
|---|---|---|---|
| uninterrupted | 4% | 0% | +2.22 |
| delivered | 4% | 1% | +1.81 |
| effortlessly | 33% | 8% | +1.35 |
| engagement | 4% | 1% | +1.34 |
| mental | 31% | 8% | +1.34 |
| casualness | 5% | 1% | +1.33 |
| monotonous | 4% | 1% | +1.32 |
| disrupt | 5% | 1% | +1.31 |
| strain | 31% | 8% | +1.30 |
| pace | 13% | 4% | +1.29 |
| exceptional | 13% | 4% | +1.28 |
| outstanding | 8% | 2% | +1.22 |
| maintaining | 4% | 1% | +1.21 |
| conveys | 15% | 5% | +1.13 |
| ease | 12% | 4% | +1.12 |
