# SALMONN MOS vs objective & neural metrics (VoiceBank-DEMAND-16k, noisy)

- pairs with all metrics + parsed SALMONN MOS: 696 of 824

## Correlation of SALMONN MOS with each metric

| metric | Pearson r | Spearman ρ | n |
|---|---|---|---|
| DNSMOS OVRL | +0.375 | +0.397 | 696 |
| DNSMOS P.808 | +0.493 | +0.489 | 696 |
| DNSMOS SIG | +0.260 | +0.261 | 696 |
| DNSMOS BAK | +0.382 | +0.383 | 696 |
| NISQA MOS | +0.450 | +0.444 | 696 |
| NISQA noisiness | +0.381 | +0.408 | 696 |
| NISQA discontinuity | +0.390 | +0.404 | 696 |
| NISQA coloration | +0.415 | +0.417 | 696 |
| PESQ (wb) | +0.446 | +0.475 | 696 |
| STOI | +0.438 | +0.442 | 696 |
| SI-SDR | +0.372 | +0.371 | 696 |
| segmental SNR | +0.406 | +0.414 | 696 |
| global SNR | +0.373 | +0.372 | 696 |

## Cross-agreement among MOS predictors (Spearman ρ)

| | SALMONN | DNSMOS | NISQA | PESQ |
|---|---|---|---|---|
| **SALMONN** | +1.00 | +0.40 | +0.44 | +0.48 |
| **DNSMOS** | +0.40 | +1.00 | +0.78 | +0.72 |
| **NISQA** | +0.44 | +0.78 | +1.00 | +0.82 |
| **PESQ** | +0.48 | +0.72 | +0.82 | +1.00 |

> Reads: how well each pair of *quality predictors* rank the 824 files the same way. SALMONN's row shows whether it agrees with purpose-built MOS predictors as much as they agree with each other.

## Scale usage (mean ± std)

- SALMONN MOS: 3.98 ± 0.73  (range 2.50–5.00)
- DNSMOS OVRL: 2.71 ± 0.51  (range 1.10–3.46)
- NISQA MOS: 3.10 ± 0.88  (range 0.89–4.79)
- PESQ: 1.99 ± 0.76  (range 1.03–4.31)
- SALMONN distinct values: [2.5, 3.5, 4.0, 4.5, 5.0]
