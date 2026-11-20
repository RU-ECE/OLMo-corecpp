> **Superseded:** the first table below used DeepGaze with its natural-photo (MIT1003) prior; the follow-up
> at the end (UEyes train prior, pipeline self-check, paired tests) replaces its reading.

# Where students look, step 1: does an off-the-shelf saliency model predict gaze on UIs? (2026-11-20)

`scripts/colab_gaze_eval.py` on a Colab A100 (after the SL-MTP decode finished). Data: UEyes (Jiang et al.,
CHI 2023; CC-BY-4.0; Zenodo 8010312), its 108-image test split (27 each: desktop, mobile, poster, web),
read by HTTP range requests from the 12.9 GB archive (CRC-checked). Ground truth: 3 s heatmaps (CC, KLD)
and 3 s fixation maps (NSS). Model: DeepGaze IIE (natural images, MIT1003 centre bias), images at max side
1024. Baselines without a model: a centred Gaussian, and a top-left-weighted Gaussian (UIs are read from
the top left).

| UI type | model | n | CC (higher better) | KLD (lower better) | NSS (higher better) |
|---|---|---|---|---|---|
| desktop | centre_gaussian | 27 | 0.111 | 1.929 | 0.330 |
| desktop | deepgaze_iie | 27 | **0.340** | 1.517 | **0.921** |
| desktop | top_left_prior | 27 | 0.330 | **1.368** | 0.851 |
| mobile | centre_gaussian | 27 | 0.088 | 1.588 | 0.156 |
| mobile | deepgaze_iie | 27 | 0.317 | 1.346 | 0.741 |
| mobile | top_left_prior | 27 | **0.488** | **0.897** | **0.992** |
| poster | centre_gaussian | 27 | 0.177 | 1.401 | 0.368 |
| poster | deepgaze_iie | 27 | 0.345 | 1.443 | 0.751 |
| poster | top_left_prior | 27 | **0.384** | **1.071** | **0.792** |
| web | centre_gaussian | 27 | 0.091 | 1.741 | 0.254 |
| web | deepgaze_iie | 27 | 0.291 | 1.565 | 0.761 |
| web | top_left_prior | 27 | **0.408** | **1.209** | **1.001** |

Gate (beats both priors on CC and NSS): desktop yes (narrowly); mobile, poster, web NO.

Reading: a natural-image saliency model is not good enough to gate "will students look at this" on
interfaces; a one-line reading-order prior beats it on 3 of 4 UI types. The next step is a model trained on
UI gaze (UEyes train split, 1,872 images; UMSI++ from the UEyes authors is the published reference) and then
on our own widgets with real student eye tracking (blocked: needs a study). n = 27 per type is small: these
are first evidence, not a benchmark result.

## Follow-up the same evening (scripts/colab_gaze_check.py): the first reading was mostly a wrong prior

Pipeline self-check over the 108 test images: CC(gt, gt) = 1.0000, KLD(gt, gt) ~ 0, CC(another image's
gt, gt) = 0.201: the metric code tells maps apart. Fixations now rescaled by coordinate (no dropped points).
New prior: the mean 3 s heatmap of the UEyes train split (1,872 images) per UI type, used as a baseline and
as DeepGaze IIE's centre bias.

| UI type | DeepGaze (MIT prior) CC / NSS | DeepGaze (UEyes train prior) CC / NSS | train-mean prior alone | top-left prior |
|---|---|---|---|---|
| desktop | 0.340 / 0.913 | **0.481 / 1.272** | 0.371 / 0.969 | 0.330 / 0.852 |
| mobile | 0.317 / 0.740 | 0.488 / **1.182** | **0.519** / 1.093 | 0.488 / 0.992 |
| poster | 0.345 / 0.756 | **0.526 / 1.128** | 0.465 / 0.960 | 0.384 / 0.795 |
| web | 0.291 / 0.756 | **0.478 / 1.211** | 0.450 / 1.117 | 0.408 / 1.002 |

Paired Wilcoxon per image against the train-mean prior (n = 27 per type): DeepGaze with the UEyes prior is
better on desktop (CC +0.102, p = 0.0004; NSS +0.237, p = 0.0007) and not significantly different on mobile,
poster, web (p 0.12-0.82). The top-left prior is significantly worse than the train-mean prior on desktop,
poster and web (p <= 0.003).

Reading (replaces the first one): image features add measurable value over a good UI prior on desktop UIs
only (CC and NSS; KLD on mobile is worse than the top-left prior, 1.023 vs 0.897, and no paired test was
run against the top-left prior); on the other types the learned layout prior does most of the work. For a "looked at" gate on Knowsy
widgets: use DeepGaze IIE with a UI prior as the first model, and the evidence says a UI-trained model is the
next step (the prior alone already carries most of the signal). Still owed: validation on students.
