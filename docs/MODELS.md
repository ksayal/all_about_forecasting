# Model sources and licenses

Checked on 2026-11-12 against PyPI and Hugging Face model cards. Licenses
change, so re-check before commercial use.

| Model | Authors | Official code | PyPI package (pinned here) | Weights (Hugging Face) | Code license | Weights license |
|---|---|---|---|---|---|---|
| Chronos-2 | Amazon | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) | `chronos-forecasting==2.3.2` | `amazon/chronos-2` | Apache-2.0 | Apache-2.0 |
| Chronos-Bolt | Amazon | same as above | `chronos-forecasting==2.3.2` | `amazon/chronos-bolt-{tiny,mini,small,base}` | Apache-2.0 | Apache-2.0 |
| TimesFM-2.5 | Google Research | [google-research/timesfm](https://github.com/google-research/timesfm) | `timesfm[torch]==3.0.2` | `google/timesfm-2.5-200m-pytorch` | Apache-2.0 | Apache-2.0 |
| TiRex | NX-AI | [NX-AI/tirex](https://github.com/NX-AI/tirex) | `tirex-ts==1.4.2` | `NX-AI/TiRex` | NXAI Community License | NXAI Community License |
| Toto-1.0 | Datadog | [DataDog/toto](https://github.com/DataDog/toto) | `toto-ts==0.2.0` | `Datadog/Toto-Open-Base-1.0` | Apache-2.0 | Apache-2.0 |
| TabPFN-TS | Prior Labs | [PriorLabs/tabpfn-time-series](https://github.com/PriorLabs/tabpfn-time-series) | `tabpfn-time-series==1.4.0` (pulls `tabpfn`) | TabPFN regressor, fetched by `tabpfn` | Apache-2.0 | Prior Labs license (check current version) |
| Moirai-2.0 | Salesforce AI Research | [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | `uni2ts==2.0.0` | `Salesforce/moirai-2.0-R-small` | Apache-2.0 | **CC-BY-NC-4.0** |
| Sundial | Tsinghua THUML | [thuml/Sundial](https://github.com/thuml/Sundial) | none; uses `transformers==4.40.1` | `thuml/sundial-base-128m` | Apache-2.0 (verify) | Apache-2.0 (verify) |
| COSMIC | AWS AI Labs | none public; paper [arXiv:2506.03128](https://arxiv.org/abs/2506.03128) | none | none | n/a | n/a |
| Stat baselines | Nixtla | [Nixtla/statsforecast](https://github.com/Nixtla/statsforecast) | `statsforecast==2.1.1` | none | Apache-2.0 | n/a |

## License notes in plain words

- **Apache-2.0**: use commercially, modify, redistribute; keep the notice.
- **NXAI Community License (TiRex)**: free for research and smaller commercial
  users; larger commercial use needs a separate agreement. Read it before production.
- **CC-BY-NC-4.0 (Moirai-2.0 weights)**: non-commercial only. Fine for this
  experimentation repo, not for a product.
- **TabPFN weights**: Prior Labs' own terms. Newer TabPFN versions are
  non-commercial; the hosted API has its own terms of service.

## About the leaderboard

The table this repo started from (Avg. Win Rate, Skill Score, Median runtime,
Leakage, #Failures) matches the format of the **fev-bench** leaderboard
(autogluon/fev). Its "Leakage %" is the share of benchmark tasks that overlap
a model's training data. Moirai-2.0 (28%) and TimesFM-2.5 / Toto (8%) had
the most, which is worth remembering when comparing scores.
