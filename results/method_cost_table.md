| method | models loaded | relative memory | rollout | engineering note |
|---|---|---|---|---|
| SFT | policy | low | no | stable supervised baseline |
| DPO | policy + frozen reference | medium | no online rollout | pairwise preference optimization |
| PPO/RLHF | policy + reference + reward + critic/value | high | yes | rollout and reward scoring dominate cost |
| GRPO/RLVR | policy + reference + verifier/reward | high | yes | group sampling and verifier reward for reasoning tasks |
