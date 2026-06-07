# DPO / GRPO / RLVR notes

## DPO

DPO uses preference pairs, typically `(prompt, chosen, rejected)`, and directly optimizes the policy against a frozen reference model. Engineering points to watch:

- chosen/rejected formatting and chat template consistency;
- reference model memory footprint;
- beta / KL strength;
- max length and truncation strategy;
- whether loss decreases while response quality actually improves.

## PPO / RLHF

PPO-style RLHF usually needs rollout generation, reward scoring, advantage estimation, and policy update. Engineering bottlenecks:

- multiple models loaded at once;
- slow rollout generation;
- reward model inference cost;
- long sequence activation memory;
- KL too high or too low causing training instability.

## GRPO / RLVR

GRPO/RLVR is attractive for verifiable tasks such as math and code because a verifier can provide rule-based or execution-based reward.

Key implementation points:

- group sampling: several completions per prompt;
- relative advantage within a group;
- answer extraction must be reliable;
- verifier false positives/negatives can corrupt reward;
- rollout throughput often dominates training time.
