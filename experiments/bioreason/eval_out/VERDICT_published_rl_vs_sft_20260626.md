# Published RL vs SFT — decisive axis test (job 8564498, 2026-06-26)

go_pred-fixed prompt, held-out test set (bioreason_pro_test), N=330, unweighted F_max
(IA.txt loaded; same metric both legs = valid A/B).

| model          | Overall F_max | BP     | CC     | MF     |
|----------------|---------------|--------|--------|--------|
| published SFT  | 0.6686        | 0.5395 | 0.7371 | 0.7291 |
| published RL   | 0.6866        | 0.5538 | 0.7577 | 0.7484 |
| **uplift**     | **+0.0180**   | +0.014 | +0.021 | +0.019 |

VERDICT: F_max IS the right axis. Published RL beats SFT by +0.018 (+2.7%), uniform across
all 3 aspects. This OVERTURNS the earlier broken-prompt result (RL 0.39 < SFT 0.41).

IMPLICATIONS:
- RL on the CORRECT (go_pred) prompt is supposed to lift F_max ~+0.018 — concrete target.
- Our prod RL run was FLAT (0.656 vs 0.657) because it trained on the COLD prompt (no go_pred),
  off-distribution from both SFT and eval. The go_pred fix (branch
  bioreason/go-pred-prompt-injection-20260626) trains on the SAME prompt where RL is known to help.
- Next: go_pred smoke (8564679) -> 4N prod go_pred run -> eval checkpoints vs SFT 0.669.
