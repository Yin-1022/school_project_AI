# 16-4 Pre-Attack Baseline Summary

**日期：2026-11-22**  
**階段：Step 16-4**  
**用途：記錄加入 Persistent Action Segment、Attack1 / Attack2 前的 IMPALA baseline。**

---

## 1. Baseline 目的

本輪測試主要用於確認目前 IMPALA training pipeline 在既有 10-action action space 下能穩定運作，並建立後續 Step 17 修改 movement semantics 與加入攻擊動作前的比較基準。

本輪重點不是追求最終戰鬥勝率，而是確認：

- Actor / Learner pipeline 是否穩定
- V-trace / importance ratio 是否正常
- batch stabilization 是否有效
- terminal partial batch 是否正常處理
- rejected policy transition 是否確實參與訓練
- Low reward penalty 是否正常生效
- reward / damage attribution 是否正確
- Ground Truth diagnostics 是否能正常追蹤 runtime 行為

---

## 2. Baseline 設定

### Environment

- Map：`castle`
- 僅測試第一關
- `cave` 已完成 map signal wiring，但本輪不納入 baseline
- Map 目前只作為 metadata，不參與 observation、reward 或 action masking

### Action Space

目前維持 10 個 action slots：

```text
0 Hold
1 Advance
2 StrafeLeft
3 StrafeRight
4 EvadeBack
5 Retreat
6 SearchTurnLeft
7 SearchTurnRight
8 PatrolStepLeft
9 PatrolStepRight
```

其中 `Retreat` 保留 action slot，但目前永久 masked。

Attack1 / Attack2 尚未加入 policy action space。

### Observation

- Frames：`(3, 8, 192, 192)`
- Extra：`(24,)`
- Previous-action one-hot：10 維

### IMPALA Batch

```text
UNROLL_LENGTH = 20
MIN_BATCH_UNROLLS = 2
MIN_BATCH_VALID_STEPS = 32
max_grad_norm = 40
```

模型由既有 BC checkpoint warmstart，再進入 IMPALA training。

---

## 3. Reward 設定

### High Reward

```text
Boss Win  = +5
Boss Loss = -5
```

### Medium Reward

以 Boss 與 Player 的 damage exchange 計算：

```text
Boss → Player damage / PLAYER_MAX_HP
-
Player → Boss damage / BOSS_MAX_HP
```

### Low Reward

目前只有 rejected Strafe：

```text
/notseeplayer
+ policy action = StrafeLeft / StrafeRight
→ -0.03
```

Reward priority：

```text
High > Medium > Low > None
```

同一 transition 若同時存在多個 channel，只選最高 priority reward 作為 learner 的 `selected_reward`。

---

## 4. Training Baseline

本次 clean baseline 共觀察：

```text
Learner updates: 12
Episodes: 16
Actor: actor_0
Map: castle
```

16 場結果：

```text
Boss Win:  0
Boss Loss: 16
```

因此本輪不能視為 policy performance 已經收斂或戰鬥能力已提升。

本輪主要結論為：

> IMPALA training pipeline 可以穩定持續更新，未觀察到明顯 learner divergence。

---

## 5. Learner Stability

### Importance Ratio

整體 `mean_rho` 長期維持在約 1 附近。

後期例如：

```text
Step 7   mean_rho = 0.9939
Step 8   mean_rho = 1.0019
Step 9   mean_rho = 1.0034
Step 10  mean_rho = 1.0035
Step 11  mean_rho = 1.0017
Step 12  mean_rho = 0.9978
```

沒有觀察到明顯 Actor / Learner policy drift。

### Entropy

本輪大致維持：

```text
0.66 ~ 0.82
```

沒有持續快速下降，因此目前沒有明顯 policy collapse。

### Value Estimate

Value mean 由初期正值逐漸向負值移動：

```text
Step 1   +0.305
Step 7   -0.197
Step 9   -0.275
Step 10  -0.394
Step 11  -0.515
Step 12  -0.493
```

由於目前所有 baseline episode 均為 Boss Loss，此趨勢與實際 outcome 方向一致。

---

## 6. Terminal-heavy Batch 現象

測試中發現 Value loss 與 pre-clip gradient norm 高度受到 batch 內 terminal transition 數量影響。

大致觀察：

```text
0 terminal:
Value loss ≈ 0.06
grad_norm ≈ 0.8 ~ 3.5

1 terminal:
Value loss ≈ 2 ~ 6
grad_norm ≈ 13 ~ 37

2 terminals:
Value loss ≈ 8 ~ 14
grad_norm ≈ 45 ~ 65
```

因此過去看到：

```text
Value loss ≈ 13
grad_norm ≈ 60
```

不代表 learner 必然發散，而可能只是該 batch 同時包含兩個 `Boss Loss = -5` terminal transition。

`grad_norm` 為 `clip_grad_norm_()` 回傳的 clipping 前 norm；實際 optimizer update 仍受：

```text
max_grad_norm = 40
```

限制。

---

## 7. Terminal-aware Learner Diagnostics

Step 16-4-B 新增 learner-side diagnostics：

```text
Terminal transitions
Terminal wins
Terminal losses

Reward priority counts:
High
Medium
Low
None
```

Commit：

```text
3516ec8
learner update_16-4-B: add terminal-aware batch diagnostics
```

Step 12 驗證結果：

```text
Valid: 32
Terminal transitions: 1
Terminal results: wins=0, losses=1

Reward priority counts:
high=1
medium=8
low=4
none=19
```

Sanity check：

```text
1 + 8 + 4 + 19 = 32 = Valid
```

且：

```text
High count = Terminal transitions = 1
```

diagnostics 正常。

---

## 8. Rejected Action Learning

Rejected Strafe transition 已不再被 drop。

流程：

```text
Policy Strafe
→ UE /notseeplayer
→ transition 保留
→ runtime fallback Advance
→ original policy Strafe 仍進 rollout
→ Low reward -0.03
```

Low reward 在實際訓練中可正確對應 rejected 次數。

例如：

```text
5 rejected → -0.15
6 rejected → -0.18
```

目前沒有觀察到 `-0.03` penalty 導致 policy 完全停止選擇 Strafe。

---

## 9. Action Distribution

本輪 policy 有明顯偏向：

```text
Advance
StrafeRight
```

初期前 5 場中：

```text
Advance + StrafeRight ≈ 85.4% actions
```

Episode 11～15：

```text
Advance + StrafeRight ≈ 79.6%
```

後期 SearchTurnRight 的使用量有增加，但目前樣本仍不足以判定為穩定 learning trend。

目前沒有明顯 entropy collapse，因此較可能是 BC warmstart 本身已有較強 action preference，而不是 IMPALA 訓練後發生 policy collapse。

---

## 10. Gameplay Baseline

目前 16 場皆為 Boss Loss，因此不以勝率判斷此次 pipeline 成功與否。

部分後期 episode 已出現：

```text
Boss → Player normalized damage = 0.8
Player → Boss normalized damage = 1.0
Net damage = -0.2
```

相較部分早期 episode 的：

```text
Net damage = -0.4 ~ -0.8
```

有較接近擊殺玩家的情況。

但由於 episode 數量有限，且 perception / combat state 差異很大，目前不將此視為已證明的 policy improvement。

---

## 11. Ground Truth / Runtime Diagnostics

大多數 episode：

```text
Unmatched Distance: 0
Unmatched Failure Signal: 0
Action Mismatch: 0
```

`Unmatched Status` 偶爾出現，最高曾在 Episode 16 出現：

```text
Unmatched Status: 7
```

同場也有：

```text
actstart_timeout
```

但目前尚無足夠證據確認兩者存在直接因果關係。

此問題目前列為 observation，不阻塞後續開發。

若後續多場持續出現高 `Unmatched Status`，再另外進行 GT timing / event lifecycle investigation。

---

## 12. Perception Observation

Perception accuracy 在不同 episode 間變動較大。

例如部分 episode：

```text
Accuracy = 100%
```

但 Episode 16：

```text
Accuracy = 54.84%
Precision = 50.00%
Recall = 100.00%
FP = 14
```

`notseeplayer` rejection 與 False Positive 有一定關聯，但不是一對一關係，因為是否產生 rejection 還取決於當下 policy 是否選擇 Strafe。

因此目前不直接以 rejection rate 作為 perception accuracy 的替代指標。

---

## 13. Baseline 結論

截至 Step 16-4：

```text
Terminal Partial Batch Handling      ✅
Batch Stabilization                  ✅
Ground Truth Diagnostics             ✅
Healing-aware Damage Tracking        ✅
Movement Failure Diagnostics         ✅
Rejected Policy Transition Learning  ✅
Low Reward v1                        ✅
Map Signal Wiring                    ✅
Controlled Training Baseline         ✅
Terminal-aware Learner Diagnostics   ✅
```

目前 IMPALA pipeline 已具備足夠穩定性，可作為後續 action semantics 修改前的 baseline。

目前尚未證明：

```text
Policy 已收斂
Boss 戰鬥能力已達可接受程度
Win rate 已提升
```

因此此 baseline 主要用於後續架構比較，而非最終性能評估。

---

## 14. Next Stage

等待 UE `/actsegend` protocol 完成後，預計進入 Step 17：

```text
17-A Persistent Action Segment
17-B Continuous Movement Integration
17-C Attack1 / Attack2 Policy Integration
17-D Attack Evaluation
17-E Reward v2
```

加入 Attack1 / Attack2 後：

```text
Action space: 10 → 12
Extra observation: 24 → 26
```

因此現有 checkpoint 不會直接視為最終相容模型。

本文件即作為上述架構變動前的 Pre-Attack / Pre-Persistent-Movement baseline。