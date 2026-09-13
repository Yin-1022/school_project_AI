# 基於深度強化學習之遊戲自主代理人與多模型決策架構 

使用學習式 Agent，讓遊戲中 Boss 能根據遊戲當下狀態自主做出決策，
取代傳統上使用 rule-based / behavior tree / scripted 的決策機制。

## Overview

- 使用 Unreal Engine 製作一款具備魂類遊戲Boss戰鬥機制的3D遊戲
- 建立能自主控制 Boss 戰鬥與移動的 Agent
- 建立遊戲與深度強化學習模型之間的即時互動架構。
- 比較不同訓練方法與決策架構的行為及訓練效果

本專案分為遊戲組和 AI 組來分工，運行方式如下：
    遊戲組 — Unreal Engine Game System
    玩家系統 / Boss 戰鬥系統 / 動畫 / 技能 / 場景 / UI / Game Flow
        ↓
        ↓ State、Frame、Event
        ↓
    AI 組 — Boss Decision System
    Observation → Neural Network → Policy → Action
        ↓
        ↓ Action
        ↓
    遊戲組 — Execution
    Boss 執行移動 / 攻擊 / 技能

此 repository 屬於本人負責的 IMPALA 部分，
遊戲部分及 AI 組另一位組員做的模型則不在此repo裡。

## My Contribution
本人擔任專題主負責人及 AI 組組長，負責IMPALA模型及其架構之實作

- Data preparation and collection
- Behavior Cloning baseline implementation
- Distributed Actor-Critic implementation
- IMPALA Actor–Learner architecture
- Unreal Engine–Python communication
- Rollout collection
- Reward design
- Action selection / masking
- Multi-actor execution
- Training and evaluation pipeline

## System Architecture

    UE Game
    ↓ observation / events
    Actor
    ↓ actions
    UE Game

    Actor
    ↓ trajectories
    Learner
    ↓ updated weights
    Actor

## Repository Scope

### Included
- IMPALA implementation
- training pipeline
- rollout / logging tools
- model weights
- evaluation scripts

### Not Included
- Unreal Engine game project
- game assets / gameplay implementation
- A3C implementation (belongs to another teammate)

## Method
### 1. 環境觀測與互動
本專題以 Unreal Engine 建立遊戲環境，並由 Python 端的 IMPALA Actor 負責接收遊戲狀態並進行即時決策。遊戲端會將畫面資訊與特定事件訊號傳送至 Actor，作為模型進行動作判斷的依據。

Actor 在取得當前狀態觀測後，透過訓練中的策略模型進行推論並選擇對應的 Boss 行為，
再將動作指令回傳至遊戲端執行。
藉由此流程，模型便能持續與遊戲環境互動並蒐集後續訓練所需的資料。
### 2. 動作空間
Boss 的行為以離散動作空間表示，包含接近玩家、位移、攻擊等不同類型的決策。
模型每次根據目前觀測輸出一個動作，並由遊戲端將該動作轉換為實際的 Boss 行為。

目前會限制部分不適合當下遊戲狀態的行為，
例如透過 Action Masking 降低不合理動作被選取的機會，並比較其對決策行為的影響。
### 3. IMPALA Actor–Learner 架構
此Repo負責專題的 IMPALA Actor–Learner 架構，將「與環境互動」與「模型訓練」分離。

Actor 負責與遊戲端即時互動，根據當前模型進行推論並蒐集遊戲過程中的觀測、動作、獎勵等資料；Learner 則集中接收 Actor 所產生的 rollout，並用這些資料更新策略模型。

當 Learner 完成模型更新後，會再提供新的模型權重給 Actor，
使 Actor 能持續使用最新策略與遊戲環境互動。
在此設計下，模型可以在不中斷環境互動的情況下持續進行資料蒐集與模型學習。
### 4. Rollout 與訓練流程
Actor 在與遊戲環境互動的過程中，會持續記錄每個決策步驟所產生的資料，
包括觀測、所選動作、獎勵以及相關遊戲事件，並將一段連續的互動資料整理成 rollout。
這些 rollout 會傳給 Learner 作為模型更新的訓練資料。

Reward 設計則根據 Boss 與玩家互動時產生的遊戲事件與行為狀態進行評估，
用以引導模型逐步學習較合理且合適的戰鬥決策。

目前 Reward Design 仍隨遊戲環境與可取得訊號持續調整，
因此現階段著重於建立穩定的訓練流程，並觀察不同獎勵與行為限制對策略表現造成的影響。
### 5. Multi-Actor Training
目前系統已支援多個 Actor 同時執行。
各 Actor 可分別連接至獨立的遊戲實例，平行進行即時推論與 rollout 蒐集，再將取得的訓練資料共同提供給單一 Learner。

此架構能增加單位時間內取得的環境互動資料量，並保留 IMPALA 將環境互動與集中式學習分離的設計，使後續能進一步評估增加 Actor 數量對訓練效率與策略學習的影響。

## Current Capabilities

- IMPALA模型已可執行即時推論，並回傳所選行為給遊戲環境
- 可同時執行多個Actor以訓練單一Learner
- 已整合 Actor–Learner之間的溝通與跨Actor之間模型權重的同步

## Preliminary Experiments

已可比對 baseline policy 和 action-masked variant 
在 decision behavior 和 intervention frequency.

目前仍然只有初步比對，待 Reward 機制與訊號機制完善會再做進一步比對

## Project Status
Work in progress

✅ Model <-> Game connection
✅ Training data collecting methods
✅ Behavior Cloning baseline
✅ Actor-Critic based RL
✅ IMPALA-based distributed training
✅ Multi-actor IMPALA
✅ Hierarchical Reward 機制
✅ Action masking / decision constraint
🚧 Reward mechanism optimization
⬜ Training quality evaluation
⬜ Ensemble learning with teammate's model 

## Repository Structure

- scripts/
    存放程式碼，其中重要檔案：
    ├── stream_infer.py       # IMPALA actor and environment interaction
    ├── impala_learner.py     # Central learner
    ├── rollout_logger.py     # Trajectory logging
    ├── train_impala.py       # Training entry point
- data/meta
    存放模型權重
- data/rollouts
    存放訓練資料
- docs/
    存放架構規則

## Setup / Usage

依賴 UE 遊戲端的環境。

1. 先執行python scripts/stream_infer.py --actor-id <actor_name> --frame-port <Number1> --action-port <Number2> --event-port <Number3>
2. 於Terminal設定遊戲環境並開啟 => 
        "UnrealEditor.exe路徑" \`
        "specialtopic.uproject路徑" \`
        -game \`
        -windowed \`
        -FramePort=<Number1> \`
        -ActionPort=<Number2> \`
        -EventPort=<Number3>
3. 執行scripts/train_impala.py
4. 遊戲端進入戰鬥時，UE和Python端將會自動連線，模型就會運作

## Limitations

目前遊戲環境未包含在 repo，所以無法單獨重現完整遊戲實驗。