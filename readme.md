## 更改advantage 結果

### 將max 拿掉全部由episolon 來確保每一次policy的範圍
### 期末報告Slides (Canva)
- Presentation link: https://www.canva.com/xxxxxxxx
### 以3個參數 0.01 0.1 和 1 來做比較
原本比較是使用strict 的prompt來做測試 發現調整參數對這個prompt沒用，因此後續只以兩個最佳的prompt simple和 baseline 來做比較，以結果來說，在0.1的情況下放大advantage value 對訓練成果有巨大的提升到1的時候 會跟0.01的結果差不多

![BaseLine Result](Baselineresult.png)
圖一:struct prompt 不同episolon的結果
![simple Result](Simpleresult.png)
圖二:simple prompt 不同episolon的結果

* **指令**: `"Think step-by-step, then provide your final answer..."`
* **特性**: 高度一致性，訓練曲線穩定。

| 檔案名稱 | 平均獎勵 (Reward) | 準確率 (Acc) | 格式獎勵 (Format) | 結論 |
| :--- | :---: | :---: | :---: | :--- |
| `simple1.log` | 1.371 | 42.5% | 0.946 | 表現極其穩定 |
| `simple01.log` | 1.351 | 41.1% | 0.940 | 與 simple1 高度重合 |
| `output_simple.txt`| 1.368 | 41.6% | 0.952 | 格式遵循度極高 |

### 2. Instruct / Baseline 提示詞組 (僅格式要求)
* **指令**: `"Respond in the following format..."`
* **特性**: 表現方差大，模型需花費更多 Rollouts 才能習得格式。

| 檔案名稱 | 平均獎勵 (Reward) | 準確率 (Acc) | 格式獎勵 (Format) | 結論 |
| :--- | :---: | :---: | :---: | :--- |
| `instruct1.log` | 0.900 | 20.6% | 0.694 | 訓練中後期波動較大 |
| `instruct01.log` | 1.182 | 30.4% | 0.877 | 該組中表現最佳者 |
| `output_baseline.txt`| 0.844 | 16.6% | 0.678 | 起步極慢，準確率低 |

---
### 1. 算法改進：動態 Policy 範圍控制
為了提升訓練靈敏度，我們進行了以下更動：
* **移除限制**: 拿掉 Advantage 計算中原本的 `max` 限制。
* **控制變因**: 改為完全由 **Epsilon ($\epsilon$)** 來確保每一次 Policy 的更新範圍，防止權重更新過大導致震盪。

### 2. Epsilon ($\epsilon$) 參數比較結果
我們針對 $\epsilon \in \{0.01, 0.1, 1\}$ 進行了消融實驗，發現不同參數對訓練成果有巨大影響：

* **$\epsilon = 0.1$ (最佳配置)**: 
    在此參數下，Advantage Value 被適度放大，顯著提升了訓練效率。無論是準確率還是格式遵循度，在 Rollout 500 之後都有爆發式的成長。
* **$\epsilon = 0.01$ (過於保守)**: 
    更新步長太小，導致模型對於 Reward 訊號反應遲鈍，訓練進度緩慢。
* **$\epsilon = 1.0$ (過於激進)**: 
    更新步長過大，導致 Policy 範圍失控，最終結果回落，表現與 0.01 相似。

> **註記**: 此參數優化在 `Simple` 與 `Baseline` Prompt 上均有顯著效果，但在 `Strict` Prompt 上無明顯作用。
