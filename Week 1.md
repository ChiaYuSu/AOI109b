# Week 1 write-up
## 課程前置作業
### 課程內容
- 自動化光學檢測
    - 光學系統 (Optical System)
    - 業師講座 (Speech)
- 機器視覺
    - 深度學習 (Deep Learning)
    - Python
- 實務
    - Practice

### 評量方式
- 期中考：30%
- 期末考：30%
- 實作：40%（採分組方式進行）

### 課程框架
- ANN
- CNN 圖像建模
- RNN 序列數據的建模
- GAN 生成數據

### 工具
- 課程平台：https://moodle.ntust.edu.tw
- 教學互動：https://www.zuvio.com.tw
- 程式執行：https://colab.research.google.com/notebooks/intro.ipynb
- 共用筆記：https://paper.dropbox.com

### 數據取得
- Aidea 工研院：https://aidea-web.tw/
- AIGO 資策會：https://aigo.org.tw/zh-tw

## 自動化光學檢測 (Automated Optical Inspection, AOI)
### 什麼是自動化光學檢測？
- 自動光學檢測，為高速高精度光學影像檢測系統，運用機器視覺做為檢測標準技術，作為改良傳統上以人力使用光學儀器進行檢測的缺點，應用層面包括從高科技產業之研發、製造品管，以至國防、民生、醫療、環保、電力等領域

### 檢測方式
#### 人工視覺檢測 (Manual Vision Inspection, MVI)
- 組成：光源、眼睛、大腦
- 優勢：成本低、色彩敏感度高、動態感測範圍大
- 劣勢：速度慢、可靠度不足、穩定度不足

#### 自動視覺檢測 (Automated Vision Inspection, AVI)
- 組成：光源、攝影機、螢幕、眼睛、大腦
- 優勢：解析度高
- 劣勢：速度慢

#### 自動化光學檢測 (Automated Vision Inspection, AOI)
- 組成：光源、攝影機、電腦
- 優勢：全自動化、可靠度高、穩定度高、可量化、容易整合、解析度高
- 劣勢：成本高

### 優劣分析
|              | 機器視覺                     | 人類視覺 |
|--------------|------------------------------|----------|
| 感測的細膩度 | 劣                           | 優       |
| 工作的複查度 | 低                           | 高       |
| 重複性工作   | 適合                         | 較不適合 |
| 可靠度       | 99.7%                        | 80.0%    |
| 工作光譜     | 可見光、紅外光、紫外光、X 光 | 可見光   |