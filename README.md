# Google Colab 語音辨識專案 (繁體中文高精準與極速版 v2.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Colab-Ready-orange.svg)](https://colab.research.google.com/)
[![CTranslate2](https://img.shields.io/badge/Backend-CTranslate2%20%2F%20FunASR-green.svg)](https://github.com/OpenNMT/CTranslate2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

專為 **Google Colab** 免費與付費 GPU（T4 / V100 / A100）環境設計的高效能、在地化繁體中文語音轉文字（ASR）與字幕（SRT）生成工具。

v2.0 版本全面升級，引入 **Hugging Face 台灣華語在地微調模型** 與 **阿里 SenseVoice-Small 極速引擎**，並解決了歷代版本在 Colab 中的環境相容性問題（如 PyAV 19+ 遞迴報錯、CUDA 12 缺少計算庫、檔名全形標點被置換等）。

---

## 🌟 v2.0 重大升級特點

### 1. 🎯 繁體中文在地化高精準模型（免提示詞、原生標準繁體）
* **`shooding/faster-whisper-large-v3-zh-TW`（🌟 推薦首選）**：
  * 基於 `openai/whisper-large-v3`，採用**數位發展部台灣在地語料庫**（`Taiwan-Tongues-ASR-CE-dataset-zhtw`）透過 LoRA 深度微調。
  * 完美理解台灣腔調、公務用語、各行業專業詞彙（如保險、金融、科技、法規）。
  * 原生輸出台灣標準繁體中文，徹底解決官方 Whisper 容易滑回簡體字或將口音聽成奇怪諧音的問題。
* **`asadfgglie/faster-whisper-large-v3-zh-TW`（⚡ 極速首選）**：
  * 基於最新 **Whisper Large-v3-turbo** 架構（解碼器從 32 層精簡至 4 層，速度提升 3~4 倍），使用 Common Voice 19 繁中資料集微調。
  * 兼具極速推論與高正確性，大幅降低顯存需求。
* **`sensevoice`（阿里 SenseVoice-Small）**：
  * 非自迴歸端到端架構，推論速度比 Whisper 快 10~15 倍。
  * v2.0 內建整合 VAD 時間軸分割與標籤過濾，自動移除 `<|HAPPY|>` 等非語音標籤，直接輸出乾淨字幕。

### 2. 🏷️ 檔名 100% 完全對齊音檔（支援全形標點）
* 徹底修復舊版使用 `slugify` 會將全形標點（如 `：`、`，`、`（）`）和空格強制轉換為減號 `-` 的問題。
* 本地音檔（Google Drive 或路徑清單）直接以主檔名輸出：
  * 音檔：`中小企業保險規劃：經紀人必學的風險管理藍海策略.mp3`
  * 字幕：`中小企業保險規劃：經紀人必學的風險管理藍海策略.srt`
* 確保外部後續自動化程式能精準對應、不會找不到檔案。

### 3. 🛡️ 深度環境相容性與防崩潰保護
* **PyAV 19+ 防遞迴修復**：直接綁定底層 Cython `av.container.core.open`，防止在 Colab 重複執行 Cell 時發生 `RecursionError: maximum recursion depth exceeded`。
* **CUDA 12 動態預載入 + 自動降級**：自動透過 `ctypes` 全域導出 `libcublasLt.so.12` 與 `libcublas.so.12`；若 GPU 出現不可逆例外，自動無縫降級為 CPU 模式完成解譯。
* **清單路徑容錯**：解析清單文字檔時自動去除前後多餘的單雙引號（`"` 或 `'`）。

### 4. ✍️ 專業級中文字幕排版優化
* **去除冗餘空格**：自動移除中文字元之間、中文與標點之間的異常空格（解決 Whisper 官方模型的空白通病），保留英文單字間的空格。
* **自訂字幕折行器**：支援指定每行最大字數（`max_words_per_line`）或最大寬度（`max_line_width`），避免單行字幕過長影響閱讀。
* **臺灣標準繁體把關**：全文字流經 OpenCC 繁體（臺灣標準）二次校準。

---

## 📊 模型對照與選型指南

| 模型名稱 (`modelType`) | 底層架構 | 訓練語料 / 特色 | 速度表現 (Colab GPU) | 繁中精準度評比 | 適用場景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`shooding/faster-whisper-large-v3-zh-TW`**<br>*(v2 預設推薦)* | Whisper Large-v3 | 台灣數位發展部官方語料庫 | 一般 (同 Large-v3) | **⭐⭐⭐⭐⭐ (最高)**<br>台灣腔調與在地詞彙極準 | 正式訪談、課程錄音、專業會議、演講逐字稿 |
| **`asadfgglie/faster-whisper-large-v3-zh-TW`** | Large-v3-turbo | Common Voice 19 繁中微調 | **極快** (比 Large-v3 快約 3~4 倍) | **⭐⭐⭐⭐ (高)**<br>大幅減少原版 Turbo 幻覺 | 大量長音檔、即時性需求高、資源受限場景 |
| **`sensevoice`** | SenseVoice-Small | 30 萬小時多語音 (非自迴歸) | **超極速** (比 Whisper 快 10~15 倍) | **⭐⭐⭐ (中偏高)**<br>標準普通話極佳，依賴 OpenCC 轉繁 | 急需粗略逐字稿、極長音檔初篩 |
| **`large-v3-turbo`** | OpenAI 官方 Turbo | 官方通用多國語言 | 極快 | **⭐⭐⭐ (中)**<br>易漏字或產生簡體 | 多國語言交替、非純中文環境 |

---

## 🚀 快速開始 (Google Colab)

### 步驟 1：建立 Colab 筆記本
1. 前往 [Google Colab](https://colab.research.google.com/) 並新增一個筆記本。
2. 點擊頂部選單 **「執行階段」->「變更執行階段類型」**，將硬體加速器選擇為 **T4 GPU**。

### 步驟 2：貼上程式碼
1. 開啟本專案的 [`whisper_transcribe_colab_v2.py`](file:///d:/@AI%20Projects/@Antigravity/fastWhisper@Colab/whisper_transcribe_colab_v2.py)。
2. 全選並複製所有內容，貼入 Colab 的第一個儲存格中。

### 步驟 3：設定參數並執行
Colab 會自動將頂部參數渲染為互動式表單，常見設定如下：

| 參數名稱 | 說明 | 範例 / 預設值 |
| :--- | :--- | :--- |
| `url` | 音訊來源（支援本機路徑、Google Drive 路徑、清單檔 `.txt`、YouTube 網址/播放清單、Vocaroo） | `/content/drive/MyDrive/record.mp3` |
| `modelType` | 選擇 ASR 模型 | `shooding/faster-whisper-large-v3-zh-TW` |
| `outputFormat` | 輸出格式 (`srt` / `txt` / `.srt.txt`) | `srt` |
| `output_path` | 字幕輸出目錄（可直接指定 Google Drive 資料夾） | `/content/drive/MyDrive/Subtitles/` |
| `overwrite_enable` | 若輸出檔案已存在是否覆蓋 | `True` |
| `word_timestamps` | 是否開啟單字級時間戳（折行器需要啟用此項） | `False` |
| `max_words_per_line` | 每行字幕最大字數（設為 0 表示不限制） | `0` (例如：`18`) |

設定完成後，點擊儲存格左側的 **「執行」按鈕 (Play)** 即可！

---

## 📁 支援的輸入來源

1. **Google Drive 本地音檔**：
   ```text
   /content/drive/MyDrive/MyAudio/演講錄音.mp3
   ```
2. **文字清單批次處理 (`.txt`)**：
   將多個檔案路徑每行一個寫入文字檔中，例如 `/content/paths.txt`：
   ```text
   /content/drive/MyDrive/Audio/檔案一.mp3
   "/content/drive/MyDrive/Audio/檔案二：帶有冒號.mp3"
   https://www.youtube.com/watch?v=xxxxxx
   ```
   *v2.0 支援雙引號/單引號自動過濾，批次處理完畢後會自動打包壓縮為 `.zip` 下載。*
3. **YouTube 影片或播放清單**：
   直接填入 YouTube 網址，程式將自動透過 `yt-dlp` 抽取最高音質並轉錄。
   *(若遇到機器人驗證，可依程式說明上傳 `cookies.txt`)*
4. **Vocaroo 線上錄音**：
   支援 `https://voca.ro/xxxxxx` 或 `https://vocaroo.com/xxxxxx` 網址直接轉錄。

---

## 📂 專案檔案結構

* [`whisper_transcribe_colab_v2.py`](file:///d:/@AI%20Projects/@Antigravity/fastWhisper@Colab/whisper_transcribe_colab_v2.py)：**v2.0 最新版本**，支援台灣在地微調模型、SenseVoice 與完整穩定性修復。
* [`whisper_transcribe_colab.py`](file:///d:/@AI%20Projects/@Antigravity/fastWhisper@Colab/whisper_transcribe_colab.py)：v1.0 穩定版（以官方 faster-whisper `large-v3-turbo` 為主）。
* [`問題與修正.md`](file:///d:/@AI%20Projects/@Antigravity/fastWhisper@Colab/%E5%95%8F%E9%A1%8C%E8%88%87%E4%BF%AE%E6%AD%A3.md)：詳細記錄了所有已知環境錯誤（PyAV、CUDA 12、路徑引號、slugify 標點問題等）之成因與解決方案的維護手冊。

---

## 🔧 常見問題與除錯 (Troubleshooting)

如果您在 Colab 上執行時遇到任何套件衝突或執行錯誤，請參閱 [`問題與修正.md`](file:///d:/@AI%20Projects/@Antigravity/fastWhisper@Colab/%E5%95%8F%E9%A1%8C%E8%88%87%E4%BF%AE%E6%AD%A3.md)。
常見問題速查：
* **`RecursionError: maximum recursion depth exceeded`**：已在 v2.0 修復，直接從 Cython 獲取純淨 `open` 函式。
* **`RuntimeError: Library libcublas.so.12 is not found`**：已在 v2.0 內建動態預載入與 CPU 降級保護。
* **檔名全形標點變成減號**：已在 v2.0 移除 slugify 改採完全對齊。
