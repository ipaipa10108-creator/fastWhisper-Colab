#@title 🇹🇼 OpenAI Whisper & Taiwan ASR 語音辨識 (繁體中文高精準與極速版 v2.0) { vertical-output: true }

#@markdown ---
#@markdown <h3>1. 📁 輸入與輸出設定</h3>
url = "/content/LY_Rec_gdrive_colab_paths.txt" #@param {type:"string"}
#@markdown <small>支援：本機/Google Drive 音檔路徑、批次清單文字檔 (.txt)、YouTube 影片/播放清單網址、Vocaroo 錄音網址</small>
output_path = "/content/drive/MyDrive/LY Class Rec/" # @param {type:"string"}
#@markdown <small>字幕輸出目錄（若目錄不存在將自動新增）</small>
outputFormat = 'srt' #@param ['srt', 'txt', '.srt.txt']
overwrite_enable = True #@param { type: 'boolean' }
start_downloading_immediately = False #@param { type: 'boolean' }
#@markdown <small>辨識完成後是否透過瀏覽器下載（若已輸出至 Google Drive 建議設為 False，避免觸發瀏覽器彈跳視窗封鎖）</small>

#@markdown ---
#@markdown <h3>2. 🎯 語音辨識模型與語言設定</h3>
#@markdown <ul>
#@markdown <li><b>shooding/faster-whisper-large-v3-zh-TW</b>：【🌟 推薦首選】採用數位發展部台灣在地語料庫 LoRA 微調，對台灣腔調、繁體在地詞彙精準度最高。</li>
#@markdown <li><b>asadfgglie/faster-whisper-large-v3-zh-TW</b>：【⚡ 極速推薦】基於最新 Turbo 架構微調 Common Voice 19 繁中，速度極快、顯存低。</li>
#@markdown <li><b>sensevoice</b>：阿里 SenseVoice-Small，極速非自迴歸辨識（推論速度比 Whisper 快 10~15 倍）。</li>
#@markdown <li><b>large-v3-turbo / large-v3</b>：OpenAI 官方原版模型。</li>
#@markdown </ul>
modelType = "shooding/faster-whisper-large-v3-zh-TW" # @param ["shooding/faster-whisper-large-v3-zh-TW", "asadfgglie/faster-whisper-large-v3-zh-TW", "large-v3-turbo", "sensevoice", "large-v3", "large-v2", "medium", "small"] {allow-input: true}
lang = "Chinese" # @param ["Chinese", "自動判斷", "English", "Japanese", "Korean"] {allow-input: true}
initial_prompt = "以下是繁體中文的句子" # @param ["以下是繁體中文的句子", "以下是台灣華語普通話的繁體中文句子。", ""] {allow-input: true}
#@markdown <small>提示詞（Whisper 專用，可用於引導專有名詞；SenseVoice 模型將自動略過）</small>

#@markdown ---
#@markdown <h3>3. ✂️ 字幕排版與折行控制</h3>
#@markdown <small>系統會自動清理中文多餘空白。若設定每行字數或寬度，系統將<b>自動啟用單字級時間戳 (word_timestamps)</b> 進行智慧排版。</small>
verbose = True #@param { type: 'boolean' }
max_words_per_line = 0 # @param {type:"integer"}
max_line_width = 0 # @param {type:"integer"}
max_line_count = 1 # @param {type:"integer"}
word_timestamps = False #@param { type: 'boolean' }
merge_same_text = True #@param { type: 'boolean' }
#@markdown <small>是否自動合併相鄰且文字相同的字幕行（同 SubtitleEdit 功能，自動消除跳針重複並延展時間軸）</small>
merge_same_time = True #@param { type: 'boolean' }
#@markdown <small>是否自動合併時間相同的字幕行（同 SubtitleEdit 功能，重複去重或轉為多行）</small>

#@markdown ---
#@markdown <h3>💡 YouTube 下載提示</h3>
#@markdown <p>如果遇到 "Sign in to confirm you're not a bot" 錯誤，請在本地 Chrome/Firefox 安裝 "Get cookies.txt LOCALLY" 擴充套件，匯出 YouTube cookies 並命名為 <code>cookies.txt</code>，上傳至左側檔案區即可。</p>
#@markdown ---
#@markdown <center><p><font color="blue">▼ 執行進度與輸出內容 ▼</p></center>

# ==============================================================================
# 1. 基礎宣告與背景環境安裝 (防範 ModuleNotFoundError 與版本衝突)
# ==============================================================================
import sys
import os
import re
import urllib.request
import subprocess
import tempfile
import shutil
import importlib
import ctypes
import glob

print(">> 正在檢查並設定最佳執行環境 (v2.0 繁中最佳化版)...")

# 檢查 ffmpeg
if not shutil.which('ffmpeg'):
    print(">> 偵測到環境缺少 ffmpeg，正在背景安裝...")
    try:
        get_ipython().system('apt-get install -y -qq ffmpeg > /dev/null 2>&1')
    except Exception:
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=False)

# 升級 yt-dlp
print(">> 正在檢查並更新 yt-dlp 至最新版本...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "yt-dlp"])
print(">> yt-dlp 更新完成。")

# 免重啟動態安裝與載入機制
def install_and_import(packages_dict):
    installed_any = False
    for import_name, install_name in packages_dict.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f">> 偵測到未安裝 {import_name}，正在背景下載並安裝 {install_name}...")
            cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir"] + install_name.split()
            subprocess.run(cmd, check=True)
            installed_any = True

    if installed_any:
        importlib.invalidate_caches()

# 必備通用套件
common_packages = {
    "opencc": "opencc-python-reimplemented",
    "yt_dlp": "yt-dlp"
}
install_and_import(common_packages)

# 依據模型選擇進行套件安裝
if modelType.lower() == 'sensevoice':
    sensevoice_packages = {
        "funasr": "funasr",
        "modelscope": "modelscope",
        "torch": "torch torchaudio",
    }
    install_and_import(sensevoice_packages)
else:
    whisper_packages = {
        "faster_whisper": "faster-whisper",
        "huggingface_hub": "huggingface_hub"
    }
    install_and_import(whisper_packages)

    # --------------------------------------------------------------------------
    # 修復 PyAV 19+ 移除 metadata_errors 導致 faster-whisper 報錯的相容性問題
    # (注意：取得底層 Cython open，避免 Jupyter/Colab 重複執行 cell 時包裝到自身導致 RecursionError)
    # --------------------------------------------------------------------------
    try:
        import av
        orig_open = None
        try:
            import av.container.core
            orig_open = av.container.core.open
        except Exception:
            try:
                import av.container
                orig_open = av.container.open
            except Exception:
                pass

        if orig_open is not None:
            def _compat_av_open(*args, **kwargs):
                kwargs.pop('metadata_errors', None)
                return orig_open(*args, **kwargs)
            av.open = _compat_av_open
        elif not getattr(av, "_compat_patched", False):
            _orig_av_open = av.open
            def _compat_av_open(*args, **kwargs):
                kwargs.pop('metadata_errors', None)
                return _orig_av_open(*args, **kwargs)
            av.open = _compat_av_open
            av._compat_patched = True
    except Exception:
        pass

    # --------------------------------------------------------------------------
    # 修復 CTranslate2 / CUDA 12 缺少 libcublas.so.12 的問題 (動態預載入)
    # --------------------------------------------------------------------------
    def preload_cuda_libraries():
        try:
            import nvidia.cublas.lib
        except ImportError:
            print(">> 偵測到缺少 CUDA 12 核心庫，正在背景下載安裝 nvidia-cublas-cu12...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"], check=True)
            importlib.invalidate_caches()

        candidate_dirs = []
        try:
            import site
            for sp in site.getsitepackages():
                candidate_dirs.append(os.path.join(sp, "nvidia"))
        except Exception:
            pass

        candidate_dirs.extend([
            f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages/nvidia",
            "/usr/local/lib/python3.13/dist-packages/nvidia",
            "/usr/local/lib/python3.12/dist-packages/nvidia",
            "/usr/local/lib/python3.11/dist-packages/nvidia",
            "/usr/local/lib/python3.10/dist-packages/nvidia",
        ])

        for nv_base in candidate_dirs:
            if os.path.isdir(nv_base):
                for sub in ["cublas", "cudnn"]:
                    lib_dir = os.path.join(nv_base, sub, "lib")
                    if os.path.isdir(lib_dir):
                        cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
                        if lib_dir not in cur_ld:
                            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{cur_ld}" if cur_ld else lib_dir

                        so_files = glob.glob(os.path.join(lib_dir, "*.so*"))
                        so_files.sort(key=lambda x: (0 if "Lt" in x else 1, x))
                        for so in so_files:
                            try:
                                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                            except Exception:
                                pass

    try:
        preload_cuda_libraries()
    except Exception as e:
        print(f">> 預載入 CUDA 庫時略過: {e}")

# ==============================================================================
# 2. 第三方擴充庫載入
# ==============================================================================
from yt_dlp import YoutubeDL
from opencc import OpenCC

try:
    import google.colab.files
    HAS_COLAB_FILES = True
except ImportError:
    HAS_COLAB_FILES = False

print("✅ 環境已就緒，開始進行參數檢查與前置作業...")

# 臺灣標準繁體中文轉換器 (二次把關，確保 100% 臺灣繁體中文)
cc = OpenCC('s2t')

def to_traditional(text):
    if not text:
        return ""
    return cc.convert(text)

# 移除中文字元之間的冗餘空格，保留英文單字間的空格
def clean_chinese_spacing(text):
    if not text:
        return ""
    pattern = re.compile(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])')
    while pattern.search(text):
        text = pattern.sub(r'\1\2', text)
    text = re.compile(r'([\u4e00-\u9fff])\s+([，。！？、；：「」『』（）《》【】])').sub(r'\1\2', text)
    text = re.compile(r'([，。！？、；：「」『』（）《》【】])\s+([\u4e00-\u9fff])').sub(r'\1\2', text)
    return text.strip()

# ==============================================================================
# 3. 參數智慧防呆、校驗與自動修正 (Foolproofing & Guardrails)
# ==============================================================================
# 基礎清理
url = url.strip().strip('"').strip("'")
output_path = output_path.strip().strip('"').strip("'")
initial_prompt = initial_prompt.strip()

# 必填檢查
if not url:
    raise ValueError("❌ [防呆錯誤] 請在上方填寫有效的音訊來源路徑或網址 (url)！")

# 輸出路徑預設與自動建立
if not output_path:
    output_path = "."
os.makedirs(output_path, exist_ok=True)

# Google Drive 自動掛載防呆
if (url.startswith('/content/drive') or output_path.startswith('/content/drive')) and not os.path.exists('/content/drive/MyDrive'):
    print(">> 偵測到使用 Google Drive 但尚未掛載，嘗試自動掛載雲端硬碟...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception as e:
        print(f"⚠️ 無法自動掛載 Google Drive ({e})，若檔案在雲端硬碟請先手動完成授權。")

# 本地檔案/清單檔案存在性檢查 (排除網址)
if not url.startswith('http://') and not url.startswith('https://'):
    if url.lower().endswith('.txt'):
        if not os.path.exists(url):
            raise FileNotFoundError(f"❌ [防呆錯誤] 找不到指定的清單文字檔: {url}\n請確認檔案已上傳至 Colab 或路徑是否正確。")
    elif not os.path.exists(url):
        raise FileNotFoundError(f"❌ [防呆錯誤] 找不到指定的音訊檔案: {url}\n若位於 Google Drive，請確認左側雲端硬碟是否已成功掛載。")

# 智慧防呆：若有設定折行限制，自動啟用 word_timestamps
if (max_words_per_line > 0 or max_line_width > 0) and not word_timestamps:
    if modelType.lower() != 'sensevoice':
        print(">> 💡 [防呆優化] 偵測到已設定每行字數/寬度限制，自動啟用單字級時間戳 (word_timestamps=True) 進行智慧折行。")
        word_timestamps = True

# 模型與參數相容性防呆提示
if modelType.lower() == 'sensevoice':
    if initial_prompt:
        print(f">> ℹ️ [防呆提示] SenseVoice 模型不支援 initial_prompt ('{initial_prompt}')，執行時將自動略過。")
    if max_words_per_line > 0 or max_line_width > 0:
        print(">> ℹ️ [防呆提示] SenseVoice 時間軸由 VAD 分句產生，不支援單字級重新折行，將以語音句子為單位輸出字幕。")

# ==============================================================================
# 4. 輔助函式與變數初始化
# ==============================================================================
audioFile = 'source.mp3'
tempFolder = '.'
title = ''
textFileList = []
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

renameSrtToTxt = False
if outputFormat == '.srt.txt':
    outputFormat = 'srt'
    renameSrtToTxt = True

def getYoutubePlaylistInfo_ydl(url):
    ydl_opts = {'no_warnings': True, 'quiet': True, 'extract_flat': True}
    if os.path.exists("cookies.txt"):
        ydl_opts['cookiefile'] = 'cookies.txt'
    with YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)

def getAudioFromYoutube_ydl(url, output_path, outputFilename):
    with tempfile.TemporaryDirectory() as temp_dir:
        download_command = ["yt-dlp", "-f", "ba", "-P", temp_dir, url]
        if os.path.exists("cookies.txt"):
             print(">> 偵測到 cookies.txt，將使用 Cookie 進行下載...")
             download_command.extend(["--cookies", "cookies.txt"])

        result = subprocess.run(download_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)

        downloaded_files = os.listdir(temp_dir)
        if not downloaded_files:
            raise FileNotFoundError("yt-dlp 下載失敗，目錄中沒有檔案。")
        source_audio_path = os.path.join(temp_dir, downloaded_files[0])

        target_path = os.path.join(output_path, outputFilename)
        convert_command = ["ffmpeg", "-i", source_audio_path, "-q:a", "0", "-map", "a", "-y", target_path]
        subprocess.run(convert_command, check=True, capture_output=True, text=True)

def getVocarooMP3URL(url):
    match = re.search(r'(https://voca\.ro|https://vocaroo\.com)/(\w{12})', url)
    return f"https://media.vocaroo.com/mp3/{match.group(2)}" if match else url

# 針對中文字幕排版特別優化的分行折行器
def split_segment_by_limits(words, max_words, max_width, max_lines=1):
    lines = []
    current_line_words = []
    current_line_len = 0

    for w in words:
        word_text = w['word']
        word_len = len(word_text)

        too_many_words = (max_words > 0 and len(current_line_words) >= max_words)
        too_long = (max_width > 0 and current_line_len + word_len > max_width)

        if too_many_words or too_long:
            if current_line_words:
                lines.append(current_line_words)
                current_line_words = [w]
                current_line_len = word_len
            else:
                current_line_words.append(w)
                current_line_len += word_len
        else:
            current_line_words.append(w)
            current_line_len += word_len

    if current_line_words:
        lines.append(current_line_words)

    srt_segments = []
    if max_lines <= 0:
        max_lines = 1

    for i in range(0, len(lines), max_lines):
        chunk_lines = lines[i:i+max_lines]
        if not chunk_lines:
            continue
        group_start = chunk_lines[0][0]['start']
        group_end = chunk_lines[-1][-1]['end']

        line_texts = []
        for line in chunk_lines:
            line_text = "".join([x['word'] for x in line]).strip()
            line_texts.append(line_text)

        srt_segments.append({
            'start': group_start,
            'end': group_end,
            'text': "\n".join(line_texts)
        })

    return srt_segments

# 仿 SubtitleEdit 智慧合併去重器 (支援合併時間相同行與合併文字相同行)
def merge_subtitle_segments(segments, merge_same_text=True, merge_same_time=True, max_gap=2.0):
    if not segments:
        return []

    # 1. 合併時間相同的行 (Merge lines with same time codes)
    if merge_same_time:
        time_merged = []
        for seg in segments:
            if not time_merged:
                time_merged.append(dict(seg))
                continue
            prev = time_merged[-1]
            # 容許 50 毫秒以內的浮點誤差
            if abs(seg['start'] - prev['start']) < 0.05 and abs(seg['end'] - prev['end']) < 0.05:
                if seg['text'].strip() == prev['text'].strip():
                    continue  # 時間相同且文字相同 -> 去重
                else:
                    # 時間相同但文字不同 -> 依 SubtitleEdit 規範合併為多行
                    prev['text'] = f"{prev['text']}\n{seg['text'].strip()}"
            else:
                time_merged.append(dict(seg))
        segments = time_merged

    # 2. 合併文字相同的相鄰行 (Merge lines with same text)
    if merge_same_text:
        text_merged = []
        for seg in segments:
            if not text_merged:
                text_merged.append(dict(seg))
                continue
            prev = text_merged[-1]
            text_curr = seg['text'].strip()
            text_prev = prev['text'].strip()
            gap = seg['start'] - prev['end']

            # 若相鄰行文字相同，且時間重疊或間隔小於設定門檻 (預設 2 秒內)
            if text_curr == text_prev and gap <= max_gap:
                prev['end'] = max(prev['end'], seg['end'])
            else:
                text_merged.append(dict(seg))
        segments = text_merged

    return segments

def format_ts(ts):
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    ms = int(round((ts % 1) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
        if s == 60:
            m += 1
            s = 0
            if m == 60:
                h += 1
                m = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def segment_to_srt(index, start, end, text):
    return f"{index}\n{format_ts(start)} --> {format_ts(end)}\n{text}\n\n"

# ==============================================================================
# 5. 語音辨識核心邏輯 (支援繁中在地微調模型、SenseVoice 與語言防呆)
# ==============================================================================
def transcribe(audioFilename, output_path, outputFilename, outputFormat):
    print('\n>>>>> 載入模型...')

    if modelType.lower() == 'sensevoice':
        print('\n>>> 正在使用阿里 SenseVoice-Small 模型進行極速辨識...')
        from funasr import AutoModel

        # 語言設定防呆轉換
        sv_lang_map = {
            "Chinese": "zh",
            "English": "en",
            "Japanese": "ja",
            "Korean": "ko",
            "自動判斷": "auto"
        }
        sv_lang = sv_lang_map.get(lang, "auto")

        model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=DEVICE,
            disable_update=True
        )
        result_raw = model.generate(input=audioFilename, language=sv_lang, use_itn=True)

        result = {'segments': []}
        if result_raw and result_raw[0].get("sentence_info"):
            for res in result_raw[0]["sentence_info"]:
                raw_text = res.get('text', '')
                # 清除 SenseVoice 可能產生的情緒與聲音事件標籤 (如 <|HAPPY|>, <|laughter|>)
                cleaned_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()
                seg_text = to_traditional(cleaned_text)

                if verbose:
                    start_s = res.get('start', 0) / 1000.0
                    end_s = res.get('end', 0) / 1000.0
                    print(f"[{format_ts(start_s)} --> {format_ts(end_s)}] {clean_chinese_spacing(seg_text)}")

                result['segments'].append({
                    'start': res.get('start', 0) / 1000.0,
                    'end': res.get('end', 0) / 1000.0,
                    'text': seg_text
                })
        print('>>> SenseVoice 辨識完成！')

    else:
        # Faster-Whisper 架構 (支援台灣華語微調模型與官方版)
        from faster_whisper import WhisperModel
        print(f'\n>>> 正在載入模型 ({modelType})...')

        compute_type = "float16" if DEVICE == "cuda" else "default"

        # 模型名稱對應
        model_mapping = {
            "shooding/faster-whisper-large-v3-zh-TW": "shooding/faster-whisper-large-v3-zh-TW",
            "asadfgglie/faster-whisper-large-v3-zh-TW": "asadfgglie/faster-whisper-large-v3-zh-TW",
            "large-v3-turbo": "turbo",
            "turbo": "turbo",
            "large-v3": "large-v3",
            "large-v2": "large-v2",
            "large": "large-v3",
            "medium": "medium",
            "small": "small"
        }
        model_size = model_mapping.get(modelType, modelType)

        model = WhisperModel(model_size, device=DEVICE, compute_type=compute_type)
        print('\n>>> 開始解譯......')

        args = {
            'beam_size': 5,
            'word_timestamps': word_timestamps,
        }
        if initial_prompt:
            args['initial_prompt'] = initial_prompt

        # 語言對應與繁中專用模型防呆
        if "zh-tw" in modelType.lower() or "zh_tw" in modelType.lower():
            # 繁中微調模型直接指定中文，避免語言偵測的額外開銷與誤判
            if lang == "自動判斷" or lang == "Chinese":
                args['language'] = 'zh'
            else:
                lang_mapping = {"English": "en", "Japanese": "ja", "Korean": "ko"}
                if lang in lang_mapping:
                    args['language'] = lang_mapping[lang]
        elif lang != "自動判斷":
            lang_mapping = {
                "Chinese": "zh",
                "English": "en",
                "Japanese": "ja",
                "Korean": "ko"
            }
            if lang in lang_mapping:
                args['language'] = lang_mapping[lang]

        try:
            segments, info = model.transcribe(audioFilename, **args)
        except RuntimeError as e:
            if "libcublas" in str(e) or "CUDA" in str(e):
                print(f">> 偵測到 CUDA 執行時期異常 ({e})，自動降級為 CPU 模式重新解譯...")
                cpu_model = WhisperModel(model_size, device="cpu", compute_type="int8")
                segments, info = cpu_model.transcribe(audioFilename, **args)
            else:
                raise e

        result = {'segments': []}
        for segment in segments:
            seg_text = to_traditional(segment.text)
            if verbose:
                print(f"[{format_ts(segment.start)} --> {format_ts(segment.end)}] {clean_chinese_spacing(seg_text)}")

            words_list = []
            if word_timestamps and segment.words:
                for w in segment.words:
                    words_list.append({
                        'start': w.start,
                        'end': w.end,
                        'word': to_traditional(w.word)
                    })

            result['segments'].append({
                'start': segment.start,
                'end': segment.end,
                'text': seg_text,
                'words': words_list if word_timestamps else None
            })

        print('>>> Whisper 辨識完成！')

    saveToFile(result, output_path, outputFilename, outputFormat)
    if renameSrtToTxt:
        pathSrt = os.path.join(output_path, f"{outputFilename}.{outputFormat}")
        pathSrtTxt = pathSrt + '.txt'
        if os.path.exists(pathSrt):
            os.rename(pathSrt, pathSrtTxt)

def saveToFile(result, output_path, filename, fileType='srt'):
    if not result or not result.get('segments'):
        print(f"警告：模型 '{modelType}' 未返回任何可辨識的片段。")
        return

    output_filepath = os.path.join(output_path, f"{filename}.{fileType}")
    os.makedirs(output_path, exist_ok=True)

    # 1. 整理基礎字幕片段
    raw_segments = []
    for seg in result['segments']:
        words = seg.get('words')
        if word_timestamps and words and (max_words_per_line > 0 or max_line_width > 0):
            sub_segments = split_segment_by_limits(
                words,
                max_words=max_words_per_line,
                max_width=max_line_width,
                max_lines=max_line_count
            )
            for sub_seg in sub_segments:
                text = clean_chinese_spacing(to_traditional(sub_seg['text']))
                if text:
                    raw_segments.append({'start': sub_seg['start'], 'end': sub_seg['end'], 'text': text})
        else:
            text = clean_chinese_spacing(to_traditional(seg['text']))
            if text:
                raw_segments.append({'start': seg['start'], 'end': seg['end'], 'text': text})

    # 2. 仿 SubtitleEdit 智慧合併去重處理 (文字相同合併 / 時間相同合併)
    processed_segments = merge_subtitle_segments(
        raw_segments,
        merge_same_text=merge_same_text,
        merge_same_time=merge_same_time
    )

    if fileType == 'txt':
        with open(output_filepath, "w", encoding="utf-8") as f:
            full_text = "，".join([seg['text'].strip() for seg in processed_segments])
            full_text = clean_chinese_spacing(to_traditional(full_text))
            f.write(full_text)

    elif fileType == 'srt':
        with open(output_filepath, "w", encoding="utf-8") as f:
            for srt_index, seg in enumerate(processed_segments, start=1):
                f.write(segment_to_srt(srt_index, seg['start'], seg['end'], seg['text']))

    print(f">> 已成功儲存結果至: {output_filepath}")

def getFilenameFromTitle(title):
    # 移除非法檔名字元（如 / \ ? * : " < > |），保留中文、全形標點（如 ：、，、（）等）與英數空格
    clean = re.sub(r'[/\\?%*:|"<>]+', '_', title).strip()
    return clean if clean else "audio"

def getOutputTextFilename(filename):
    ext = outputFormat + ('.txt' if renameSrtToTxt else '')
    return os.path.join(output_path, f"{filename}.{ext}")

def isOutputTextFileExists(filename):
    return os.path.exists(getOutputTextFilename(filename))

def removeAudioFile(filename):
    if os.path.exists(filename): os.remove(filename)

def convertNewLineFormat(filename):
    if not os.path.exists(filename): return
    try:
        with open(filename, 'r+', encoding='utf-8') as f:
            data = f.read().replace('\r\n', '\n').replace('\r', '\n')
            f.seek(0); f.write(data); f.truncate()
    except Exception as e:
        print(f"處理檔案 {filename} 換行時出錯: {e}")

def getAudioAndTranscribe(urlOrFile):
    title, outputFilename, local_audio_file = None, None, None

    if 'https://' in urlOrFile:
        print('>> 處理網址: ' + urlOrFile)
        title = 'web_source_' + str(hash(urlOrFile))[-6:]
        local_audio_file = audioFile

        outputFilename = getFilenameFromTitle(title)
        if isOutputTextFileExists(outputFilename) and not overwrite_enable:
            print('>>>檔案已存在，跳過。')
            textFileList.append(getOutputTextFilename(outputFilename))
            return title

        print('>> 下載影音檔案中...')
        try:
            removeAudioFile(local_audio_file)
            if 'voca.ro' in urlOrFile or 'vocaroo.com' in urlOrFile:
                 urllib.request.urlretrieve(getVocarooMP3URL(urlOrFile), local_audio_file)
            else:
                 getAudioFromYoutube_ydl(urlOrFile, tempFolder, local_audio_file)

            if 'youtube.' in urlOrFile or 'youtu.' in urlOrFile:
                 info = getYoutubePlaylistInfo_ydl(urlOrFile)
                 title = info.get('title', title)
        except subprocess.CalledProcessError as e:
            print(f'>>> 無法由網址下載影音: \n--- yt-dlp 錯誤訊息 ---\n{e.stderr}')
            return None
        except Exception as e:
            print(f'>>> 無法由網址下載影音: {e}')
            return None
        outputFilename = getFilenameFromTitle(title)
    else:
        print('>> 處理本地檔案: ' + urlOrFile)
        title = os.path.splitext(os.path.basename(urlOrFile))[0]
        local_audio_file = urlOrFile
        # 本地檔案檔名與音檔名稱保持 100% 完全一致 (不經過任何置換)
        outputFilename = title

    textFileList.append(getOutputTextFilename(outputFilename))

    if isOutputTextFileExists(outputFilename) and not overwrite_enable:
        print('>>>檔案已存在，跳過辨識。')
        return title

    if os.path.exists(local_audio_file):
        print('\n>> ' + title)
        transcribe(local_audio_file, output_path, outputFilename, outputFormat)
    else:
        print(f'>>>找不到語音檔: {local_audio_file}')

    return title

def parseTxtFileAndTranscribe(txtFile):
    if not os.path.exists(txtFile):
        print('找不到檔案: ' + txtFile)
        return
    with open(txtFile, 'r', encoding='utf-8') as file:
        lines = [line.strip().strip('"').strip("'") for line in file if line.strip()]
    for line in lines:
        full_path = line
        if not 'https://' in line and not os.path.isabs(line) and not os.path.exists(line):
            path_from_list = os.path.join(os.path.dirname(txtFile), line)
            if os.path.exists(path_from_list): full_path = path_from_list
        print(f'\n\n>>>>>> 正在處理清單中的項目: {full_path}')
        getAudioAndTranscribe(full_path)

# ==============================================================================
# 6. 主執行區
# ==============================================================================
removeAudioFile(audioFile)

# 主邏輯處理
if not 'https://' in url and url.lower().endswith('.txt'):
    parseTxtFileAndTranscribe(url)
    title = os.path.splitext(os.path.basename(url))[0]
else:
    try:
        info = getYoutubePlaylistInfo_ydl(url)
        if 'entries' in info and info['entries']:
            print('\n>>檢測到播放列表，開始處理...')
            title = info.get('title', 'youtube_playlist')
            for video in info['entries']:
                 if 'webpage_url' in video:
                     print(f'\n\n>>>>>> 正在處理列表中的影片: {video.get("title", "未知標題")}')
                     getAudioAndTranscribe(video['webpage_url'])
        else:
            title = getAudioAndTranscribe(url)
    except Exception:
        title = getAudioAndTranscribe(url)

# 依設定進行自動下載
if textFileList and start_downloading_immediately:
    print('\n>> 準備下載結果...')
    valid_files = [f for f in textFileList if os.path.exists(f)]
    if not valid_files:
        print(">> 沒有產生任何可下載的檔案。")
    elif HAS_COLAB_FILES:
        for textFile in valid_files: convertNewLineFormat(textFile)
        if len(valid_files) > 1:
            print('\n>> 壓縮並下載辨識結果...')
            zip_filename = getFilenameFromTitle(title) + '-' + outputFormat.replace('.', '') + '.zip'
            zip_cmd = ["zip", "-j", zip_filename] + valid_files
            subprocess.run(zip_cmd, stdout=subprocess.DEVNULL)
            google.colab.files.download(zip_filename)
        else:
            print('\n>> 下載辨識結果...')
            google.colab.files.download(valid_files[0])
    else:
        print(">> 偵測到非 Google Colab 環境，跳過瀏覽器自動下載，檔案已完整保存在 output_path。")
