<div align="center">
    <h1>
    MiDashengLM
    </h1>
    <b><em>Efficient audio understanding with general audio captions</em></b></em></b>
    <p>
    </p>
    <a href="https://arxiv.org/abs/2508.03983"><img src="https://img.shields.io/badge/arXiv-2508.03983-b31b1b" alt="version"></a>
    <a href="https://huggingface.co/mispeech/MiDashengLM-7B"><img src="https://img.shields.io/badge/HuggingFace-7B-ffcc66" alt="version"></a>
    <a href="https://modelscope.cn/models/midasheng/midashenglm-7b"><img src="https://img.shields.io/badge/ModelScope-7B-7448ce" alt="version"></a>
    <a href="https://modelscope.cn/studios/midasheng/MiDashengLM-7B"><img src="https://img.shields.io/badge/Demo-Gradio-ffcc66" alt="version"></a>
    <a href="https://xiaomi-research.github.io/dasheng-lm/"><img src="https://img.shields.io/badge/Demo-Page-0366d6" alt="version"></a>
</div>

## 📢 News

- **2025-09-04**: vLLM now officially supports MiDashengLM. [Deploy dasheng-lm with vLLM](#deploy-with-vllm).
- ​**2025-09-01**: vLLM integration PR submitted to the official vLLM repository. Preview available in our fork during review. See [Issue #17](https://github.com/xiaomi-research/dasheng-lm/issues/17#issuecomment-3241301450) for details.

## 🔥 Key Highlights

**State-of-the-Art Performance**
   - Outperforms Qwen2.5-Omni-7B, Kimi-Audio-Instruct-7B on **multiple key audio understanding tasks**.

**High Efficiency**
   - **3.2×** throughput speedup at comparable batch sizes compared to Qwen2.5-Omni-7B.
   - **20x** throughput speedup by increasing furhter batchsizes. We tested up to a **batch size=512** for 30s audio input on 80GB GPUs. Baselines only support batch size = 8.
   - Time-to-first-token (TTFT) speedup of up to **4x** compared to Qwen2.5-Omni-7B.

**Caption-based Alignment**
   - Trained with **general audio captions** (instead of ASR transcripts) to achieve holistic audio understanding.

**Full Transparency**
   - **Public-source** training data and reproducible pipeline.
   - Apache License 2.0 for **both research and commercial use**.

<div align="center">
    <img src="fig/capabilities_plot_7b-1.png" width="600">
</div>

## Acknowledgment and Model Foundation

Although MiDashengLM demonstrates superior audio understanding performance and efficiency compared to Qwen2.5-Omni models,
we acknowledge **Qwen2.5-Omni as a remarkable and respected foundational work** in the field.
Our model specifically uses [Qwen2.5-Omni-7B Thinker](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) as the initialization for decoder training, building upon its robust architecture and weight initialization.

The audio encoder is built upon [Dasheng](https://github.com/XiaoMi/dasheng), an open-source audio encoder for general audio understanding with state-of-the-art performance.
**Dasheng serves as the core foundation enabling MiDashengLM's exceptional performance**.

## Framework

MiDashengLM integrates the powerful Dasheng audio encoder with
the Qwen2.5-Omni-7B Thinker decoder through a unique caption-based alignment strategy.
Unlike conventional ASR-driven approaches,
our model leverages general audio captions to capture comprehensive audio representations encompassing speech, environmental sounds, and musical elements
in a unified textual format. This design enables holistic audio understanding while maintaining exceptional computational efficiency.

<img src="fig/Framework-1.png" width="800">

### Why Captions Instead of ASR?

ASR Limitations:
  - Discards huge amount of non-speech audio (music/environmental sounds).
  - Misses paralinguistic info (speaker emotion, acoustic properties).
  - Monotonic alignment provides trivial learning signal.

Caption Advantages:
  - Utilizes all audio content.
  - Captures global audio context.
  - Non-monotonic alignment provides a hard learning signal.

### Novel Open Source Dataset for Training: ACAVCaps

ACAVCaps is a meticulously curated 38,662-hour collection of general audio captions derived from the open-source [ACAV100M audio repository](https://acav100m.github.io/).
While leveraging ACAV100M's extensive raw audio materials, we completely re-engineered the annotation process to create a dataset for holistic audio understanding.
We devide the dataset into six categories:

| Category | Example Caption |
|----------|-----------------|
| Pure Speech | "A female voice narrates historical competition with synthetic modulation" |
| Pure Sound | "Outdoor scene with wind, birds, duck quacking and background noise" |
| Pure Music | "Crowd cheering with electronic synthesizer-driven soundscape" |
| Mixed Music | "The audio features a crowd cheering and clapping alongside electronic music with a synthesizer-driven, dark, and energetic soundscape." |
| Mixed Speech | "A Russian voice demonstrates a synthesizer’s capabilities over an experimental electronic backdrop, explaining its sound design and value in a gritty, vocal-fry tone." |
| Mixed Sound | "A man speaks in English about entering a city and village, accompanied by the sounds of a running vehicle." |

The figure below illustrates our data curation pipeline for ACAVCaps:

<img src="fig/acavcaps-1.png" width="800">

Each caption is generated through a three-step process:

1. **Multi-expert analysis** (speech, vocal, music, acoustics)
2. **LLM reasoning** synthesizing metadata with [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
3. **Filtering** for audio-text consistency with [Dasheng-GLAP](https://github.com/xiaomi-research/dasheng-glap)

We will **release the ACAVCaps dataset** after the ICASSP 2026 review process.

## Usage

### Load Model

```python
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

model_id = "mispeech/midashenglm-7b"

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
```

### Construct Prompt

```python
user_prompt = "Caption the audio."  # You may try any other prompt

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful language and speech assistant."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {
                "type": "audio",
                "path": "/path/to/example.wav",
                # or "url": "https://example.com/example.wav"
                # or "audio": np.random.randn(16000)
            },
        ],
    },
]
```

### Generate Output

```python
import torch

with torch.no_grad():
    model_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=True,
        return_dict=True,
    )
    generation = model.generate(**model_inputs)
    output = tokenizer.batch_decode(generation, skip_special_tokens=True)  # ["An engine is idling."]
```

### Fine-tuning

We appreciate the [ms-swift](https://github.com/modelscope/ms-swift) implementation contributed by [@JimmyMa99](https://github.com/JimmyMa99) in [ms-swift#5325](https://github.com/modelscope/ms-swift/pull/5325).

We are actively developing **MDL-Toolkit**, a user-friendly fine-tuning toolkit scheduled for release in September 2025.

### Deploy with VLLM
vLLM is a fast and easy-to-use library for LLM inference and serving.

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
# Set up using Python-only build (without compilation)
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .

# Full build (with compilation)
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

You can find sample code for offline execution in the VLLM repository [audio_language](https://github.com/vllm-project/vllm/blob/51d5e9be7dbf4d914374447548dd01f9bfb68f89/examples/offline_inference/audio_language.py#L150).

```bash
# Offline inference
python3 examples/offline_inference/audio_language.py -m midashenglm

# Online serving using OpenAI-compatible server
python3 -m vllm.entrypoints.openai.api_server --model mispeech/midashenglm-7b --tensor-parallel-size 1 --served-model-name default --port 8000 --dtype float16 --max_model_len 4096 --trust_remote_code
```

You may want to use [hf-mirror](https://hf-mirror.com/) as a mirror of Hugging Face.

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Results

MiDashengLM delivers solid performance across diverse audio understanding tasks.

### Audio Captioning Results

| Domain   | Dataset        | MiDashengLM    | Qwen2.5-Omni-7B  | Kimi-Audio-Instruct |
|:--------:|:--------------:|:--------------:|:----------------:|:-------------------:|
| Music    | MusicCaps      | **59.71**      | 43.71            | 35.43               |
| Music    | Songdescriber  | **45.39**      | 45.31            | 44.63               |
| Sound    | AudioCaps      | **62.18**      | 60.79            | 49.00               |
| Sound    | ClothoV2       | **49.20**      | 47.55            | 48.01               |
| Sound    | AutoACD        | **66.52**      | 55.93            | 44.76               |

*Metrics: FENSE (higher is better).*

### Audio and Paralinguistic Classification

| Dataset          | Metric | MiDashengLM    | Qwen2.5-Omni-7B | Kimi-Audio-Instruct |
|:----------------:|:------:|:--------------:|:----------------:|:------------------:|
| VoxCeleb1        | ACC↑   | **92.36**      | 59.71            | 82.72              |
| VoxLingua107     | ACC↑   | **93.41**      | 51.03            | 73.65              |
| VoxCeleb-Gender  | ACC↑   | 96.12          | **99.82**        | 99.69              |
| VGGSound         | ACC↑   | **52.11**      | 0.97             | 2.20               |
| Cochlscene       | ACC↑   | **74.06**      | 23.88            | 18.34              |
| NSynth           | ACC↑   | **80.52**      | 60.45            | 38.09              |
| FMA              | ACC↑   | 63.73          | **66.77**        | 27.91              |
| FSDKaggle2018    | ACC↑   | **75.25**      | 31.38            | 24.75              |
| AudioSet         | mAP↑   | **8.86**       | 6.48             | 3.47               |
| FSD50K           | mAP↑   | **37.58**      | 23.87            | 27.23              |

### ASR Performance

| Dataset            | Language    | MiDashengLM | Qwen2.5-Omni-7B | Kimi-Audio-Instruct |
|:------------------:|:-----------:|:--------------:|:------------:|:-------------------:|
| LibriSpeech test-clean  | English | 3.7           | 1.7          | **1.3**             |
| LibriSpeech test-other  | English | 6.2           | 3.4          | **2.4**             |
| People's Speech    | English     | 27.8           | 28.6         | **22.3**            |
| AISHELL2 Mic       | Chinese     | 3.2            | **2.5**      | 2.7                 |
| AISHELL2 iOS       | Chinese     | 2.9            | **2.6**      | **2.6**             |
| AISHELL2 Android   | Chinese     | 3.1            | 2.7          | **2.6**             |
| GigaSpeech2        | Indonesian  | **20.8**       | 21.2         | >100                |
| GigaSpeech2        | Thai        | **36.9**       | 53.8         | >100                |
| GigaSpeech2        | Viet        | **18.1**       | 18.6         | >100                |

*Metrics: WER/CER (lower is better).*

### Question Answering Results

| Dataset      | Subset  | Metric | MiDashengLM    | Qwen2.5-Omni-7B  | Kimi-Audio-Instruct |
|:------------:|:-------:|:------:|:--------------:|:----------------:|:-------------------:|
| MuChoMusic   |         | ACC↑   | **71.35**      | 64.79            | 67.40               |
| MMAU         | Sound   | ACC↑   | 68.47          | 67.87            | **74.17**           |
| MMAU         | Music   | ACC↑   | 66.77          | **69.16**        | 61.08               |
| MMAU         | Speech  | ACC↑   | **63.66**      | 59.76            | 57.66               |
| MMAU         | Average | ACC↑   | **66.30**      | 65.60            | 64.30               |
| MusicQA      |         | FENSE↑ | **62.35**      | 60.60            | 40.00               |
| AudioCaps-QA |         | FENSE↑ | **54.31**      | 53.28            | 47.34               |

*Metrics: Higher is better.*

### Reproduction Instructions

To reproduce our results, we provide:

- Prompts ([prompt.csv](evaluate/prompt.csv))
- Evaluation scripts
- Example JSONL files

#### 1. Install Dependencies for Evaluation (No need this for inference)

```bash
pip install -r requirements.txt
```

#### 2. Generate Model Outputs

Generate responses using the model's official framework with prompts from [prompt.csv](evaluate/prompt.csv).

#### 3. Convert Outputs to JSONL Format

Format model outputs using the [example JSONL](evaluate/jsonl) files:

| Task | Example File |
|------|--------------|
| Automatic Speech Recognition | [MiDashengLM_LibriSpeech_test-clean.jsonl](evaluate/jsonl/MiDashengLM_LibriSpeech_test-clean.jsonl) |
| Single-target Audio Tagging | [MiDashengLM_NSynth.jsonl](evaluate/jsonl/MiDashengLM_NSynth.jsonl) |
| Gender Recognition | [MiDashengLM_VoxCeleb-Gender.jsonl](evaluate/jsonl/MiDashengLM_VoxCeleb-Gender.jsonl) |
| Multi-target Audio Tagging | [MiDashengLM_FSD50K.jsonl](evaluate/jsonl/MiDashengLM_FSD50K.jsonl) |
| Audio Captioning | [MiDashengLM_AutoACD.jsonl](evaluate/jsonl/MiDashengLM_AutoACD.jsonl) |
| Open Audio Question Answering | [MiDashengLM_MusicQA.jsonl](evaluate/jsonl/MiDashengLM_MusicQA.jsonl) |
| Audio QA with Options | [MiDashengLM_MuChoMusic.jsonl](evaluate/jsonl/MiDashengLM_MuChoMusic.jsonl) |

#### 4. Evaluate Results

Execute the corresponding evaluation scripts:

```bash
# Automatic Speech Recognition (WER)
# Uses: lang, text, model_output
python evaluate/wer/compute_wer.py -i evaluate/jsonl/MiDashengLM_LibriSpeech_test-clean.jsonl

# Single-target Audio Tagging (ACC)
# Uses: label, model_output
python evaluate/compute_at_acc.py -i evaluate/jsonl/MiDashengLM_NSynth.jsonl

# Gender Recognition (ACC)
# Uses: label, model_output
python evaluate/compute_gender_acc.py -i evaluate/jsonl/MiDashengLM_VoxCeleb-Gender.jsonl

# Multi-target Audio Tagging (mAP)
# Uses: dataset_name, label, model_output, model_name
python evaluate/compute_map.py -i evaluate/jsonl/MiDashengLM_FSD50K.jsonl

# Audio Captioning (FENSE)
# Uses: audio, text, model_output
python evaluate/compute_fense.py -i evaluate/jsonl/MiDashengLM_AutoACD.jsonl

# Open Audio QA (FENSE)
# Uses: audio, answer, model_output
python evaluate/compute_fense.py -i evaluate/jsonl/MiDashengLM_MusicQA.jsonl

# Audio QA with Options (ACC)
# Uses: answer, model_output
python evaluate/compute_qa_acc.py -i evaluate/jsonl/MiDashengLM_MuChoMusic.jsonl
```

#### 5. Evaluate on MECAT and MMAU benchmarks

Please refer to the official repositories for evaluation on the [MECAT](https://github.com/xiaomi-research/mecat)
and [MMAU](https://github.com/Sakshi113/mmau) benchmarks.

## Efficiency

MiDashengLM demonstrates superior inference efficiency compared to Qwen2.5-Omni-7B,
achieving 3.2× speedup at comparable batch sizes and an overall potential speedup of 20.2× with larger batches.

<img src="fig/batchsize_1_comparison_7b-1.png" width="800">

| Batch Size | MiDashengLM (samples/s) | Qwen2.5-Omni-7B (samples/s) | Speedup |
|:----------:|:-----------------------:|:----------------------------:|:-------:|
| 1          | 0.45                    | 0.36                         | 1.25x   |
| 4          | 1.40                    | 0.91                         | 1.53x   |
| 8          | 2.72                    | 1.15                         | 2.36x   |
| 16         | 5.18                    | OOM                          | -       |
| 32         | 9.78                    | OOM                          | -       |
| 64         | 17.07                   | OOM                          | -       |
| 128        | 22.73                   | OOM                          | -       |
| 200        | 25.15                   | OOM                          | -       |

*Tested on 80GB GPU with 30s audio, 100-token output.*

## Training Data

MiDashengLM is trained exclusively on publicly available datasets across five categories: Speech, Sound and General Audio, Speech and Paralinguistic, Music, and Question Answering. All datasets are listed below with their respective tasks, lengths, and supervised fine-tuning (SFT) usage.

<img src="fig/pretraining_sampling_rates-1.png" width="1200">

### Speech Training Data

This table lists speech-related datasets used for tasks like Automatic Speech Recognition (ASR), keyword spotting (KWS), and speech-to-text translation (S2TT).
The column “SFT?” indicates whether the dataset is used for supervised fine-tuning.

| Data                   | Task      | Length(h) | SFT? |
|:----------------------:|:---------:|:---------:|:----:|
| LibriSpeech            | ASR       | 960       | √    |
| LibriHeavy             | ASR       | 50,000    | X    |
| GigaSpeech             | ASR       | 10,000    | √    |
| GigaSpeech2            | ASR       | 30,000    | √    |
| WeNetSpeech            | ASR       | 10,000    | √    |
| Yodas                  | ASR       | 320,000   | X    |
| CommonVoice-17.0       | ASR       | 5,000     | √    |
| AISHELL-1              | ASR       | 100       | √    |
| AISHELL-2              | ASR       | 1,000     | √    |
| AISHELL-3              | ASR       | 70        | √    |
| LJSpeech-1.1           | ASR       | 37        | X    |
| LibriTTS               | ASR       | 585       | X    |
| MultiLingualSpokenWords| KWS       | 5,000     | X    |
| Emilia                 | ASR       | 101,000   | √    |
| CovoST-v2              | S2TT      | 2,880     | √    |
| Fleurs                 | S2TT      | 1,224     | X    |
| MSR-86K                | ASR, LangID| 86,000    | √    |
| ACAV100M-Speech        | ASR       | 55,754    | X    |
| Must-C                 | ASR,S2TT  | 1,000     | √    |
| MLS                    | ASR       | 50,000    | X    |
| SpgiSpeech             | ASR       | 5,000     | X    |
| PeoplesSpeech          | ASR       | 30,000    | X    |
| KeSpeech               | ASR       | 1,400     | √    |
| LAION-300M             | Caption   | 230,000   | X    |
| **Total**              |           | **997,010**| **258.410** |

### Sound and General Audio Datasets

| Dataset         | Task                     | Length(h) | SFT? |
|:--------------:|:------------------------:|:---------:|:----:|
| FSD50k         | Sound Event              | 77        | √    |
| AudioSet       | Sound Event              | 5,200     |      |
| AudioSet-strong| Sound Event              | 220       | X    |
| VGGSound       | Sound Event              | 540       | √    |
| FSDKaggle2018  | Sound Event              | 20        | √    |
| FSDKaggle2019  | Sound Event              | 100       |      |
| ARCA23k        | Sound Event              | 120       | X    |
| AutoACD        | Audio(Sound) Caption     | 5,200     | √    |
| AudioSetCaps   | Audio(Sound) Caption     | 6,000     | √    |
| SoundVECaps    | Audio(Sound) Caption     | 5,000     | √    |
| WavCaps        | Audio(Sound) Caption     | 7,567     | √    |
| Audiocaps      | Audio(Sound) Caption     | 100       | √    |
| Clothov2       | Audio(Sound) Caption     | 17        | √    |
| TACOS          | Audio(Sound) Caption     | 98        | √    |
| CochlScene     | SoundScape               | 500       | √    |
| BirdSet        | SoundScape               | 7,000     | X    |
| ACAVCaps       | General Caption          | 38,662    | √    |
| **Total**      |                          | **76.421**| **69.081** |

### Speech and Paralinguistic Datasets

| Dataset            | Task                          | Length(hours) | SFT? |
|:------------------:|:-----------------------------:|:-------------:|:----:|
| IEMOCAP            | Emotion                       | 8             | √    |
| Meld               | Emotion                       | 12            | √    |
| SUBESCO            | Emotion                       | 9             | X    |
| RAVDESS-Speech     | Emotion                       | 2             | X    |
| RAVDESS-Song       | Emotion                       | 1             | X    |
| CREMA-D            | Emotion                       | 4             | X    |
| ESD                | Emotion                       | 29            | X    |
| VocalSound         | Vocal sound classification    | 20            | √    |
| NonSpeech7k        | Vocal sound classification    | 3             | √    |
| VoxLingua107       | Language identification       | 7,200         | √    |
| CommonLanguage     | Language identification       | 45            | √    |
| YLACombe           | Language identification       | 5             | X    |
| VoxCeleb1          | Speaker verification          | 76            | √    |
| CNCeleb            | Speaker verification & age    | 2,100         | √    |
| VoxCeleb2          | Speaker verification          | 1,000         | √    |
| VoxBlink1          | Speaker verification          | 1,300         |      |
| VoxBlink2          | Speaker verification          | 2,600         | √    |
| VoxTube            | Language identification       | 5,200         | √    |
| LibriCount         | Speaker counting              | 8             | √    |
| FluentSpeechCommands | Intent classification & gender | 17          | X    |
| SpeechOcean762     | Speaker age                   | 5             | X    |
| ASVSpoof5          | Spoof detection               | 603           | X    |
| **Total**          |                               | **20,247**    | **19,572** |

### Music-Related Datasets

Covers music captioning, genre recognition, instrument classification, and singing style identification.

| Dataset          | Task                              | Length(h) | SFT? |
|:---------------:|:---------------------------------:|:---------:|:----:|
| MusicCaps       | Music Caption                     | 15        | √    |
| Songdescriber   | Music Caption                     | 23        | √    |
| LPMusicCaps-MTT | Music Caption                     | 18        | √    |
| LPMusicCaps-MSD | Music Caption                     | 1,000     | √    |
| VocalSet        | Singing style identification      | 10        | X    |
| FreeMusicArchive| Genre recognition                 | 610       | √    |
| MTG-Jamendo     | Instrument classification Genre recognition | 3,768 | √    |
| NSynth          | Instrument classification         | 360       | √    |
| GoodSounds      | Instrument classification         | 28        | √    |
| chMusic         | Instrument classification         | 1         | √    |
| CTIS            | Instrument classification         | 1         | √    |
| **Total**       |                                   | **5,824** | **5,814** |

### Question Answering Datasets

Used for training on audio-visual QA, environment QA, and music QA tasks. Most support SFT.

| Dataset    | Task            | # QA     | SFT? |
|:---------:|:---------------:|:--------:|:----:|
| AVQA      | Environment QA  | 36,114   | √    |
| ClothoAQA | Environment QA  | 6,175    | √    |
| TACOS+    | Environment QA  | 40,019   | √    |
| MusicQA   | Music QA        | 112,878  | √    |
| SIFT-50M  | Speech QA       | 21,430,000 | √  |
| ACAV-QA   | General QA      | 24,371   | √    |

## Citation

MiDashengLM is under the Apache License 2.0, and we encourage its use in **both research and business applications**.

If you find MiDashengLM useful in your research, please consider citing our work:

```bibtex
@techreport{midashenglm7b,
  title      = {MiDashengLM: Efficient Audio Understanding with General Audio Captions},
  author     = {{Horizon Team, MiLM Plus}}, 
  institution= {Xiaomi Inc.},
  year       = {2025},
  note       = {Contributors: Heinrich Dinkel et al. (listed alphabetically in Appendix B)},
  url        = {https://arxiv.org/abs/2508.03983},
  eprint     = {2508.03983},
}
```
