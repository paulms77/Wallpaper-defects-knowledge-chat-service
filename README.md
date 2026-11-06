# Wallpaper Defects Knowledge Chat Service

A Korean question-answering model for interior-finishing (wallpaper) defects,
built by QLoRA fine-tuning a 10.7B Korean LLM on a free Colab T4.

| | |
| --- | --- |
| **Competition** | [도배 하자 질의 응답 처리 : 한솔데코 시즌2 AI 경진대회](https://dacon.io/competitions/official/236216/overview/description) |
| **Host** | Hansol Deco · DACON |
| **Base model** | [`yanolja/KoSOLAR-10.7B-v0.2`](https://huggingface.co/yanolja/KoSOLAR-10.7B-v0.2) |
| **Environment** | macOS · Google Colab T4 (free tier) |

<img width="152" alt="KoSOLAR" src="https://github.com/paulms77/Wallpaper-defects-knowledge-chat-service/assets/69188065/4a6c7c59-4fc9-4431-83ac-eb9069d8f280">

---

## The constraint

A 10.7B-parameter model in fp16 needs ~21 GB just for weights. A free Colab T4
has **16 GB**. Fine-tuning it normally is not close to possible.

Everything below follows from that single constraint.

## Fitting the model

### Double quantization

The base model is loaded in 4-bit with nested quantization — the quantization
constants are themselves quantized:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

This brings 21 GB down to roughly 6 GB. `nf4` (4-bit NormalFloat) is designed
for normally-distributed weights and loses less than plain int4.

### QLoRA

With the base weights frozen in 4-bit, only small low-rank adapters train. The
number of trainable parameters drops by orders of magnitude, and so does
optimiser state — which is what actually blows up memory during training.

## Fixing the data

The training set is small, so most of the work is making each example count.

### Re-ranking

Each question comes with five candidate answers of uneven quality, in no
particular order. They are ranked by SentenceTransformer cosine similarity to
the question, and only the top ones are kept as training targets.

The guiding question was literally *"do all the terms in the question appear in
the answer?"* and *"how similar are the question and answer?"* — a crude proxy
for answer quality, but enough to drop the clearly off-topic candidates.

→ Extracted as [Re-Ranking-experiment](https://github.com/paulms77/Re-Ranking-experiment)

### Back-translation

Surviving answers are augmented by round-tripping `ko → en → ko`, which
produces a differently-worded example with the same meaning. Papago is used
rather than Google Translate because it preserves Korean construction
terminology noticeably better; DeepL was also tried.

→ Extracted as [Back-Translation-Papago](https://github.com/paulms77/Back-Translation-Papago)
and [Back-Translation-Google](https://github.com/paulms77/Back-Translation-Google)

### Sentence shuffling

Reordering sentences within a multi-sentence answer gives another valid variant
and discourages the model from memorising a fixed answer template.

### Vocabulary processing and cleaning

Domain terms are searched out and normalised. Specific defects found by
inspection and repaired:

- `TRAIN_222` — answer 5 contains a URL
- `TRAIN_333` — answer 5 contains English text
- several rows where the 장점/단점 (pros/cons) labels in `질문_1` / `질문_2`
  are swapped

That last one matters more than it looks: a mislabelled pros/cons pair teaches
the model the opposite of the truth, and no amount of augmentation fixes a
wrong label.

## Data

| File | Contents |
| --- | --- |
| [`dataset.jsonl`](dataset.jsonl) | Processed prompt/completion pairs ready for fine-tuning |

```json
{"prompt": "면진장치에 사용되는 주요 기술은 무엇인가요?",
 "completion": "진동 격리기는 지면으로부터 진동 에너지를 흡수하여 …"}
```

The original competition data is at the
[DACON data page](https://dacon.io/competitions/official/236216/data).

## Setup

```bash
git clone https://github.com/paulms77/Wallpaper-defects-knowledge-chat-service.git
cd Wallpaper-defects-knowledge-chat-service

pip install transformers==4.30 peft bitsandbytes accelerate datasets \
            sentence-transformers pytorch_lightning
pip install googletrans==4.0.0-rc1
```

> Pin `transformers==4.30`. Later versions change the PEFT and quantization
> interfaces this notebook uses.

For back-translation via the Papago Open API, supply credentials through the
environment — none are committed here:

```bash
export NAVER_CLIENT_ID="..."
export NAVER_CLIENT_SECRET="..."
```

## Usage

```bash
jupyter notebook llm_finetuning_chat_service.ipynb
```

The notebook runs in order: data loading → re-ranking → back-translation
augmentation → cleaning → 4-bit model load → QLoRA fine-tune → inference.

Check the GPU first — the quantization config assumes CUDA:

```python
print_system_specs()
!nvidia-smi
```

## Techniques

- Re-ranking
- Back-translation (Papago, Google, DeepL)
- Sentence shuffling
- Vocabulary processing
- Double quantization
- QLoRA fine-tuning

## Related

| Repository | Relation |
| --- | --- |
| [Hansol-Deco](https://github.com/paulms77/Hansol-Deco) | **Season 3** — construction accident response generation (RAG) |
| [Dacon](https://github.com/paulms77/Dacon) | Season 1 — wallpaper defect type classification (vision) |
| [Re-Ranking-experiment](https://github.com/paulms77/Re-Ranking-experiment) | Answer re-ranking, extracted from this project |
| [Back-Translation-Papago](https://github.com/paulms77/Back-Translation-Papago) | Augmentation, extracted from this project |
| [Back-Translation-Google](https://github.com/paulms77/Back-Translation-Google) | Augmentation, extracted from this project |
