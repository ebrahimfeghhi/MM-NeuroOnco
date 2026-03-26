# Training & Data Setup

## Data Placement

All images must live under `/home/ebrahim/MM-NeuroOnco/images/` as the data root.
Training JSONL files reference images with relative paths like `glioma/16_t1_t2_t1ce_flair_mask/BraTS2021_XXXXX/BraTS2021_XXXXX_t2.png`,
which are resolved as `<data_root>/<image_path>`.

### BraTS2021 Data (Dataset 16)

The BraTS2021 images (t1, t2, t1ce, flair + segmentation mask) must be placed at:

```
images/glioma/16_t1_t2_t1ce_flair_mask/
  BraTS2021_00000/
    BraTS2021_00000_flair.png
    BraTS2021_00000_seg.png
    BraTS2021_00000_t1.png
    BraTS2021_00000_t1ce.png
    BraTS2021_00000_t2.png
  BraTS2021_00002/
  ...
```

After placing the data, regenerate metadata with:

```bash
uv run --with opencv-python-headless --with tqdm python \
  data_processing/metadata_extraction/metadata_16.py \
  --dataset_root /home/ebrahim/MM-NeuroOnco/images \
  --output_json metadata/metadata_16.json \
  --category glioma \
  --keyword 16 \
  --verbose
```

This produces `metadata/metadata_16.json` with 5,000 entries (1,250 cases x 4 modalities).

---

## Training Datasets

Three JSONL files under `training/` correspond to different training regimes:

| File | Used for |
|---|---|
| `train_no_cot_closed.jsonl` | NeuroOnco-GPT (standard SFT) |
| `train_cot_closed.jsonl` | NeuroOnco-GPT-CoT (chain-of-thought SFT) |
| `train_open.jsonl` | Open-ended question variant |

---

## SFT Training (NeuroOnco-GPT / NeuroOnco-GPT-CoT)

Training is done via `instruction_data/run_tumor_1epoch.sh` using the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework with LoRA, for 1 epoch on Qwen3-VL-8B.

```bash
bash instruction_data/run_tumor_1epoch.sh \
  --llamafactory_dir /path/to/LLaMA-Factory \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name <dataset_name_in_llamafactory> \
  --dataset_dir /path/to/llamafactory/data \
  --output_dir ./saves/neuroonco_gpt
```

- To train **NeuroOnco-GPT**: point `--dataset_name` at the `train_no_cot_closed.jsonl` data.
- To train **NeuroOnco-GPT-CoT**: point `--dataset_name` at the `train_cot_closed.jsonl` data.

Key fixed hyperparameters (from the script defaults):

| Param | Value |
|---|---|
| Finetuning type | LoRA (`q_proj`, `v_proj`) |
| Epochs | 1 |
| Learning rate | 1e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Precision | bf16 |
| Template | qwen2_vl |

---

## Evaluation

- `instruction_data/qwen3_vl_eval.py` — single-GPU evaluation
- `instruction_data/Qwen3VL_open_eval_Batch_lora.py` — batched LoRA evaluation

Both scripts accept a `--data_root` argument (or `MMNO_DATA_ROOT` env var) pointing to `/home/ebrahim/MM-NeuroOnco/images`.
