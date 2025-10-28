
  

# Unofficial WIP Finetuning repo for VibeVoice

  

# Hardware requirements

  

To train a VibeVoice 1.5B LoRa, a machine with at least 16gb VRAM is recommended.

To train a VibeVoice 7B LoRa, a machine with at least 48gb VRAM is recommended.

Keep in mind longer audios increase VRAM requirements

  

# Installation

It is recommended to install this in a fresh environment. Specifically, the Dockerized environment `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` has been tested to work.

  

Transformers version 4.51.3 is known to work, while other versions have errors related to Qwen2 architecture.

  
```
git clone https://github.com/voicepowered-ai/VibeVoice-finetuning

pip install -e .

pip uninstall -y transformers && pip install transformers==4.51.3

(OPTIONAL) wandb login

(OPTIONAL) export HF_HOME=/workspace/hf_models
```

  

# Usage

  

## VibeVoice 1.5B / 7B (LoRA) fine-tuning

  

  

We put some code together for training VibeVoice (7B) with LoRA. This uses the vendored VibeVoice model/processor and trains with a dual loss: masked CE on text tokens plus diffusion MSE on acoustic latents.

  

  

Requirements:

  

- Download a compatible VibeVoice 7B or 1.5b checkpoint (config + weights) and its processor files (preprocessor_config.json) or run straight from HF model.

- A 24khz audio dataset with audio files (target audio), text prompts (transcriptions) and optionally voice prompts (reference audio)


### Training with Local JSONL Dataset

For Linux: 

```
python -m src.finetune_vibevoice_lora \
--model_name_or_path aoi-ot/VibeVoice-Large \
--processor_name_or_path src/vibevoice/processor \
--train_jsonl prompts.jsonl \
--text_column_name text \
--audio_column_name audio \
--output_dir outputTrain3 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 128 \
--learning_rate 2.5e-5 \
--num_train_epochs 10 \
--logging_steps 10 \
--save_steps 200 \
--save_total_limit 10 \
--report_to tensorboard \
--remove_unused_columns False \
--bf16 True \
--do_train \
--gradient_clipping \
--gradient_checkpointing False \
--ddpm_batch_mul 4 \
--diffusion_loss_weight 1.4 \
--train_diffusion_head True \
--ce_loss_weight 0.04 \
--voice_prompt_drop_rate 0.2 \
--lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
--lr_scheduler_type cosine \
--warmup_ratio 0.03 \
--max_grad_norm 0.8 \
--llm_use_8bit True \
--optim adamw_8bit
```

For Windows
```
python -m src.finetune_vibevoice_lora `
--model_name_or_path aoi-ot/VibeVoice-1.5B `
--processor_name_or_path src/vibevoice/processor `
--train_jsonl prompts.jsonl `
--text_column_name text `
--audio_column_name audio `
--output_dir outputTrain3 `
--per_device_train_batch_size 4 `
--gradient_accumulation_steps 32 `
--learning_rate 5e-5 `
--num_train_epochs 10 `
--logging_steps 10 `
--save_steps 200 `
--save_total_limit 10 `
--report_to tensorboard `
--remove_unused_columns False `
--bf16 True `
--do_train `
--gradient_clipping `
--gradient_checkpointing False `
--ddpm_batch_mul 4 `
--diffusion_loss_weight 1.4 `
--train_diffusion_head True `
--ce_loss_weight 0.04 `
--voice_prompt_drop_rate 0.2 `
--lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj `
--lr_scheduler_type cosine `
--warmup_ratio 0.03 `
--max_grad_norm 0.8
```

### JSONL format:

  

You can provide an optional `voice_prompts` key. If it is omitted, a voice prompt will be automatically generated from the target audio.

  

**Example without a pre-defined voice prompt (will be auto-generated):**

`{"text": "Speaker 0: Speaker0 transcription.", "audio": "/workspace/wavs/segment_000000.wav"}`

  

**Example with a pre-defined voice prompt:**

`{"text": "Speaker 0: Speaker0 transcription.", "audio": "/workspace/wavs/segment_000000.wav", "voice_prompts": "/path/to/a/different/prompt.wav"}`

  

**Example with multiple speakers and voice prompts:**

`{"text": "Speaker 0: How is the project coming along?\nSpeaker 1: It's going well, we should be finished by Friday.", "audio": "/data/conversations/convo_01.wav", "voice_prompts": ["/data/prompts/alice_voice_prompt.wav", "/data/prompts/bob_voice_prompt.wav"]}`

  
  
  

# Notes:

  

- Audio is assumed to be 24 kHz; input audio will be loaded/resampled to 24 kHz.

  

- If you pass raw NumPy arrays or torch Tensors as audio (without sampling rate metadata), the collator assumes they are already 24 kHz. To trigger resampling, provide dicts like {"array": <np.ndarray>, "sampling_rate": <int>} or file paths.

  

- Tokenizers (acoustic/semantic) are frozen by default. LoRA is applied to the LLM (Qwen) and optionally to the diffusion head.

  

- The collator builds interleaved sequences with speech placeholders and computes the required masks for diffusion loss.

- If a voice_prompts column is not provided in your dataset for a given sample, a voice prompt is **automatically generated** by taking a random clip from the target audio. This fallback ensures the model's voice cloning ability is maintained. You can override this behavior by providing your own voice prompts.

- Said voice_prompts are randomly dropped during training to improve generalization. Drop rates of 0.2 and 0.25 have been tested with satisfactory results.

  

- The model learns to emit a closing `[speech_end]` token after target placeholders.

  

- For multi‑speaker prompts, ensure `voice_prompts` list order matches `Speaker 0/1/...` tags in your text.

  

- LoRA adapters are saved under `output_dir/lora` after training.

  

  

# Acknowledgements

  

- [VibeVoice](https://github.com/microsoft/VibeVoice)

  

- [chatterbox-finetuning](https://github.com/stlohrey/chatterbox-finetuning)

  
  

## Training Script Arguments

  

Comprehensive list of all the command-line arguments available for the fine-tuning script.

  

### Model & Architecture Arguments

Controls the base model, its configuration, and which components are trained.

  

*  `--model_name_or_path`

*  **What it does:** Specifies the path to the pretrained VibeVoice base model. This can be a local directory or a Hugging Face Hub repository ID.

*  **Required:** Yes.

*  **Example:**

```bash

--model_name_or_path aoi-ot/VibeVoice-Large

```

  

*  `--processor_name_or_path`

*  **What it does:** Specifies the path to the VibeVoice processor configuration. If not provided, it defaults to the `model_name_or_path`.

*  **Example:**

```bash

--processor_name_or_path src/vibevoice/processor

```

  

*  `--train_full_diffusion_head`

*  **What it does:** A boolean flag to enable **full fine-tuning** of the diffusion prediction head. When enabled, all parameters of the diffusion head become trainable.

*  **Example:**

```bash

--train_full_diffusion_head True

```

  

*  `--train_connectors`

*  **What it does:** A boolean flag to enable training of the acoustic and semantic connectors, which bridge different parts of the model.

*  **Example:**

```bash

--train_connectors True

```


  

*  `--lora_r`

*  **What it does:** The rank (`r`) of the LoRA decomposition. A smaller number means fewer trainable parameters.

*  **Default:**  `8`

*  **Example:**

```bash

--lora_r 16

```

  

*  `--lora_alpha`

*  **What it does:** The scaling factor for the LoRA weights. A common practice is to set `lora_alpha` to be four times the value of `lora_r`.

*  **Default:**  `32`

*  **Example:**

```bash

--lora_alpha 64

```

*  `--lora_wrap_diffusion_head`

*  **What it does:** An **alternative** to `--train_full_diffusion_head`. If `True`, it applies LoRA adapters to the diffusion head instead of fine-tuning it fully, enabling more parameter-efficient training of the head. Must only use `--train_full_diffusion_head` or `--lora_wrap_diffusion_head`

*  **Default:**  `False`

  
  
  

*  `--layers_to_freeze`

*  **What it does:** Comma-separated indices of diffusion head layers to freeze (e.g., '0,1,5,7,8'). [Diffusion head layer indices](https://github.com/voicepowered-ai/VibeVoice-finetuning/blob/main/diff_head_layers.txt)

*  **Default:**  `None`

### Data & Processing Arguments

Defines the dataset to be used, its structure, and how it should be processed.

  

*  `--train_jsonl`

*  **What it does:** Path to your local training data file in JSONL (JSON Lines) format. Each line should be a JSON object with keys for text and audio path.

*  **Example:**

```bash

--train_jsonl prompts.jsonl

```

  

*  `--validation_jsonl`

*  **What it does:** Optional path to a local validation data file in JSONL format.

*  **Example:**

```bash

--validation_jsonl validation_prompts.jsonl

```

  

*  `--text_column_name`

*  **What it does:** The name of the key in your JSONL file that contains the text transcription/prompt.

*  **Default:**  `text`

*  **Example:**

```bash

--text_column_name "prompt"

```

  

*  `--audio_column_name`

*  **What it does:** The name of the key in your JSONL file that contains the path to the audio file.

*  **Default:**  `audio`

*  **Example:**

```bash

--audio_column_name "file_path"

```

  

*  `--voice_prompt_drop_rate`

*  **What it does:** The probability (from 0.0 to 1.0) of randomly dropping the conditioning voice prompt during training. This acts as a regularizer.

*  **Default:**  `0.0`

*  **Example:**

```bash

--voice_prompt_drop_rate 0.2

```

  

### Core Training Arguments

Standard Hugging Face `TrainingArguments` that control the training loop, optimizer, and saving.

  

*  `--output_dir`

*  **What it does:** The directory where model checkpoints and final outputs will be saved.

*  **Required:** Yes.

*  **Example:**

```bash

--output_dir output_model

```

  

*  `--per_device_train_batch_size`

*  **What it does:** The number of training examples processed per GPU in a single step.

*  **Example:**

```bash

--per_device_train_batch_size 8

```

  

*  `--gradient_accumulation_steps`

*  **What it does:** The number of forward passes to accumulate gradients for before performing an optimizer step. This effectively increases the batch size without using more VRAM.

*  **Example:**

```bash

--gradient_accumulation_steps 16

```

  

*  `--learning_rate`

*  **What it does:** The initial learning rate for the optimizer.

*  **Example:**

```bash

--learning_rate 2.5e-5

```

  

*  `--num_train_epochs`

*  **What it does:** The total number of times to iterate over the entire training dataset.

*  **Example:**

```bash

--num_train_epochs 5

```

  

*  `--logging_steps`

*  **What it does:** How often (in steps) to log training metrics like loss.

*  **Example:**

```bash

--logging_steps 10

```

  

*  `--save_steps`

*  **What it does:** How often (in steps) to save a model checkpoint.

*  **Example:**

```bash

--save_steps 100

```

  

*  `--report_to`

*  **What it does:** The integration to report logs to. Can be `wandb`, `tensorboard`, or `none`.

*  **Example:**

```bash

--report_to wandb

```

  

*  `--remove_unused_columns`

*  **What it does:** Whether to remove columns from the dataset not used by the model's `forward` method. **This must be set to `False`** for this script to work correctly.

*  **Example:**

```bash

--remove_unused_columns False

```

  

*  `--bf16`

*  **What it does:** Enables mixed-precision training using `bfloat16`. This speeds up training and reduces memory usage on compatible GPUs (NVIDIA Ampere series and newer).

*  **Example:**

```bash

--bf16 True

```

  

*  `--gradient_checkpointing`

*  **What it does:** A memory-saving technique that trades compute for memory. Useful for training large models on limited VRAM.

*  **Example:**

```bash

--gradient_checkpointing True

```

  

*  `--lr_scheduler_type`

*  **What it does:** The type of learning rate schedule to use (e.g., `linear`, `cosine`, `constant`).

*  **Example:**

```bash

--lr_scheduler_type cosine

```

  

*  `--warmup_ratio`

*  **What it does:** The proportion of total training steps used for a linear warmup from 0 to the initial learning rate.

*  **Example:**

```bash

--warmup_ratio 0.03

```

  

### Custom VibeVoice Training Arguments

Special arguments to control VibeVoice-specific training behaviors.

  

*  `--gradient_clipping`

*  **What it does:** A custom boolean flag that acts as the master switch for gradient clipping. If you include this flag, the value from `--max_grad_norm` will be used to prevent exploding gradients.

*  **Example:**

```bash

--gradient_clipping

```

*  `--max_grad_norm`

*  **What it does:** The maximum value for gradient clipping. Only active if `--gradient_clipping` is also used.

*  **Default:**  `1.0`

*  **Example:**

```bash

--max_grad_norm 0.8

```

  

*  `--diffusion_loss_weight`

*  **What it does:** A float that scales the importance of the diffusion loss (for speech generation quality) in the total loss calculation.

*  **Example:**

```bash

--diffusion_loss_weight 1.4

```

  

*  `--ce_loss_weight`

*  **What it does:** A float that scales the importance of the Cross-Entropy loss (for text prediction accuracy) in the total loss calculation.

*  **Example:**

```bash

--ce_loss_weight 0.04

```

  

*  `--ddpm_batch_mul`

*  **What it does:** An integer multiplier for the batch size used specifically within the diffusion process.

*  **Example:**

```bash

--ddpm_batch_mul 4


```


