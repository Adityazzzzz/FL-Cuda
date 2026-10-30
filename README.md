# Federated EEG Emotion Decoding with Privacy Audit

Built on top of [FedBS](https://github.com/TianwangJia/FedBS) (Tianwang Jia et al.), extended for the DEAP dataset with a subject re-identification attack engine, Differential Privacy defense, and a Gemini-powered BCI chatbot demo.

<!-- **Research question:** Does Federated Learning actually protect user privacy, or can an attacker still recover biometric identity from the shared model weights? -->

---

## Project Structure

```
.
├── data/DEAP/          # DEAP dataset files (s1.dat ... s32.dat)
├── output/save_models/ # Saved checkpoints, organized by run timestamp
├── src/
│   ├── train.py        # Main training entry point (FL loop + privacy audit)
│   ├── client.py       # Local training + Differential Privacy engine
│   ├── server.py       # Global model aggregation + evaluation
│   ├── attack.py       # Honest-but-curious server re-identification attack
│   ├── evaluate.py     # Standalone metric evaluator for saved checkpoints
│   ├── chatbot.py      # EEG-conditioned Gemini chatbot demo
│   ├── datasets.py     # DEAP and BNCI dataset loaders
│   ├── model.py        # EEGNet architecture
│   ├── SAM.py          # Sharpness-Aware Minimization optimizer
│   └── utils.py        # Early stopping, weight init
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.12, CUDA-capable GPU.

```bash
# 1. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with CUDA (adjust cu124 to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies
pip install -r requirements.txt
```

**Verify GPU access:**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Dataset

This project uses the [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) (32 subjects, 40 EEG channels, 40 video trials each).

- Place the `.dat` files (`s1.dat` to `s32.dat`) in `data/DEAP/`.
- The loader extracts the first 32 EEG channels and binarizes the Valence rating (>5.0 = Class 1, <=5.0 = Class 0).

---

## Training

All training runs use Leave-One-Subject-Out (LOSO) evaluation. Each subject takes a turn as the held-out test subject while the rest train as federated clients.

**Baseline (no Differential Privacy):**
```bash
python src/train.py \
  --model eegnet \
  --global_epochs 10 \
  --local_epochs 2 \
  --data_path "./data/DEAP" \
  --fedbs True \
  --rho 0.1 \
  --channels 32 \
  --samples 8064 \
  --class_num 2 \
  --sub_id "1,2,3,4,5,6,7,8,9"
```

**With Differential Privacy defense:**
```bash
python src/train.py \
  --model eegnet \
  --global_epochs 10 \
  --local_epochs 2 \
  --data_path "./data/DEAP" \
  --fedbs True \
  --rho 0.1 \
  --channels 32 \
  --samples 8064 \
  --class_num 2 \
  --sub_id "1,2,3,4,5,6,7,8,9" \
  --dp True \
  --noise_multiplier 0.5
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--global_epochs` | 200 | Number of federated communication rounds |
| `--local_epochs` | 2 | Local training epochs per client per round |
| `--fedbs` | False | Enable FedBS (SAM optimizer + BN-skip) |
| `--rho` | 0.1 | SAM perturbation radius |
| `--dp` | False | Enable Differential Privacy |
| `--clip_bound` | 1.0 | L2 norm clipping threshold for DP |
| `--noise_multiplier` | 0.1 | Gaussian noise scale for DP |
| `--sub_id` | 1..9 | Comma-separated list of subject IDs to include |
| `--data_path` | — | Path to dataset directory |

Checkpoints are saved to `output/save_models/<timestamp>/Model_ServerSub<id>.pth`.

---

## Evaluation

At the end of every training run, the terminal prints a summary block and saves `evaluation_results.csv` in the checkpoint directory. The summary includes:

- Run config: `global_epochs`, `local_epochs`, FedBS and DP settings
- Per-subject matrix with rows for:
  - Test accuracy (%)
  - Error ratio (%) and (0-1 decimal)
  - Attack success rate per subject (%)
- Aggregate column: Mean and Median across all subjects

**Sample output format:**
```
================================================================================
                           FINAL EVALUATION RESULTS
================================================================================
  Run config:  global_epochs=10  local_epochs=2
               FedBS=ON  (rho=0.1)
               Differential Privacy=OFF
  Subjects:    [1, 2, 3, 4, 5, 6, 7, 8, 9]
--------------------------------------------------------------------------------

[Per-Subject Matrix]
                        1      2      3   ...  Mean  Median
Test Acc (%)        47.50  40.00  60.00  ...  50.56   47.50
Error Ratio (%)     52.50  60.00  40.00  ...  49.44   52.50
Error Ratio (0-1)  0.5250 0.6000 0.4000  ...  0.4944  0.5250
Attack Success (%)  75.00  50.00  62.50  ...  62.50   62.50

--------------------------------------------------------------------------------
Aggregate Metrics:
  Mean Accuracy:        50.56%
  Median Accuracy:      47.50%
  Mean Error Ratio:     49.44%  (0.4944)
  Median Error Ratio:   52.50%  (0.5250)
  Mean Attack Success:  62.50%
  Median Attack Success:62.50%
================================================================================
```

To re-evaluate any saved checkpoint directory without retraining:
```bash
python src/evaluate.py \
  --model_dir "./output/save_models/<timestamp>" \
  --global_epochs 10 \
  --local_epochs 2
```

`--global_epochs` and `--local_epochs` are optional display-only arguments.  
Attack results per subject are read automatically from the `evaluation_results.csv` saved during the original training run if it exists.

---

## Privacy Audit

The training loop includes an honest-but-curious server attack (`src/attack.py`). After each round, the server logs the weight updates from each client and trains a Random Forest classifier to predict which subject sent which update (subject re-identification).

**Baseline result (no DP, 9 subjects):**
- Attack success rate: ~62.5% (random chance = 12.5%)

**With DP (noise_multiplier=0.5):**
- Attack success rate: ~9.7% (below random chance)

---

## Chatbot Demo

Runs a trained EEGNet checkpoint on a single DEAP trial, decodes the subject's emotional valence, and sends the result to Gemini as a hidden system instruction. The same user message is sent twice — once with a neutral instruction and once with the decoded mood — to show how the response changes.

**Requires a Gemini API key.**

```bash
python src/chatbot.py \
  --model_path "./output/save_models/<timestamp>/Model_ServerSub1.pth" \
  --data_path "./data/DEAP" \
  --subject_id 1 \
  --trial_idx 0 \
  --prompt "I have a mountain of work to do and I don't even know where to start." \
  --api_key "YOUR_GEMINI_API_KEY"
```

Alternatively, export the key as an environment variable:
```bash
export GEMINI_API_KEY="YOUR_KEY"
python src/chatbot.py --model_path "./output/save_models/<timestamp>/Model_ServerSub1.pth" ...
```

### Chatbot Arguments

| Argument | Description |
|---|---|
| `--model_path` | Path to a trained `.pth` checkpoint |
| `--data_path` | Path to DEAP directory |
| `--subject_id` | Subject to load EEG from (1-32) |
| `--trial_idx` | Trial index within that subject (0-39) |
| `--prompt` | The user message to send to Gemini |
| `--api_key` | Gemini API key (optional if env var is set) |

---

## FL Algorithms

| Algorithm | How it works |
|---|---|
| **FedAvg** | Server averages client weights each round |
| **FedProx** | Adds a regularization penalty to prevent client models drifting far from the global model |
| **FedBS** | Skips BatchNorm layer aggregation (each client keeps its own BN stats) and uses SAM optimizer to find flatter minima |

---

## Citation

```
Tianwang Jia, Lubin Meng, Siyang Li, Jiajing Liu and Dongrui Wu.
Federated Motor Imagery Classification for Privacy-Preserving Brain-Computer Interfaces.
IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2024.
```