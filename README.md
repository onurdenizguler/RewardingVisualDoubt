# RewardingVisualDoubt

**Confidence Calibration for Medical Large Vision Language Models**

This repository contains the implementation of reinforcement learning-based confidence calibration for **RaDialog**, a medical vision-language model for interactive radiology report generation. Building upon the original RaDialog model by Chantal Pellegrini et al. and the reward framework from the *Rewarding Doubt* study by Paul Stangel et al., this work explores using PPO (Proximal Policy Optimization) to improve model confidence calibration across two key tasks: **Binary Q&A** and **Report Generation**.

## Overview

Medical AI systems must not only provide accurate predictions but also express well-calibrated confidence in their outputs. This project addresses the challenge of confidence calibration in vision-language models by fine-tuning RaDialog using reinforcement learning techniques, PPO specifically.

### Key Results

This work successfully extends the Rewarding Doubt method to multimodal medical vision-language models and achieves substantial improvements in confidence calibration:

**Binary Q&A Task:**
- **ECE: 0.01** (reduced from 0.21 baseline) - a 95% reduction in calibration error
- Successfully validated the Rewarding Doubt approach on LVLMs with binary classification tasks

**Report Generation Task:**
- **ECE: 0.03** with the Quadratic-Blend reward (reduced from 0.43 baseline vanilla verbalization)
- **σ_CBND: 0.44** - Low within-bin dispersion, indicating consistent accuracy within confidence bins
- **D_NCKL: 0.18** - Rich diversity of expressed confidence values, avoiding mode collapse
- **Monotonic calibration curve** - Confidence estimates align with actual accuracy across all confidence levels
- **Strong generalization**: ECE of 0.07 on out-of-distribution test set
- **On-par with white-box methods**: Performance comparable to the Trained Probe method (ECE: 0.03) while maintaining the advantages of a generative approach

**Novel Contributions:**
- **Quadratic-Blend Reward**: A novel reward function that addresses the chronic mode collapse issue in RL-based confidence calibration, achieving superior diversity (D_NCKL: 0.18 vs 0.31 for cross-entropy reward)
- **Difficulty-balanced repetitive training**: Demonstrated that periodic exposure to balanced difficulty levels prevents overfitting to mean accuracy
- **Multimodal PPO**: Successfully adapted TRL's PPO to handle vision token embeddings in LVLMs

The results demonstrate that reinforcement learning-based confidence calibration is not only feasible for multimodal medical LVLMs but can achieve calibration performance matching sophisticated white-box methods while being naturally integrated into the generative process.

**Comparison to Baseline Methods (Report Generation):**

| Method | ECE ↓ | σ_CBND ↓ | D_NCKL ↓ | Notes |
|--------|-------|----------|----------|-------|
| Vanilla Verbalization | 0.43 | 0.55 | 0.73 | Overconfident, poor calibration |
| P(True) | 0.45 | 0.54 | 0.54 | White-box, but uncalibrated |
| Sequence Probability | 0.23 | 0.50 | 0.38 | White-box, moderate performance |
| Trained Probe | 0.03 | 0.44 | 0.17 | White-box, requires internal state access |
| **PPO + Cross-Entropy Reward** | **0.03** | **0.53** | **0.31** | Our method |
| **PPO + Quadratic-Blend Reward** | **0.03** | **0.44** | **0.18** | **Our method (best)** |

*Lower values are better. σ_CBND = Confidence-Bin Normalized Dispersion, D_NCKL = Normalized Confidence KL Divergence*

### Technical Contribution

A key technical contribution of this work is the **modification of the TRL library's PPO implementation** to accommodate vision-language models. The original PPO code from TRL was designed for text-only models, but RaDialog processes both images and text. The modifications in `src/RewardingVisualDoubt/training/llava_ppo.py` enable proper handling of **vision token embeddings** that are inserted into the sequence when processing chest X-ray images. This required careful tracking and masking of image embedding positions during the PPO training loop to ensure correct gradient computation and reward attribution.

Other features are listed as follows:

- **Two Task Domains**: 
  - **Binary Q&A**: Answering yes/no questions about the presence of CheXpert findings in chest X-ray images
  - **Report Generation**: Free-form radiology report generation from chest X-ray images

- **Training Paradigms**:
  - Supervised Fine-Tuning (SFT) for initial confidence-aware training
  - Proximal Policy Optimization (PPO) for reward-based confidence calibration
  - Custom reward functions: log-likelihood based (from *Rewarding Doubt*), continuous accuracy reward, and novel **Quadratic-Blend** reward for report generation

- **Evaluation Metrics**:
  - Calibration metrics (ECE, confidence distribution analysis)
  - GREEN score evaluation for report quality assessment (deployed via llama.cpp)
  - Generated predictions with confidences stored as evaluation results

- **Well-Structured Codebase**: Domain-driven design with clear separation of concerns across:
  - Dataset management and preprocessing
  - Model training (SFT and PPO)
  - Inference and generation
  - Evaluation and calibration metrics
  - Reward computation
  - Infrastructure utilities



## Repository Structure

```
RewardingVisualDoubt/
├── workflows/                      # Training entry points
│   ├── binary_qa/                 # Binary Q&A task workflows
│   │   ├── radialog_binary_qa_ppo_training.py
│   │   ├── radialog_binary_qa_stf_training.py
│   │   └── evaluations/           # Evaluation results (JSON files with predictions)
│   └── report_generation/         # Report generation task workflows
│       ├── radialog_report_generation_ppo_training.py
│       ├── radialog_report_generation_sft_training.py
│       └── evaluations/           # Evaluation results (JSON files with predictions)
├── src/RewardingVisualDoubt/      # Core implementation modules
│   ├── dataset/                   # MIMIC-CXR dataset handling and preprocessing
│   ├── training/                  # PPO and SFT training implementations
│   ├── evaluation/                # Calibration and performance metrics utilities
│   ├── reward.py                  # Reward function definitions
│   ├── response.py                # Response parsing and confidence extraction
│   ├── inference/                 # Generation and inference utilities
│   ├── green/                     # GREEN score evaluation (llama.cpp integration)
│   ├── prompter/                  # Prompt engineering for different tasks
│   ├── vllm/                      # Model loading and management
│   └── infrastructure/            # Utilities and helper functions
└── tests/                         # Unit tests
```

## Methodology

### Reward Functions

#### Binary Q&A Reward

The reward function for Binary Q&A training is based on log-likelihood, directly adopted from the *Rewarding Doubt* study:

```python
reward = log(p_correct) if answer_correct else log(1 - p_correct)
```

Where `p_correct` is the model's expressed confidence. This encourages the model to:
- Express high confidence when correct
- Express low confidence when incorrect
- Learn proper confidence calibration through the PPO algorithm

#### Report Generation Rewards

For report generation, we extended the reward framework with two novel approaches:

1. **Continuous Accuracy Reward**: An adaptation of the log-likelihood reward for continuous accuracy scores obtained from GREEN score evaluation
2. **Quadratic-Blend Reward**: A novel reward function designed specifically for report generation that balances accuracy and confidence calibration using a quadratic (L2-loss) mechanism

The GREEN score is computed using a RadLlama model deployed as an llama.cpp server that accepts API requests for efficient evaluation of generated radiology reports. This provides a continuous measure of report quality that can be used as the accuracy signal in the reward function.

### Training Pipeline

1. **Supervised Fine-Tuning (SFT)**: Initial training to teach the model to output confidence values alongside answers/reports
2. **PPO Training**: Reinforcement learning to optimize confidence calibration using task-specific reward functions
3. **Evaluation**: Generated predictions with confidence values are saved for downstream calibration analysis

## Installation and Setup

### 1. Dependencies 
In a conda managed environment, install with the following:

```bash
conda create -n llava_hf python=3.10
conda activate llava_hf
pip install pip==24.0
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 2. Local Dependency - RaDialog

The package uses the previously developed and unpackaged **RaDialog** [repo](https://huggingface.co/ChantalPellegrini/RaDialog-interactive-radiology-report-generation) by introducing a simple setup.py to its local clone:

```python
from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="radialog",
    packages=find_packages(),
    install_requires=requirements,
)
```

Then, the package is ready to be installed at the root directory of the local clone of the repo by running:

```bash
pip install -e .
```

This installation allows `RewardingVisualDoubt` to import the package by a simple import: `import radialog`

### 3. Installation 
Run the following line at the root directory:
```shell
pip install -e . --config-settings editable_mode=compat
```

### 4. llama.cpp Setup (for Report Generation)

For report generation tasks using GREEN score evaluation, you'll need to set up llama.cpp. See [BUILD_LLAMACPP.md](BUILD_LLAMACPP.md) for detailed instructions.

## Entry Points

Training workflows are located in the `workflows/` directory:
- `workflows/binary_qa/radialog_binary_qa_ppo_training.py` - PPO training for Binary Q&A
- `workflows/binary_qa/radialog_binary_qa_stf_training.py` - SFT training for Binary Q&A
- `workflows/report_generation/radialog_report_generation_ppo_training.py` - PPO training for Report Generation
- `workflows/report_generation/radialog_report_generation_sft_training.py` - SFT training for Report Generation

Key hyperparameters can be configured at the top of each training script. Training configurations for report generation are managed through parameter dataclasses in `src/RewardingVisualDoubt/training/parameters.py`.

**Note**: Evaluation results (JSON files containing model predictions with generated confidences) are stored in the `evaluations/` subdirectories within each workflow. The evaluation pipeline itself is not included in this repository - evaluation utilities are available in the source code modules, but custom evaluation scripts must be written to use them.

## Key Components

### Dataset Management (`src/RewardingVisualDoubt/dataset/`)
- MIMIC-CXR dataset integration
- Custom dataset classes for Binary Q&A and Report Generation
- Preprocessing and prompt engineering
- DataLoader construction with proper collation

### Training (`src/RewardingVisualDoubt/training/`)
- PPO implementation for LLaVA models
- SFT training utilities
- Checkpointing and logging
- Training step orchestration

### Evaluation Utilities (`src/RewardingVisualDoubt/evaluation/`)
- Calibration metrics (ECE, reliability diagrams)
- Confidence distribution analysis
- White-box evaluation tools

### Reward System (`src/RewardingVisualDoubt/reward.py`)
- Log-likelihood based reward functions (from *Rewarding Doubt*)
- Continuous accuracy reward for report generation
- Novel Quadratic-Blend reward for report generation
- Confidence normalization utilities
- Task-specific reward computation

### Response Processing (`src/RewardingVisualDoubt/response.py`)
- Parsing and extraction of confidence values from generated text
- Response validation and formatting
- Handling of different response formats for Binary Q&A and Report Generation

### Prompt Engineering (`src/RewardingVisualDoubt/prompter/`)
- Task-specific prompt construction for Binary Q&A
- Report generation prompts with confidence elicitation
- SFT and inference prompt variations
- Self-guiding confidence request injection

### Inference (`src/RewardingVisualDoubt/inference/`)
- Generation utilities for model inference
- Batch processing and dataloader-based generation
- Stopping criteria and token sampling configuration

### Infrastructure (`src/RewardingVisualDoubt/infrastructure/`)
- Development utilities and helper functions
- Code reloading for interactive development
- System configuration and device management

### Model Management (`src/RewardingVisualDoubt/vllm/`)
- LLaVA model loading with LoRA adapters
- Tokenizer configuration with image support
- Model checkpointing and merging utilities

### GREEN Integration (`src/RewardingVisualDoubt/green/`)
- llama.cpp server management
- GREEN score computation for report quality
- Report-specific evaluation pipeline

## Citations

This work builds upon several key contributions in medical AI and confidence calibration:

```bibtex
@article{pellegrini2024radialog,
  title={RaDialog: A Large Vision-Language Model for Radiology Report Generation and Conversational Assistance},
  author={Pellegrini, Chantal and Eken, Seyran and Ahmad, Ayhan and Mücke, Marie and Inhoffen, Clemens and Ziegelmayer, Sophia and Schnabel, Jonas and Nörenberg, Dirk and Sabel, Bastian and Pfeuffer, Philip and Rueckert, Daniel and Braren, Rickmer and Kaissis, Georgios},
  journal={Medical Imaging with Deep Learning (MIDL)},
  note={Accepted for publication at MIDL 2025},
  year={2024}
}

@article{stangel2025rewarding,
  title={Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models},
  author={Stangel, Paul and Bani-Harouni, David and Pellegrini, Chantal and {\"O}zsoy, Ege and Zaripova, Kamilia and Keicher, Matthias and Navab, Nassir},
  journal={arXiv preprint arXiv:2503.02623},
  year={2025}
}

@article{ostmeier2022green,
  title={GREEN: Generative Radiology Report Evaluation and Error Notation},
  author={Ostmeier, Sophie and others},
  year={2022}
}

@article{johnson2019mimic,
  title={MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
  author={Johnson, Alistair EW and Pollard, Tom J and Berkowitz, Seth J and Greenbaum, Nathaniel R and Lungren, Matthew P and Deng, Chih-ying and Mark, Roger G and Horng, Steven},
  journal={Scientific Data},
  volume={6},
  number={1},
  pages={317},
  year={2019}
}
```

## License

[Add your license information here]

## Acknowledgments

- Original RaDialog model by Chantal Pellegrini et al.
- *Rewarding Doubt* framework by Paul Stangel et al.
- GREEN score evaluation method by Sophie Ostmeier et al.
- MIMIC-CXR dataset
- llama.cpp for efficient inference
- TRL library for PPO implementation (modified for vision-language models)