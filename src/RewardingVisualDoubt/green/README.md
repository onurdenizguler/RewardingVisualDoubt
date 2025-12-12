# GREEN Score Integration

This module implements the **GREEN (Generative Radiology Report Evaluation and Error Notation)** score evaluation pipeline for assessing the quality of generated radiology reports. It is adapted from the original [Stanford AIMI GREEN repository](https://github.com/Stanford-AIMI/GREEN) with custom modifications to enable efficient deployment via llama.cpp.

## Overview

The GREEN score method uses a specialized language model (RadLlama2-7b) to evaluate candidate radiology reports against reference reports by identifying clinically significant errors, clinically insignificant errors, and matched findings. The final GREEN score is computed as:

```
GREEN Score = matched_findings / (matched_findings + clinically_significant_errors)
```

This provides a continuous accuracy metric in the range [0, 1] that can be used as a reward signal during reinforcement learning training.

## Architecture

### Core Components

1. **Domain Definitions** (`domain.py`)
   - Enumerations for GREEN categories (significant/insignificant errors, matched findings)
   - Error subcategories (false reports, missing findings, location errors, severity errors, etc.)
   - Default generation parameters for RadLlama2-7b

2. **Evaluation** (`evaluate.py`)
   - `parse_error_counts()`: Regex-based parsing of RadLlama2-7b responses to extract error counts
   - `compute_green()`: GREEN score calculation from parsed response
   - **Source**: Adapted from Stanford AIMI GREEN repository

3. **Prompt Engineering** (`prompter.py`)
   - `make_prompt()`: Constructs structured evaluation prompts comparing reference and candidate reports
   - Implements the GREEN evaluation criteria and desired output format
   - **Source**: Adapted from Stanford AIMI GREEN repository

4. **Response Processing** (`postprocessing.py`)
   - `clean_responses()`: Removes chat template artifacts and special tokens from model outputs
   - **Source**: Adapted from Stanford AIMI GREEN repository

### llama.cpp Integration (Custom Implementation)

5. **Server Management** (`llama_server.py`)
   - `start_llama_server()`: Launches llama.cpp server with quantized RadLlama2-7b model
   - `is_server_alive()`: Health check for running server
   - `kill_all_llama_servers()`: Cleanup utility
   - Supports both F16 and Q4_K_M quantization levels
   - **Custom implementation for efficient GPU deployment**

6. **Client API** (`llama_client.py`)
   - `sync_fetch_completion()`: Synchronous API calls to llama.cpp server
   - `async_run_batch()`: Async batch processing for multiple reports
   - `sync_tokenize()`: Token counting via llama.cpp tokenizer API
   - `sync_apply_chat_template()`: Chat template formatting via API
   - **Custom implementation for llama.cpp REST API**

7. **Model Setup** (`llama_setup.py`)
   - `download_radllama_gguf()`: Downloads quantized GGUF models from HuggingFace
   - Uses `mradermacher/GREEN-RadLlama2-7b-GGUF` repository
   - **Custom implementation for GGUF model management**

8. **Pipeline** (`pipeline.py`)
   - `get_green_score_for_single_generated_report()`: End-to-end GREEN score for one report
   - `get_green_score_for_batch_of_generated_reports()`: Batch processing pipeline
   - Orchestrates prompt creation, API calls, response parsing, and score computation

## Why llama.cpp?

The original GREEN implementation uses PyTorch and Transformers for model inference. However, this created dependency conflicts with the RaDialog training pipeline. To resolve this:

1. **Quantization**: RadLlama2-7b is converted to GGUF format with 4-bit quantization (Q4_K_M)
   - Reduces memory footprint from ~14GB to ~4GB
   - Enables concurrent deployment with RaDialog on the same GPU
   - Validation shows minimal impact on GREEN scores (MAE: 0.03, Pearson: 0.96)

2. **API-based Architecture**: llama.cpp runs as a standalone server process
   - Isolated dependencies - no Python package conflicts
   - REST API enables clean separation between RaDialog training and GREEN evaluation
   - Efficient C++ implementation with GPU acceleration

3. **Continuous Batching**: llama.cpp's continuous batching support
   - Enables efficient processing of multiple reports during PPO training
   - Lower latency compared to PyTorch batching for small batch sizes

## Usage

### Setup

```python
from RewardingVisualDoubt.green import llama_setup, llama_server, Quantization

# Download the quantized model (one-time setup)
llama_setup.download_radllama_gguf(
    quantization=Quantization.Q4_K_M
)

# Start the llama.cpp server
llama_server.start_llama_server(
    port=8080,
    quantization=Quantization.Q4_K_M
)

# Wait for server to be ready
import time
while not llama_server.is_server_alive(port=8080):
    time.sleep(1)
```

### Single Report Evaluation

```python
from RewardingVisualDoubt.green import pipeline

reference_report = "The cardiac silhouette is normal. No pleural effusion or pneumothorax."
candidate_report = "Normal cardiac size. Clear lungs bilaterally."

green_score = pipeline.get_green_score_for_single_generated_report(
    gt_report=reference_report,
    generated_report=candidate_report,
    port=8080
)

print(f"GREEN Score: {green_score}")  # e.g., 0.85
```

### Batch Evaluation

```python
from RewardingVisualDoubt.green import pipeline

reference_reports = [...]  # List of ground truth reports
candidate_reports = [...]  # List of generated reports

green_scores = pipeline.get_green_score_for_batch_of_generated_reports(
    gt_reports=reference_reports,
    generated_reports=candidate_reports,
    port=8080
)
```

## Quantization Validation

We validated that 4-bit quantization preserves GREEN score behavior:

| Comparison | MAE ↓ | Pearson ↑ | Spearman ↑ |
|------------|-------|-----------|------------|
| PyTorch F16 vs llama.cpp F16 | 0.05 | 0.94 | 0.95 |
| llama.cpp F16 vs llama.cpp Q4_K_M | 0.03 | 0.96 | 0.97 |
| PyTorch F16 vs llama.cpp Q4_K_M | 0.05 | 0.94 | 0.94 |

These results demonstrate that the quantized model maintains high correlation with the original while enabling practical deployment during RL training.

## Credits

### Original GREEN Method
- **Paper**: "GREEN: Generative Radiology Report Evaluation and Error Notation" by Sophie Ostmeier et al.
- **Repository**: [Stanford-AIMI/GREEN](https://github.com/Stanford-AIMI/GREEN)
- **Components adapted**: Evaluation logic (`evaluate.py`), prompt engineering (`prompter.py`), response parsing (`postprocessing.py`)

### Custom Contributions
- **llama.cpp Integration**: Server management, REST API client, async batching (`llama_server.py`, `llama_client.py`)
- **GGUF Model Setup**: Quantized model download and management (`llama_setup.py`, `shared.py`)
- **Validation Study**: Quantization impact analysis for Q4_K_M vs F16

## Dependencies

- **llama.cpp**: Must be built with GPU support (see main repository's `BUILD_LLAMACPP.md`)
- **HuggingFace Hub**: For downloading GGUF models
- **aiohttp**: For async API calls
- **requests**: For sync API calls
- **psutil**: For server process management

## Configuration

Default configuration in `shared.py`:
- Model repository: `mradermacher/GREEN-RadLlama2-7b-GGUF`
- Default quantization: Q4_K_M
- Default port: 8080
- Context size: 4096 tokens
- GPU layers: 999 (offload all to GPU)

## Troubleshooting

**Server won't start:**
- Ensure llama.cpp is built with GPU support
- Check GPU memory availability (needs ~4GB for Q4_K_M)
- Verify the GGUF model file exists in the models directory

## License

This module combines:
- Original GREEN code (check Stanford AIMI repository for license)
- Custom llama.cpp integration code (MIT License, see repository root)

When using this code, please cite both the original GREEN work and this thesis.
