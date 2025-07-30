# Steps taken (takes around 30 mins)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
scl enable gcc-toolset-9 bash
conda deactivate
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.2e
cmake --build build --config Release -j
```

## To run the server

```bash
/home/guests/deniz_gueler/repos/llama.cpp/build/bin/llama-server --model /home/guests/deniz_gueler/repos/RewardingVisualDoubt/models/radllama_gguf/GREEN-RadLlama2-7b.Q4_K_M.gguf --n-gpu-layers 999
```
