
# %%
%load_ext autoreload
%autoreload 2
import tokenize
from datasets import Dataset, load_dataset
from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1
from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from RewardingVisualDoubt import dataset, inference, mimic_cxr, prompter, vllm

model= vllm.load_pretrained_llava_model()
tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(model_base=vllm.LLAVA_BASE_MODEL_NAME)

dataset_generator = dataset.create_dataset_generator_from_mimic_cxr_dataset_df(
    mimic_cxr_df=mimic_cxr.create_mimic_cxr_dataset_df(),
    prompter=prompter.build_report_generation_instruction_from_findings,
    tokenizer=tokenizer,
    device=model.device,
)



ds = Dataset.from_generator(generator=dataset_generator)
