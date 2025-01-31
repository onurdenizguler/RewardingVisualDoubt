import torch


def generate_report_for_single_study(
    model, tokenizer, input_ids, image_tensor, stopping_criteria
) -> str:

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            use_cache=True,
            max_new_tokens=300,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.pad_token_id,
        )
    pred = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip().replace("</s>", "")
    return pred


def binary_qa_for_single_study(model):
    # TODO
    pass
