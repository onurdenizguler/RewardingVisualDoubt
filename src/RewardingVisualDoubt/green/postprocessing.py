def clean_responses(response: str) -> str:
    if "[Explanation]:" in response:
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        if ("[Explanation]:\n    <Explanation>\n" or "[Explanation]:\n<Explanation>") in response:
            response = response.split("[Explanation]:")[1]
        else:
            response = response.split("[Explanation]:")[-1]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    return response.replace("</s>", "").replace("<unk>", "")
