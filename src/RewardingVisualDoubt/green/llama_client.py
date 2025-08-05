import asyncio
import dataclasses

import aiohttp
import requests

from .llama_server import PORT


@dataclasses.dataclass
class CompletionResponse:
    index: int
    content: str
    tokens: list
    id_slot: int
    stop: bool
    model: str
    tokens_predicted: int
    tokens_evaluated: int
    generation_settings: dict
    prompt: str
    has_new_line: bool
    truncated: bool
    stop_type: str
    stopping_word: str
    tokens_cached: int
    timings: dict


def create_llama_cpp_response_from_response_json(data: dict) -> CompletionResponse:
    return CompletionResponse(
        index=data.get("index", 0),
        content=data.get("content", ""),
        tokens=data.get("tokens", []),
        id_slot=data.get("id_slot", 0),
        stop=data.get("stop", False),
        model=data.get("model", ""),
        tokens_predicted=data.get("tokens_predicted", 0),
        tokens_evaluated=data.get("tokens_evaluated", 0),
        generation_settings=data.get("generation_settings", {}),
        prompt=data.get("prompt", ""),
        has_new_line=data.get("has_new_line", False),
        truncated=data.get("truncated", False),
        stop_type=data.get("stop_type", ""),
        stopping_word=data.get("stopping_word", ""),
        tokens_cached=data.get("tokens_cached", 0),
        timings=data.get("timings", {}),
    )


create_null_llama_cpp_response = lambda: CompletionResponse(
    index=0,
    content="",
    tokens=[],
    id_slot=0,
    stop=False,
    model="",
    tokens_predicted=0,
    tokens_evaluated=0,
    generation_settings={},
    prompt="",
    has_new_line=False,
    truncated=False,
    stop_type="",
    stopping_word="",
    tokens_cached=0,
    timings={},
)


async def async_fetch_completion(prompt, session, n_predict=2048):
    try:
        async with session.post(
            f"http://localhost:{PORT}/completions",
            json={"prompt": prompt, "n_predict": n_predict, "stream": False},
            timeout=30,
        ) as resp:
            return await resp.json()
    except Exception as e:
        return {"error": str(e)}


async def async_run_batch(prompts, n_predict=2048):
    async with aiohttp.ClientSession() as sess:
        tasks = [async_fetch_completion(p, sess, n_predict) for p in prompts]
        return await asyncio.gather(*tasks)


def sync_tokenize(
    content: str,
    add_special: bool = True,
    with_pieces: bool = True,
    port: int = PORT,
) -> list[int] | list[dict]:
    payload = {
        "content": content,
        "add_special": add_special,
        "with_pieces": with_pieces,
    }
    try:
        response = requests.post(f"http://localhost:{port}/tokenize", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("tokens")
    except requests.exceptions.RequestException:
        return []


def sync_apply_chat_template(
    messages: list[dict],
    port: int = PORT,
) -> str:
    payload = {
        "messages": messages,
    }
    try:
        response = requests.post(
            f"http://localhost:{port}/apply-template", json=payload, timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("prompt")
    except requests.exceptions.RequestException as e:
        print(e)
        return ""


def sync_fetch_completion(
    prompt: str | list[str],
    n_predict: int = 2048,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    port: int = PORT,
) -> CompletionResponse:
    temp = temperature if do_sample else 0.0
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": False,
        "temperature": temp,
        "top_p": top_p if top_p is not None else 1.0,
        # "top_k": 0, # keep as is for the moment even though llama.cpp defaults to 40 which is not what we desire
    }
    try:
        response = requests.post(f"http://localhost:{port}/completions", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return create_llama_cpp_response_from_response_json(data)
    except requests.exceptions.RequestException as e:
        return create_null_llama_cpp_response()
