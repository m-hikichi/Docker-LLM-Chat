from typing import Optional
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline


def load_huggingface_pipeline(
    model_path: str,
    max_length: Optional[int] = 20,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 1.0,
) -> HuggingFacePipeline:
    """
    Args:
        model_path : The path to the Llama model file.
        max_length : The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens. Its effect is overridden by max_new_tokens, if also set.
        max_new_tokens : The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        temperature : The value used to modulate the next token probabilities.
        top_k : The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p : If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
    )
    return HuggingFacePipeline(pipeline=pipe)


if __name__ == "__main__":
    # load llm model
    llm_model = load_huggingface_pipeline(
        "/workspace/models/Qwen2-7B-Instruct", max_length=512
    )

    # create prompt
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""<|im_start|>system
あなたは誠実で優秀な日本人のアシスタントです。<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
""",
    )

    # construct chain
    chain = prompt | llm_model

    # inference
    for text in chain.stream({"query": "有名な犬種をリスト形式で教えてください"}):
        print(text, flush=True, end="")
