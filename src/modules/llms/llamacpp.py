from typing import Optional
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate


def load_llamacpp_model(
    model_path: str,
    n_ctx: int = 512,
    seed: int = -1,
    n_batch: Optional[int] = 8,
    n_gpu_layers: Optional[int] = None,
    max_tokens: Optional[int] = 256,
    temperature: Optional[float] = 0.8,
    top_p: Optional[float] = 0.95,
    repeat_penalty: Optional[float] = 1.1,
    top_k: Optional[int] = 40,
    streaming: bool = True,
    verbose: bool = True,
) -> LlamaCpp:
    """
    Args:
        model_path : The path to the Llama model file.
        n_ctx : Token context window.
        seed : Seed. If -1, a random seed is used.
        n_batch : Number of tokens to process in parallel. Should be a number between 1 and n_ctx.
        n_gpu_layers : Number of layers to be loaded into gpu memory. Default None.
        max_tokens : The maximum number of tokens to generate.
        temperature : The temperature to use for sampling.
        top_p : The top-p value to use for sampling.
        repeat_penalty : The penalty to apply to repeated tokens.
        top_k : The top-k value to use for sampling.
        streaming : Whether to stream the results, token by token.
        verbose : Print verbose output to stderr.
    """
    return LlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        top_k=top_k,
        streaming=streaming,
        verbose=verbose,
    )


if __name__ == "__main__":
    # load llm model
    llm_model = load_llamacpp_model(
        model_path="/workspace/models/Llama-3-ELYZA-JP-8B-GGUF/Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
        n_gpu_layers=-1,
        temperature=0.2,
        top_p=0.95,
        top_k=50,
    )

    # create prompt
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""<|start_header_id|>system<|end_header_id|>

あなたは誠実で優秀な日本人のアシスタントです。<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    )

    # construct chain
    chain = prompt | llm_model

    # inference
    for text in chain.stream({"query": "有名な犬種をリスト形式で教えてください"}):
        print(text, flush=True, end="")
    print()
