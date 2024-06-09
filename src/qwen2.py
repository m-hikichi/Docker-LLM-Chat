from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline


model_path = "/workspace/models/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
streamer = TextStreamer(tokenizer, skip_prompt=True)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.1,
    top_p=0.95,
    top_k=40,
    max_new_tokens=256,
    streamer=streamer,
)
hf = HuggingFacePipeline(pipeline=pipe)


prompt = PromptTemplate(
    input_variables=["query"],
    template="""<|im_start|>system
あなたは誠実で優秀な日本人のアシスタントです。<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
""",
)

chain = prompt | hf

for text in chain.stream({"query": "有名な犬種をリスト形式で教えてください"}):
    print(text, flush=True, end="")
