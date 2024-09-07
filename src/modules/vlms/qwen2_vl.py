from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)  # The code of Qwen2-VL has been in the latest Hugging face transformers and we advise you to build from source with command `pip install git+https://github.com/huggingface/transformers`


def load_vlm_model(model_path: str):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor


if __name__ == "__main__":
    # load vlm model
    vlm_model, processor = load_vlm_model(
        model_path="/workspace/models/Qwen2-VL-2B-Instruct"
    )

    # load image
    image = Image.open("./dogs.png")

    # create prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "画像には何匹の犬が映っていますか"},
            ],
        }
    ]

    # preparation for inference
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    output_ids = vlm_model.generate(**inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(output_text)
