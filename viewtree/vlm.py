"""Qwen2.5-VL wrapper for single-turn multi-image QA."""
import numpy as np
import torch
from PIL import Image

MC_SUFFIX = "Answer with the option's letter from the given choices directly."
NUM_SUFFIX = "Do not respond with anything other than a single number!"


class QwenVL:
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda",
                 max_pixels=448 * 448, adapter=None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device
        self.max_pixels = max_pixels

    @torch.no_grad()
    def ask(self, images, prompt, max_new_tokens=32):
        from qwen_vl_utils import process_vision_info

        content = []
        for im in images:
            if isinstance(im, np.ndarray):
                im = Image.fromarray(im)
            content.append({"type": "image", "image": im,
                            "min_pixels": 224 * 224, "max_pixels": self.max_pixels})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        trimmed = out[0, inputs.input_ids.shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)


def build_prompt(row, preamble):
    q = row["question"]
    if row["options"]:
        opts = "\n".join(row["options"])
        return f"{preamble}{q}\n{opts}\n{MC_SUFFIX}"
    return f"{preamble}{q}\n{NUM_SUFFIX}"
