import clip
import open_clip
import torch

BIOMED_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class BaseClipWrapper:
    def __init__(self, model, preprocess, device):
        self.model = model.to(device)
        self.model.eval()
        self.preprocess = preprocess
        self.device = device

    def encode_image(self, images):
        return self.model.encode_image(images)

    def encode_text(self, tokens):
        return self.model.encode_text(tokens)

    def tokenize(self, texts):
        raise NotImplementedError


class OpenAIClipWrapper(BaseClipWrapper):
    def __init__(self, clip_name, device):
        model, preprocess = clip.load(clip_name, device=device)
        super().__init__(model, preprocess, device)

    def tokenize(self, texts):
        return clip.tokenize(texts)


class BiomedClipWrapper(BaseClipWrapper):
    def __init__(self, device):
        model, preprocess = open_clip.create_model_from_pretrained(BIOMED_MODEL_ID)
        super().__init__(model, preprocess, device)
        self.tokenizer = open_clip.get_tokenizer(BIOMED_MODEL_ID)

    def tokenize(self, texts):
        return self.tokenizer(texts)


def load_clip_encoder(name, device):
    name = name.lower()
    if name == "biomedclip":
        return BiomedClipWrapper(device)
    # default to OpenAI CLIP variants
    return OpenAIClipWrapper(name, device)
