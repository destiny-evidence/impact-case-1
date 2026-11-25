from ._abc import _HuggingfaceClassifierConfig


class ClimateBertConfig(_HuggingfaceClassifierConfig):
    name = 'climatebert'
    model_name = 'climatebert/distilroberta-base-climate-f'


class T5Config(_HuggingfaceClassifierConfig):
    name = 't5-small'
    model_name = 'google-t5/t5-small'
    lr_range = (1e-4, 1e-3)  # T5 is fine-tuned at higher learning rates than the BERT family


class SciBertConfig(_HuggingfaceClassifierConfig):
    name = 'scibert'
    model_name = 'allenai/scibert_scivocab_uncased'


class SciNCLBertConfig(_HuggingfaceClassifierConfig):
    name = 'scincl'
    model_name = 'malteos/scincl'


class ModernBertConfig(_HuggingfaceClassifierConfig):
    name = 'modernbert'
    model_name = 'answerdotai/ModernBERT-base'
    max_length = 2048  # about 99% of title + abstract texts fit; the model supports up to 8,192 tokens
    micro_batch_size = 4  # long sequences: keeps memory per step low, gradient accumulation gives the effective batch of 16


class ClimateModernBertConfig(ModernBertConfig):
    name = 'climatemodernbert'
    model_name = 'CMB-ClimateModernBERT/Merge_Soup_LRD'  # ModernBERT-base with continued pretraining on climate text


# TODO: https://huggingface.co/FacebookAI/roberta-large
# TODO: https://huggingface.co/google/flan-t5-base
