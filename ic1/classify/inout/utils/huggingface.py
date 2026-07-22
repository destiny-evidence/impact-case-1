import logging

from ic1.core.config import settings

logger = logging.getLogger(__name__)


def ensure_offline_models(models: list[str] | None = None):
    from huggingface_hub import snapshot_download

    if models is None:
        from ic1.classify.inout.models import MODEL_CONFIGS
        from ic1.classify.inout.models.configs._abc import _HuggingfaceClassifierConfig

        models = [config.model_name for config in MODEL_CONFIGS if issubclass(config, _HuggingfaceClassifierConfig)]

    for model in models:
        logger.info(f'Downloading model: {model} so it is available offline in {settings.OFFLINE_MODELS_DIR}')
        snapshot_download(
            repo_id=model,
            repo_type='model',
            cache_dir=settings.OFFLINE_MODELS_DIR,
            force_download=False,
        )
