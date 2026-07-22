import logging
from ic1.core.config import settings

logger = logging.getLogger(__name__)


def ensure_offline_models(models: list[str] | None = None, force: bool = False) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.file_download import repo_folder_name

    if models is None:
        from ic1.classify.inout.configs import MODEL_CONFIGS
        from ic1.classify.inout.configs._abc import _HuggingfaceClassifierConfig

        models = [config.model_name for config in MODEL_CONFIGS.values() if issubclass(config, _HuggingfaceClassifierConfig)]

    for model in models:
        model_dir = settings.OFFLINE_MODELS_DIR / repo_folder_name(repo_id=model, repo_type='model')

        if model_dir.exists() and not force:
            logger.info(f'Model "{model}" already cached in {model_dir}')
            continue

        logger.info(f'Downloading model: {model} so it is available offline in {settings.OFFLINE_MODELS_DIR}')
        snapshot_download(
            repo_id=model,
            repo_type='model',
            cache_dir=settings.OFFLINE_MODELS_DIR,
            force_download=False,
        )
