import asyncio
import boto3
from functools import lru_cache
from app.config import Config
from app.retry import with_retry


@lru_cache(maxsize=1)
def _get_bedrock_client(region_name: str):
    """Create and cache the Boto3 Bedrock Runtime client."""
    return boto3.client("bedrock-runtime", region_name=region_name)


def _apply_guardrail_sync(config: Config, text: str, source: str) -> dict:
    """Apply the configured safety guardrail to the provided text."""
    # Reuse the cached client and apply the guardrail without invoking a foundation model
    client = _get_bedrock_client(config.aws_region)
    return client.apply_guardrail(
        guardrailIdentifier=config.bedrock_guardrail_id,
        guardrailVersion=config.bedrock_guardrail_version,
        source=source,
        content=[{"text": {"text": text}}],
    )


async def validate_input(config: Config, text: str) -> tuple[bool, str]:
    """Validate user input against the configured safety guardrail."""
    response = await with_retry(
        # Run the synchronous Bedrock call in a worker thread to avoid blocking the event loop
        lambda: asyncio.to_thread(_apply_guardrail_sync, config, text, "INPUT"),
        max_retries=config.llm_max_retries,
        delay=config.llm_retry_delay,
    )
    if response.get("action") == "GUARDRAIL_INTERVENED":
        # The guardrail intervened on the user prompt
        return False, "Input blocked by safety guardrail."
    # The guardrail did not intervene
    return True, ""


async def validate_output(config: Config, text: str) -> tuple[bool, str]:
    """Validate LLM-generated output against the configured safety guardrail."""
    response = await with_retry(
        # Run the synchronous Bedrock call in a worker thread to avoid blocking the event loop
        lambda: asyncio.to_thread(_apply_guardrail_sync, config, text, "OUTPUT"),
        max_retries=config.llm_max_retries,
        delay=config.llm_retry_delay,
    )
    if response.get("action") == "GUARDRAIL_INTERVENED":
        # The guardrail intervened on the LLM-generated output
        return False, "Output blocked by safety guardrail."
    # The guardrail did not intervene
    return True, ""
