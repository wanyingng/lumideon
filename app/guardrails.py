import asyncio
import boto3
from app.config import Config
from app.retry import with_retry


def _apply_guardrail_sync(config: Config, text: str, source: str) -> dict:
    """Apply safety guardrail to text and detect for any policy violations."""
    # Initialize Bedrock Runtime client and invoke AWS Bedrock Guardrail API
    client = boto3.client("bedrock-runtime", region_name=config.aws_region)
    return client.apply_guardrail(
        guardrailIdentifier=config.bedrock_guardrail_id,
        guardrailVersion=config.bedrock_guardrail_version,
        source=source,
        content=[{"text": {"text": text}}],
    )


async def validate_input(config: Config, text: str) -> tuple[bool, str]:
    """Validate user prompt against safety guardrail before sending to LLM."""
    response = await with_retry(
        lambda: asyncio.to_thread(_apply_guardrail_sync, config, text, "INPUT"),
        max_retries=config.llm_max_retries,
        delay=config.llm_retry_delay,
    )
    if response.get("action") == "GUARDRAIL_INTERVENED":
        # User prompt contained harmful information
        return False, "Input blocked by safety guardrail."
    # User prompt is safe
    return True, ""


async def validate_output(config: Config, text: str) -> tuple[bool, str]:
    """Validate LLM generated output against safety guardrail before sending to user."""
    response = await with_retry(
        lambda: asyncio.to_thread(_apply_guardrail_sync, config, text, "OUTPUT"),
        max_retries=config.llm_max_retries,
        delay=config.llm_retry_delay,
    )
    if response.get("action") == "GUARDRAIL_INTERVENED":
        # LLM generated output contained harmful information
        return False, "Output blocked by safety guardrail."
    # LLM generated output is safe
    return True, ""
