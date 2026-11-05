import asyncio
import boto3
from botocore.config import Config as BotoConfig
from botocore.client import BaseClient
from functools import lru_cache

from app.config import Config


@lru_cache(maxsize=1)
def _get_bedrock_client(region_name: str, max_retries: int) -> BaseClient:
    """Creates and caches the Boto3 Bedrock Runtime client.

    Args:
        region_name: The AWS region in which the Bedrock Runtime client operates.
        max_retries: The maximum number of retries after the initial request.

    Returns:
        A cached Bedrock Runtime client configured with the specified retry and timeout settings.
    """
    return boto3.client(
        "bedrock-runtime",
        region_name=region_name,
        config=BotoConfig(
            retries={
                "mode": "standard",
                "total_max_attempts": max_retries + 1,
            },
            connect_timeout=5,
            read_timeout=15,
        ),
    )


def _apply_guardrail_sync(config: Config, text: str, source: str) -> dict:
    """Applies the configured safety guardrail to the provided text.

    Args:
        config: The application configuration instance.
        text: The text content to evaluate against the guardrail.
        source: The content source, either "INPUT" or "OUTPUT".

    Returns:
        A dictionary containing the Bedrock ApplyGuardrail API response.
    """
    # Reuse the cached client and apply the guardrail without invoking a foundation model
    client = _get_bedrock_client(
        config.aws_region,
        config.llm_max_retries,
    )

    return client.apply_guardrail(
        guardrailIdentifier=config.bedrock_guardrail_id,
        guardrailVersion=config.bedrock_guardrail_version,
        source=source,
        content=[{"text": {"text": text}}],
    )


async def validate_input(config: Config, text: str) -> tuple[bool, str]:
    """Validates user input against the configured safety guardrail.

    Args:
        config: The application configuration instance.
        text: The user input text to evaluate.

    Returns:
        A tuple containing:
            - bool: True if the guardrail does not intervene, False otherwise.
            - str: An explanatory message if the guardrail intervenes; otherwise, an empty string.
    """
    # Run the synchronous Bedrock call in a worker thread to avoid blocking the event loop
    response = await asyncio.to_thread(
        _apply_guardrail_sync,
        config,
        text,
        "INPUT",
    )

    if response.get("action") == "GUARDRAIL_INTERVENED":
        # The guardrail intervened on the user prompt
        return False, "Input blocked by safety guardrail."
    # The guardrail did not intervene
    return True, ""


async def validate_output(config: Config, text: str) -> tuple[bool, str]:
    """Validates LLM-generated output against the configured safety guardrail.

    Args:
        config: The application configuration instance.
        text: The LLM-generated text to evaluate.

    Returns:
        A tuple containing:
            - bool: True if the guardrail does not intervene, False otherwise.
            - str: An explanatory message if the guardrail intervenes; otherwise, an empty string.
    """
    # Run the synchronous Bedrock call in a worker thread to avoid blocking the event loop
    response = await asyncio.to_thread(
        _apply_guardrail_sync,
        config,
        text,
        "OUTPUT",
    )

    if response.get("action") == "GUARDRAIL_INTERVENED":
        # The guardrail intervened on the LLM-generated output
        return False, "Output blocked by safety guardrail."
    # The guardrail did not intervene
    return True, ""
