from fastapi import Request, HTTPException


async def require_api_key(request: Request) -> None:
    """Check if a valid API key is present in the client request headers for authentication purposes."""
    # Retrieve the application configuration from the request
    config = request.app.state.config
    # Check if API key authentication is enabled; skip check if no key is configured
    if not config.api_key:
        return
    # Validate the API key present in the client request headers matches the configured key in the Secrets Manager
    key = request.headers.get("X-API-Key", "")
    if key != config.api_key:
        # If the API key does not match, raise an exception for proper error handling
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
