FROM tensorzero/gateway:2026.6.0

COPY tensorzero/tensorzero.toml /app/config/tensorzero.toml
COPY tensorzero/templates/ /app/config/templates/

CMD ["--config-file", "/app/config/tensorzero.toml"]
