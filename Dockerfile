# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENCLAW_CONFIG_DIR=/config \
    TOKENCLAW_DB=/data/tokenclaw.sqlite3 \
    TOKENCLAW_HOST=0.0.0.0 \
    TOKENCLAW_LOCAL_RULES_ONLY=1 \
    TOKENCLAW_MANAGED=0 \
    TOKENCLAW_MANAGED_ROUTING=0 \
    TOKENCLAW_MANAGED_CRUNCH=0 \
    TOKENCLAW_MANAGED_CACHE=0 \
    TOKENCLAW_RECOMMENDATIONS_ENABLED=0 \
    TOKENCLAW_RECOMMENDATION_ENABLED=0 \
    TOKENCLAW_POLICY_DECISIONS_ENABLED=0 \
    TOKENCLAW_POLICY_DECISION_ENABLED=0 \
    TOKENCLAW_MANAGED_FEEDBACK_DRAIN_INTERVAL_SECONDS=0 \
    TOKENCLAW_MANAGED_FEEDBACK_ACTIVATION_DRAIN_BATCH_LIMIT=0

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir ".[server]" \
    && chmod +x /app/docker-entrypoint.sh \
    && addgroup --system tokenclaw \
    && adduser --system --ingroup tokenclaw --home /home/tokenclaw tokenclaw \
    && mkdir -p /data /config /home/tokenclaw \
    && chown -R tokenclaw:tokenclaw /data /config /home/tokenclaw

USER tokenclaw

VOLUME ["/data", "/config"]
EXPOSE 4000 4002 4003

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD for url in http://127.0.0.1:4002/tokenclaw/stats http://127.0.0.1:4003/health http://127.0.0.1:4000/health; do python -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=3).read()" "$url" && exit 0; done; exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["tokenclaw", "start"]
