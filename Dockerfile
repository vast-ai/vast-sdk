# Base image with Python 3.10 and Debian/Ubuntu userland
FROM python:3.10-slim

# Install dependencies needed by the script: git, curl, openssl, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        openssl \
        bash \
    && rm -rf /var/lib/apt/lists/*

# Default envs (can be overridden at `docker run`)
ENV WORKSPACE_DIR=/workspace \
    REPORT_ADDR=https://run.vast.ai \
    USE_SSL=true \
    WORKER_PORT=3000 \
    PYTHONUNBUFFERED=1

# Create workspace directory
RUN mkdir -p /workspace

# Copy the startup script into the container
COPY start_server.sh /usr/local/bin/start_server.sh

# Make it executable
RUN chmod +x /usr/local/bin/start_server.sh

# Work from workspace (script also cd's there, but this is nice)
WORKDIR /workspace

# The container will run the pyworker startup script
ENTRYPOINT ["/bin/bash", "/usr/local/bin/start_server.sh"]
