FROM debian:bookworm-backports

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dev tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gdb \
    cmake \
    ninja-build \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    # PipeWire development libraries
    libpipewire-0.3-dev \
    libspa-0.2-dev \
    # Google Test dependencies (needed if building from source)
    libgtest-dev \
    # flatbuffers for JAX serialization
    libflatbuffers-dev flatbuffers-compiler-dev \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Compile and install Google Test from source provided by libgtest-dev
# The Debian package only provides source, we need to build it.
RUN cd /usr/src/googletest \
    && cmake . \
    && make \
    && make install \
    # Clean up build files
    && cd / \
    && rm -rf /usr/src/googletest

# --- Install Bazelisk ---
ARG BAZELISK_VERSION="v1.26.0"
ARG TARGETARCH="amd64"

RUN curl -sSL "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-${TARGETARCH}" -o /usr/local/bin/bazelisk \
    && chmod +x /usr/local/bin/bazelisk \
    # Create a symlink so 'bazel' command uses bazelisk
    && ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel

# Set up a non-root user (optional but recommended)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Switch to non-root user
USER $USERNAME

# Set working directory
WORKDIR /workspaces

COPY --chown=$USERNAME:$USERNAME python/requirements.txt .

RUN python3 -m venv /workspaces/.venv
ENV PATH="/workspaces/.venv/bin:${PATH}"
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt
