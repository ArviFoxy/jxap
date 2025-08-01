FROM debian:bookworm-backports

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dev tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    # Build tools.
    build-essential \
    curl \
    gdb \
    cmake \
    ninja-build \
    git \
    pkg-config \
    ccache \
    patchelf \
    # For the LLVM script.
    wget \
    lsb-release \
    software-properties-common \
    gnupg \
    # Python
    python3 \
    python3-pip \
    python3-venv \
    # PipeWire development libraries
    libpipewire-0.3-dev \
    libspa-0.2-dev \
    libdbus-1-dev \
    libglib2.0-dev \
    # Google Test dependencies (needed if building from source)
    libgtest-dev \
    # flatbuffers for JAX serialization
    libflatbuffers-dev flatbuffers-compiler-dev \
    # Protocol buffers
    libprotobuf-dev

# Install GCC 12 (default toolchain)
RUN apt-get install -y --no-install-recommends gcc-12

# Install LLVM
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 20 all
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 1000

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
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to non-root user
USER $USERNAME

# Set working directory
WORKDIR /workspaces

COPY --chown=$USERNAME:$USERNAME python/requirements.txt .

RUN python3 -m venv /workspaces/.venv
ENV PATH="/workspaces/.venv/bin:${PATH}"
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt
