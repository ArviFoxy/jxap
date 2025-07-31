# **JaX Audio Plugins (JXAP)**

JaX Audio Plugins (JXAP) is a (work-in-progress\!\!) runtime for real-time audio plugins written in Python. It uses **JAX**, a library for high-performance numerical computing, to create plugins that can be run as PipeWire nodes.

The goal is to combine the ease of Python while delivering performance comparable to hand-written C/C++ code. JXAP plugins can use the JAX ecosystem for numerical computation (including numpy, scipy, neural networks, optimization, automatic differentiation) and the XLA JIT compiler to generate optimized, vectorized machine-code.

### **Features**

* **High-Level Development**: Write high-level code in Python using the JAX ecosystem.  
* **Real-Time Ready**: Integrates with PipeWire for real-time digital signal processing (DSP).  
* **High-Performance**: Whole-plugin optimization and vectorization using the XLA compiler.  
* **Advanced Optimization**: The JIT compiler leverages the host’s CPU extensions and static audio graph parameters (e.g., buffer size, sample rate) to perform optimizations that are difficult in a typical C++ implementation.  
* **Cross-Platform**: The exported plugin bytecode is platform-independent and allows for compilation on both CPUs and GPUs.

### **Limitations**

* **Vectorized Programming Model**: JAX achieves its performance by operating on entire arrays at once. This requires writing code in a "vectorized" or "array-oriented" style, which can be a shift for those accustomed to writing explicit loops.

### Motivation 

This project grew out of my DIY audio system, which is curretly a 3.1 wireless setup (using roc-streaming) with a central linux DSP.

The in-room correction is currently done with a parametric eq (aka array of peaking filters). The filter are learned using with gradient descent (jax+optax) based on mic measurements. This apprpach is very flexible and easy to experiment with, but quite inefficient in that it's using 100 band eq per speaker (to make optimization easier).

With 2 sources (TV or music), each with its own eq profile and 4 speakers (so far) we get 800 peaking filters. The current implementation uses pipewire's filter-graph and quite a lot of pipewire nodes (15+) for all the plumbing. This ends up choking on latency, but even worse it becomes very unstable in incomprehensible way once you add all the pipewire nodes (possibly bugs in PW..).

The goal is to reduce all of these PW nodes (filters and plumbing) into a single, self-contained plugin, vectorize the filters, and reduce the complexity of the PW graph. Implementing channel routing in modular and testable Python should be much simpler than a zoo of PW configs.

Moreover with full support for jax it is possible to reuse the same jax code both for optimization and real-time processing. With support for flax nnx neural networks should technically just work. It would be even possible to attempt online learning - by feeding a microphone stream to the plugin and using gradient descent during live processing.

### Current state

Plan:
- [x] Plugin Python API (stateful DSL using flax nnx ✅)
- [x] Plugin export (to StableHLO)
- [x] Basic MLIR passes (type refinement, constant folding)
- [x] PJRT JIT runner
- [ ] Pipewire runner (currently buggy, why is PW like this 😩.. writing llvm compiler passes was easier)
- [ ] End-to-end safety tests (sample corruption, safe LUFS levels, generating test wavs). As a reusable library to be able to check plugins before running them on real speakers.
- [ ] Benchmarks (at pluging runner level)
- [ ] Compilation cache / prewarming
- [ ] Library of basic filter components
- [ ] Orbax support for plugin weights

Possible some day:
- [ ] Synchronous CPU runtime with IREE
- [ ] Domain-specific optimization: for example using the DSP MLIR dialect
- [ ] Benchmarking and optimizing the GPU runtime (if it can provide low enough latency)
- [ ] Packaging to other formats: LADSPA, DSSI, LV2, VST

## **Architecture**

The architecture separates the plugin logic (Python) from the real-time audio engine (C++).

1. **Plugin Definition (Python)**: You define a plugin by creating a class that inherits from jxap.Plugin and implements init and process methods. The DSP logic is written using JAX for array manipulation and flax NNX for state management.  
2. **Exporting (Python to MLIR)**: The Python code uses `jax.export` to trace the JAX functions and convert them into MLIR (StableHLO dialect). This, along with metadata, is packaged into a .jxap file (a zip archive).  
3. **Loading and Compilation (C++)**: The C++ runtime loads the .jxap file and once graph parameters (sample rate, buffer size, etc.) are known runs an MLIR pipeline to do some optimization (refining dynamic shapes into static, constant folding).  
4. **Execution (C++ and PJRT)**: The refined MLIR is compiled into native machine code using the XLA PJRT runtime, creating a high-performance, JIT-compiled version of the plugin.  
5. **Audio I/O (C++)**: A C++ host (jxap\_pipewire\_run) connects to the system's audio server (e.g., PipeWire) to handle real-time audio I/O, feeding buffers to the compiled PJRT executable.

## **Example: A Simple Phaser Plugin**

This example demonstrates the end-to-end workflow with a simple phase-shifting plugin, which is a fundamental building block for many audio effects.

### **1\. The Plugin Code**

This plugin uses a first-order all-pass filter to shift the phase of the incoming audio signal. This filter needs to remember the last input and output sample - stateful variables are marked by `jxap.State` which is a kind of `nnx.Varaible`. The file also includes a main function that exports the packaged plugin. Under the hood the stateful updates are transformed into a pure functional form before being exported.

`plugins/phaser\ plugin.py`:

```python

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import jxap


class PhaserPlugin(jxap.Plugin):
    """A simple all-pass filter to create a phase shift effect."""

    # The center frequency of the filter in Hz. This controls the filter's response.
    center_freq_hz: float = 440.0

    # Names of the input and output buffers. These will correspond to Pipewire
    # port names. All ports are single channel but there can be any number of ports.
    input = jxap.InputPort("input")
    output = jxap.OutputPort("output")

    # State of the audio filter. Variables that change are wrapped with "jxap.State".
    last_input: jxap.State[jax.Array]
    last_output: jxap.State[jax.Array]

    def init(self, inputs, sample_rate):
        """Initializes the filter's state with silence."""
        del inputs, sample_rate  # Unused.
        self.last_input = jxap.State(0.0)
        self.last_output = jxap.State(0.0)

    def process(
        self,
        inputs: dict[jxap.InputPort, jxap.Buffer],
        sample_rate: Float[Array, ""],
    ) -> dict[jxap.OutputPort, jxap.Buffer]:
        """Processes one buffer of audio. All samples are float32."""
        # Calculate the filter coefficient 'alpha' from the desired center frequency.
        # This makes the filter's effect consistent across different sample rates.
        # All of this computation will be inlined by the JIT compiler.
        tan_theta = jnp.tan(jnp.pi * self.center_freq_hz / sample_rate)
        alpha = (1.0 - tan_theta) / (1.0 + tan_theta)

        def allpass_step(carry, x_n):
            """Processes a single sample through the all-pass filter."""
            x_prev, y_prev = carry
            y_n = alpha * x_n + x_prev - alpha * y_prev
            return (x_n, y_n), y_n

        # `jxap.Buffer` is just an alias for a Jax array with one dimension.
        input_buffer: Float[Array, "BufferSize"] = inputs[self.input]

        # Use jax.lax.scan for efficient vectorized processing of the filter.
        initial_state = (self.last_input.value, self.last_output.value)
        (final_input, final_output), output_buffer = jax.lax.scan(
            allpass_step,
            initial_state,
            input_buffer,
        )
        # Update the plugin's state.
        self.last_input.value = final_input
        self.last_output.value = final_output
        return {self.output: output_buffer}


# --- Exporting Logic ---
_OUTPUT_PATH = flags.DEFINE_string("output_path", "plugins/phaser_plugin.jxap",
                                   "Where to write the plugin.")


def main(_):
    plugin = PhaserPlugin()
    jxap.export.export_plugin(plugin).save(_OUTPUT_PATH.value)
    print(f"Saved plugin to {_OUTPUT_PATH.value}")


if __name__ == "__main__":
    app.run(main)
```

### **2\. Export the Plugin**

Run the plugin file directly from the workspace root to create the .jxap file.

\# Ensure you are in the VS Code terminal within the dev container  
python3 python/plugins/phaser\_plugin.py \--output\_path=./phaser\_plugin.jxap

This command packages the PhaserPlugin's MLIR representation into phaser\_plugin.jxap. This is a portable self-contained plugin package.

### **3\. Build and Install the C++ Runner**

Use the pre-configured CMake tasks in VS Code or run the commands manually. This step only needs to be done once, or whenever you change the C++ code.

\# Configure CMake (only needs to be done once)  
cmake \-S . \-B build \-DCMAKE\_BUILD\_TYPE=Release \-DCMAKE\_INSTALL\_PREFIX=./install

\# Build and install the executable  
cmake \--install build

This compiles the jxap\_pipewire\_run executable and installs it into the install/bin directory.

### **4\. Run the Plugin with PipeWire**

Launch the C++ host and point it to your exported plugin file.

\# The node\_name will be visible in your PipeWire graph  
./install/bin/jxap\_pipewire\_run \\  
    \--plugin\_path=./phaser\_plugin.jxap \\  
    \--node\_name="JXAP Phaser Plugin"

You can now use a patchbay like qpwgraph to route audio to the "JXAP Phaser Plugin" input and connect its output to your speakers.

## **Repository Structure**

The repository contains the C++ core, the Python library, and configuration files.

| Path | Description |
| :---- | :---- |
| .devcontainer/ | Dockerfile and configuration for a complete VS Code Dev Containers environment. |
| jxap/ | The core C++ library, including the PJRT plugin runner, MLIR pipeline, and PipeWire host. |
| python/jxap/ | The Python library for defining plugins and the exporting mechanism. |
| python/plugins/ | Example plugins. |
| CMakeLists.txt | The main CMake file for building all C++ targets. It fetches and builds dependencies. |
| XLA.cmake | CMake script for fetching the OpenXLA repository and building the PJRT CPU plugin with Bazel. |
| StableHLO.cmake | CMake script to fetch and build the StableHLO and LLVM/MLIR projects. |

## **Development Setup**

The recommended way to develop for JXAP is using VS Code Dev Containers. This provides a consistent, pre-configured environment with all necessary dependencies and tools, avoiding manual setup on your host machine.

### **Prerequisites**

* **Visual Studio Code**  
* **Docker Desktop** (or another container runtime compatible with the Dev Containers extension)  
* The [**Dev Containers extension**](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code.

### **Getting Started**

1. **Clone the Repository:**  
   git clone https://github.com/arvifoxy/jxap.git  
   cd jxap

2. **Open in Dev Container:**  
   * Open the cloned repository folder in VS Code.  
   * VS Code will detect the .devcontainer configuration and show a notification in the bottom-right corner.  
   * Click **"Reopen in Container"**.  
3. **Container Build:**  
   * VS Code will now build the Docker image defined in .devcontainer/Dockerfile. This can take several minutes on the first run as it downloads and installs all dependencies, including the C++ toolchain, Python, and the XLA/MLIR toolchain.  
   * Once the build is complete, you will have a terminal inside the running container, with the repository mounted and ready. The Python environment is already set up with all packages from requirements.txt
