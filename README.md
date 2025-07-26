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

## **Architecture**

The architecture separates the plugin logic (Python) from the real-time audio engine (C++).

1. **Plugin Definition (Python)**: You define a plugin by creating a class that inherits from jxap.Plugin and implements init and update methods. The DSP logic is written using JAX for array manipulation and Equinox for state management.  
2. **Exporting (Python to MLIR)**: A Python script (export.py) uses jax.export to trace the JAX functions and convert them into MLIR in the StableHLO dialect. This, along with metadata, is packaged into a .jxap file (a zip archive).  
3. **Loading and Compilation (C++)**: The C++ runtime loads the .jxap file and uses an MLIR pipeline to refine the dynamic shapes into static ones suitable for real-time processing.  
4. **Execution (C++ and PJRT)**: The refined MLIR is compiled into native machine code using the XLA PJRT (Plugin-based Runtime) C API, creating a high-performance, JIT-compiled version of the plugin.  
5. **Audio I/O (C++)**: A C++ host (jxap\_pipewire\_run) connects to the system's audio server (e.g., PipeWire) to handle real-time audio I/O, feeding buffers to the compiled PJRT executable.

## **Example: A Simple Phaser Plugin**

This example demonstrates the end-to-end workflow with a simple phase-shifting plugin, which is a fundamental building block for many audio effects.

### **1\. The Plugin Code**

This plugin uses a first-order all-pass filter to shift the phase of the incoming audio signal. Unlike a simple gain effect, this filter needs to remember the last input and output sample, making it a "stateful" plugin. The file also includes a main function that exports the packaged plugin.

`plugins/phaser\_plugin.py`:

```python
from absl import app
from absl import flags
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import jxap


class PhaserState(jxap.Module):
    """State for the phaser, holding the last input and output samples."""
    last_input: Float[Array, ""]
    last_output: Float[Array, ""]


class PhaserPlugin(jxap.Plugin[PhaserState]):
    """A simple all-pass filter to create a phase shift effect."""

    # The center frequency of the filter in Hz. This controls the filter's response.
    center_freq_hz: float

    # Names of the input and output buffers. These will correspond to Pipewire
    # port names. All ports are single channel but there can be any number of ports.
    input_name: str = "input"
    output_name: str = "output"

    @property
    def input_buffer_names(self):
        return [self.input_name]

    def init(self, inputs: dict[str, jxap.Buffer],
             sample_rate: Float[Array, ""]) -> PhaserState:
        """Initializes the filter's state with silence."""
        del inputs, sample_rate  # Unused.
        return PhaserState(last_input=jnp.array(0.0),
                           last_output=jnp.array(0.0))

    def update(
        self, state: PhaserState, inputs: dict[str, jxap.Buffer],
        sample_rate: Float[Array, ""]
    ) -> tuple[PhaserState, dict[str, jxap.Buffer]]:
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
        input_buffer: Float[Array, "BufferSize"] = inputs[self.input_name]

        # Use jax.lax.scan for efficient vectorized processing of the filter.
        initial_state = (state.last_input, state.last_output)
        (final_input, final_output), output_buffer = jax.lax.scan(
            allpass_step,
            initial_state,
            input_buffer,
        )

        # The final samples become the state for the next buffer.
        new_state = PhaserState(last_input=final_input,
                                last_output=final_output)

        return new_state, {self.output_name: output_buffer}


# --- Exporting Logic ---
_OUTPUT_PATH = flags.DEFINE_string("output_path", "plugins/phaser_plugin.jxap",
                                   "Where to write the plugin.")


def main(_):
    # Here we create a plugin instance with specific parameters and export it.
    plugin = PhaserPlugin(center_frequency_hz=440.0)
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