
#include "jxap/pjrt_plugin_runner.h"

#include <cstring>  // For strlen
#include <fstream>
#include <iostream>
#include <stdexcept>  // For std::runtime_error
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "jxap/stablehlo_passes.h"
#include "jxap/utils.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_cpu.h"
#include "xla/pjrt/proto/compile_options.pb.h"

namespace jxap {
namespace {

absl::Status ErrorToStatus(PJRT_Error* error, const PJRT_Api* api) {
  PJRT_Error_Message_Args msg_args;
  msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  msg_args.extension_start = nullptr;
  msg_args.error = error;
  api->PJRT_Error_Message(&msg_args);

  PJRT_Error_GetCode_Args code_args;
  code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  code_args.extension_start = nullptr;
  code_args.error = error;
  api->PJRT_Error_GetCode(&code_args);

  std::string error_message(msg_args.message, msg_args.message_size);
  return absl::Status(static_cast<absl::StatusCode>(code_args.code), msg_args.message);
}

void DestroyError(PJRT_Error* error, const PJRT_Api* api) {
  if (error == nullptr) return;

  PJRT_Error_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.error = error;
  api->PJRT_Error_Destroy(&destroy_args);
}

void LogAndDestroyError(PJRT_Error* error, const PJRT_Api* api) {
  if (error == nullptr) return;
  LOG(ERROR) << "PJRT error: " << ErrorToStatus(error, api);
  DestroyError(error, api);
}

#define RETURN_IF_PJRT_ERROR(error_expr, api)           \
  {                                                     \
    PJRT_Error* error = error_expr;                     \
    if (error != nullptr) {                             \
      absl::Status status_ = ErrorToStatus(error, api); \
      DestroyError(error, api);                         \
      return status_;                                   \
    }                                                   \
  }

#define RETURN_IF_ERROR(status_expr)    \
  {                                     \
    absl::Status status_ = status_expr; \
    if (!status_.ok()) {                \
      return status_;                   \
    }                                   \
  }

}  // namespace

class PJRTContext {
 public:
  // Owned and destroyed by this object.
  const PJRT_Api* api = nullptr;
  PJRT_Client* client = nullptr;
  PJRT_Device* device = nullptr;

  // Non-copyable.
  PJRTContext(const PJRTContext&) = delete;
  PJRTContext& operator=(const PJRTContext&) = delete;

 protected:
  PJRTContext() = default;

 public:
  static absl::StatusOr<std::unique_ptr<PJRTContext>> Create() {
    std::unique_ptr<PJRTContext> ctx(new PJRTContext());
    ctx->api = GetPjrtApi();
    const PJRT_Api* api = ctx->api;
    if (api == nullptr) {
      return absl::FailedPreconditionError(
          "Failed to get PJRT CPU API. Ensure "
          "GetPjrtApi() is implemented correctly.");
    }

    // Verify API version (optional but good practice)
    if (api->pjrt_api_version.major_version != PJRT_API_MAJOR ||
        api->pjrt_api_version.minor_version < PJRT_API_MINOR) {  // Allow newer minor versions
      LOG(ERROR) << "Warning: PJRT API version mismatch. Expected " << PJRT_API_MAJOR << "."
                 << PJRT_API_MINOR << ", Got " << api->pjrt_api_version.major_version << "."
                 << api->pjrt_api_version.minor_version;
    }

    // Initialize the Plugin
    PJRT_Plugin_Initialize_Args plugin_init_args;
    plugin_init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    plugin_init_args.extension_start = nullptr;
    RETURN_IF_PJRT_ERROR(api->PJRT_Plugin_Initialize(&plugin_init_args), api);
    LOG(INFO) << "PJRT Plugin initialized.";

    // Create a PJRT Client
    PJRT_Client_Create_Args client_create_args;
    client_create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    client_create_args.extension_start = nullptr;
    client_create_args.create_options = nullptr;
    client_create_args.num_options = 0;
    client_create_args.kv_get_callback = nullptr;
    client_create_args.kv_get_user_arg = nullptr;
    client_create_args.kv_put_callback = nullptr;
    client_create_args.kv_put_user_arg = nullptr;
    client_create_args.kv_try_get_callback = nullptr;
    client_create_args.kv_try_get_user_arg = nullptr;
    client_create_args.client = nullptr;
    RETURN_IF_PJRT_ERROR(api->PJRT_Client_Create(&client_create_args), api);
    ctx->client = client_create_args.client;
    LOG(INFO) << "PJRT Client created.";

    // Get an addressable device
    PJRT_Client_AddressableDevices_Args devices_args;
    devices_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    devices_args.extension_start = nullptr;
    devices_args.client = ctx->client;
    RETURN_IF_PJRT_ERROR(api->PJRT_Client_AddressableDevices(&devices_args), api);
    if (devices_args.num_addressable_devices == 0) {
      return absl::FailedPreconditionError("No addressable devices found.");
    }
    ctx->device = devices_args.addressable_devices[0];
    LOG(INFO) << "Using first addressable device.";

    return ctx;
  }

  ~PJRTContext() {
    if (api == nullptr) return;
    if (client != nullptr) {
      PJRT_Client_Destroy_Args client_destroy_args;
      client_destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
      client_destroy_args.extension_start = nullptr;
      client_destroy_args.client = client;
      LogAndDestroyError(api->PJRT_Client_Destroy(&client_destroy_args), api);
    }
  }
};

class PJRTExecutable {
 public:
  // Not owned.
  const PJRT_Api* api = nullptr;
  // Owned and destroyed by this object.
  PJRT_LoadedExecutable* loaded_executable = nullptr;
  PJRT_Executable* executable = nullptr;

 protected:
  PJRTExecutable() = default;

 public:
  static absl::StatusOr<std::unique_ptr<PJRTExecutable>> Load(const std::string& mlir_code,
                                                              PJRTContext* ctx) {
    std::unique_ptr<PJRTExecutable> exec(new PJRTExecutable());
    exec->api = ctx->api;

    // Compile the MLIR program
    PJRT_Program program_desc;
    program_desc.struct_size = PJRT_Program_STRUCT_SIZE;
    program_desc.extension_start = nullptr;
    program_desc.code = const_cast<char*>(mlir_code.c_str());
    program_desc.code_size = mlir_code.length();
    const char* format_str = "mlir";
    program_desc.format = format_str;
    program_desc.format_size = strlen(format_str);

    PJRT_Client_Compile_Args compile_args;
    compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.extension_start = nullptr;
    compile_args.client = ctx->client;
    compile_args.program = &program_desc;
    // Compile options
    xla::CompileOptionsProto compile_options;
    xla::ExecutableBuildOptionsProto* build_options =
        compile_options.mutable_executable_build_options();
    build_options->set_num_replicas(1);
    build_options->set_num_partitions(1);
    build_options->set_optimization_level(xla::ExecutionOptions_EffortLevel_EFFORT_O2);
    build_options->set_memory_fitting_level(xla::ExecutionOptions_EffortLevel_EFFORT_O2);
    compile_options.set_compile_portable_executable(false);
    std::string compile_options_str = compile_options.SerializeAsString();
    compile_args.compile_options = compile_options_str.c_str();
    compile_args.compile_options_size = compile_options_str.length();
    // Output field:
    compile_args.executable = nullptr;

    RETURN_IF_PJRT_ERROR(ctx->api->PJRT_Client_Compile(&compile_args), ctx->api);
    exec->loaded_executable = compile_args.executable;
    std::cout << "MLIR program compiled." << std::endl;

    // Get the executable
    PJRT_LoadedExecutable_GetExecutable_Args get_exec_args;
    get_exec_args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
    get_exec_args.extension_start = nullptr;
    get_exec_args.loaded_executable = exec->loaded_executable;
    RETURN_IF_PJRT_ERROR(ctx->api->PJRT_LoadedExecutable_GetExecutable(&get_exec_args), ctx->api);
    exec->executable = get_exec_args.executable;

    return exec;
  }

  ~PJRTExecutable() {
    if (executable != nullptr) {
      PJRT_Executable_Destroy_Args destroy_exec_args;
      destroy_exec_args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
      destroy_exec_args.extension_start = nullptr;
      destroy_exec_args.executable = executable;
      LogAndDestroyError(api->PJRT_Executable_Destroy(&destroy_exec_args), api);
    }
    if (loaded_executable != nullptr) {
      PJRT_LoadedExecutable_Destroy_Args destroy_exec_args;
      destroy_exec_args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
      destroy_exec_args.extension_start = nullptr;
      destroy_exec_args.executable = loaded_executable;
      LogAndDestroyError(api->PJRT_LoadedExecutable_Destroy(&destroy_exec_args), api);
    }
  }

  void PrintStats(absl::string_view path, absl::string_view method) const {
    if (executable == nullptr) return;

    PJRT_Executable_GetCompiledMemoryStats_Args stats_args;
    stats_args.struct_size = PJRT_Executable_GetCompiledMemoryStats_Args_STRUCT_SIZE;
    stats_args.extension_start = nullptr;
    stats_args.executable = executable;
    LogAndDestroyError(api->PJRT_Executable_GetCompiledMemoryStats(&stats_args), api);

    LOG(INFO) << "-------------------------------------------------------------"
                 "-------------------";
    LOG(INFO) << "| Compilation statistics for plugin \"" << path << "\" method " << method << ":";
    LOG(INFO) << "| Generated code size (MB) : "
              << stats_args.generated_code_size_in_bytes / (1024.0 * 1024.0);
    LOG(INFO) << "| Argument size       (MB) : "
              << stats_args.argument_size_in_bytes / (1024.0 * 1024.0);
    LOG(INFO) << "| Output size         (MB) : "
              << stats_args.output_size_in_bytes / (1024.0 * 1024.0);
    LOG(INFO) << "| Alias size          (MB) : "
              << stats_args.alias_size_in_bytes / (1024.0 * 1024.0);
    LOG(INFO) << "| Temp size           (MB): "
              << stats_args.temp_size_in_bytes / (1024.0 * 1024.0);
    LOG(INFO) << "| Host generated code size (MB): "
              << stats_args.host_generated_code_size_in_bytes / (1024.0 * 1024.0);
    LOG(INFO) << "| Host argument size       (MB): "
              << stats_args.host_argument_size_in_bytes / (1024.0 * 1024.0);
    LOG(INFO) << "| Host output size         (MB): "
              << stats_args.host_output_size_in_bytes / (1024.0 * 1024.0);
    LOG(INFO) << "| Host alias size          (MB): "
              << stats_args.host_alias_size_in_bytes / (1024.0 * 1024.0);
    LOG(INFO) << "| Host temp size           (MB): "
              << stats_args.host_temp_size_in_bytes / (1024.0 * 1024.0);
    LOG(INFO) << "-------------------------------------------------------------"
                 "-------------------";
  }
};

PJRTCompiledPlugin::~PJRTCompiledPlugin() {}

PJRTPluginRunner::~PJRTPluginRunner() {}

absl::StatusOr<std::unique_ptr<PJRTPluginRunner>> PJRTPluginRunner::LoadPlugin(
    absl::string_view path) {
  std::unique_ptr<PJRTPluginRunner> plugin(new PJRTPluginRunner());

  absl::StatusOr<std::unique_ptr<PJRTContext>> status_or_ctx = PJRTContext::Create();
  RETURN_IF_ERROR(status_or_ctx.status());
  plugin->ctx_ = std::move(status_or_ctx.value());

  absl::StatusOr<std::string> init_fn_mlir = ReadFile(absl::StrCat(path, "-init"));
  RETURN_IF_ERROR(init_fn_mlir.status());
  plugin->init_fn_mlir_ = std::move(init_fn_mlir.value());

  absl::StatusOr<std::string> update_fn_mlir = ReadFile(absl::StrCat(path, "-update"));
  RETURN_IF_ERROR(update_fn_mlir.status());
  plugin->update_fn_mlir_ = std::move(update_fn_mlir.value());

  return plugin;
}

absl::StatusOr<std::unique_ptr<PJRTCompiledPlugin>> PJRTPluginRunner::Compile(
    const std::set<std::string>& input_buffers, const std::set<std::string>& output_buffers,
    int buffer_size, float sample_rate) {
  std::unique_ptr<PJRTCompiledPlugin> compiled_plugin(new PJRTCompiledPlugin());
  compiled_plugin->input_buffers_ = input_buffers;
  compiled_plugin->output_buffers_ = output_buffers;
  compiled_plugin->buffer_size_ = buffer_size;
  compiled_plugin->sample_rate_ = sample_rate;

  std::vector<ArgumentTransform> transforms;
  transforms.push_back(RefineType(MlirTensorType({}, "i32")));  // platform index
  for (const std::string& _ : input_buffers) {
    transforms.push_back(RefineType(MlirTensorType({buffer_size}, "f32")));
  }
  transforms.push_back(ReplaceWithConstant(sample_rate));  // sample rate

  auto status_or_init_mlir = MlirTransformArguments(init_fn_mlir_, transforms);
  RETURN_IF_ERROR(status_or_init_mlir.status());
  std::string init_mlir = status_or_init_mlir.value();

  auto status_or_init_excutable = PJRTExecutable::Load(init_mlir, ctx_.get());
  RETURN_IF_ERROR(status_or_init_excutable.status());
  compiled_plugin->init_fn_ = std::move(status_or_init_excutable.value());

  return compiled_plugin;
}

}  // namespace jxap

/*
// Helper to await and then destroy an event
void AwaitAndDestroyEvent(PJRT_Event* event, const PJRT_Api* api, const
std::string& event_name) { if (!event) return;

    PJRT_Event_Await_Args await_args;
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.extension_start = nullptr;
    await_args.event = event;
    // PJRT_Event_Await returns the error status *of the operation that the
event is tracking*. PJRT_Error* operation_error =
api->PJRT_Event_Await(&await_args); CheckError(operation_error, api, "Operation
error from PJRT_Event_Await for " + event_name);

    // Now destroy the event itself
    PJRT_Event_Destroy_Args destroy_event_args;
    destroy_event_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    destroy_event_args.extension_start = nullptr;
    destroy_event_args.event = event;
    PJRT_Error* destroy_err = api->PJRT_Event_Destroy(&destroy_event_args);
    CheckError(destroy_err, api, "PJRT_Event_Destroy for " + event_name);
}


int main() {

    PJRT_Client* client = nullptr;
    PJRT_LoadedExecutable* loaded_executable = nullptr;
    PJRT_Executable* underlying_executable = nullptr; // For querying properties
    PJRT_Buffer* input_buffer = nullptr;
    PJRT_Buffer* output_buffer_from_execute = nullptr;
    PJRT_Device* device = nullptr;

    try {
        // 4.2 Get number of outputs to prepare output buffer structure
        PJRT_Executable_NumOutputs_Args num_outputs_args;
        num_outputs_args.struct_size =
PJRT_Executable_NumOutputs_Args_STRUCT_SIZE; num_outputs_args.extension_start =
nullptr; num_outputs_args.executable = underlying_executable;
        CheckError(api->PJRT_Executable_NumOutputs(&num_outputs_args), api,
"PJRT_Executable_NumOutputs"); size_t num_outputs_per_device =
num_outputs_args.num_outputs;

        if (num_outputs_per_device != 1) { // This example specifically expects
one output throw std::runtime_error("Example requires 1 output, but MLIR model
has " + std::to_string(num_outputs_per_device));
        }
        std::cout << "Executable has " << num_outputs_per_device << " output(s)
per device." << std::endl;

        // 5. Prepare input data
        std::vector<float> host_input_data = {10.0f, 20.0f, 30.0f, 40.0f};
        const int64_t input_dims[] = {static_cast<long
long>(host_input_data.size())}; size_t num_input_dims = 1;

        // 6. Create input buffer on device from host data
        PJRT_Client_BufferFromHostBuffer_Args bhfh_args;
        bhfh_args.struct_size =
PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE; bhfh_args.extension_start =
nullptr; bhfh_args.client = client; bhfh_args.data = host_input_data.data();
        bhfh_args.type = PJRT_Buffer_Type_F32;
        bhfh_args.dims = input_dims;
        bhfh_args.num_dims = num_input_dims;
        bhfh_args.byte_strides = nullptr; // Null for dense row-major
        bhfh_args.num_byte_strides = 0;
        bhfh_args.host_buffer_semantics =
PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes; bhfh_args.device =
device; bhfh_args.memory = nullptr; // Use default memory for the device
        bhfh_args.device_layout = nullptr; // Use default layout
        // Output fields:
        bhfh_args.done_with_host_buffer = nullptr;
        bhfh_args.buffer = nullptr;

        CheckError(api->PJRT_Client_BufferFromHostBuffer(&bhfh_args), api,
"PJRT_Client_BufferFromHostBuffer (input)"); input_buffer = bhfh_args.buffer;
        PJRT_Event* input_transfer_event = bhfh_args.done_with_host_buffer;

        AwaitAndDestroyEvent(input_transfer_event, api, "input H2D transfer");
        std::cout << "Input buffer created on device and H2D transfer complete."
<< std::endl;

        // 7. Prepare for Execution
        PJRT_Buffer* const* single_arg_list = &input_buffer; // Array of input
buffers for one execution PJRT_Buffer* const* const* all_devices_arg_lists =
&single_arg_list; // List of arg lists (for one device)

        // Prepare space for output buffers: PJRT_LoadedExecutable_Execute will
populate this.
        // For 1 device, 1 output:
        std::vector<PJRT_Buffer*>
device_output_buffers_list(num_outputs_per_device); // e.g., {nullptr}
        PJRT_Buffer** device_output_buffers_array_ptr =
device_output_buffers_list.data(); PJRT_Buffer** const*
output_lists_for_execute_arg = &device_output_buffers_array_ptr;

        PJRT_ExecuteOptions execute_options;
        execute_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
        execute_options.extension_start = nullptr;
        execute_options.launch_id = 0; // Default
        execute_options.non_donatable_input_indices = nullptr;
        execute_options.num_non_donatable_input_indices = 0;
        execute_options.context = nullptr;
        execute_options.send_callbacks = nullptr;
        execute_options.recv_callbacks = nullptr;
        execute_options.num_send_ops = 0;
        execute_options.num_recv_ops = 0;

        PJRT_LoadedExecutable_Execute_Args execute_args;
        execute_args.struct_size =
PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE; execute_args.extension_start =
nullptr; execute_args.executable = loaded_executable; execute_args.options =
&execute_options; execute_args.argument_lists = all_devices_arg_lists;
        execute_args.num_devices = 1; // Executing on a single device
        execute_args.num_args = 1;    // The MLIR function takes one argument
        execute_args.output_lists = output_lists_for_execute_arg; // API
populates this execute_args.execute_device = device; // Explicitly specify
single device for execution
        // Output field for completion event:
        PJRT_Event* execution_complete_event = nullptr;
        execute_args.device_complete_events = &execution_complete_event; // For
single device execution

        // 8. Execute the model
        CheckError(api->PJRT_LoadedExecutable_Execute(&execute_args), api,
"PJRT_LoadedExecutable_Execute"); std::cout << "Execution launched." <<
std::endl;

        AwaitAndDestroyEvent(execution_complete_event, api, "model execution");
        std::cout << "Execution completed." << std::endl;

        output_buffer_from_execute = device_output_buffers_list[0]; // Retrieve
the populated output buffer if (!output_buffer_from_execute) { throw
std::runtime_error("Execute did not return an output buffer.");
        }

        // 9. Get output buffer details to prepare host memory
        PJRT_Buffer_Dimensions_Args out_dim_args;
        out_dim_args.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
        out_dim_args.extension_start = nullptr;
        out_dim_args.buffer = output_buffer_from_execute;
        CheckError(api->PJRT_Buffer_Dimensions(&out_dim_args), api,
"PJRT_Buffer_Dimensions (output)");

        PJRT_Buffer_ElementType_Args out_type_args;
        out_type_args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
        out_type_args.extension_start = nullptr;
        out_type_args.buffer = output_buffer_from_execute;
        CheckError(api->PJRT_Buffer_ElementType(&out_type_args), api,
"PJRT_Buffer_ElementType (output)");

        if (out_type_args.type != PJRT_Buffer_Type_F32) {
            throw std::runtime_error("Output buffer type is not F32 as
expected.");
        }
        size_t num_output_elements = 1;
        std::cout << "Output buffer has " << out_dim_args.num_dims << "
dimension(s): "; for (size_t i = 0; i < out_dim_args.num_dims; ++i) {
            num_output_elements *= out_dim_args.dims[i];
            std::cout << out_dim_args.dims[i] << " ";
        }
        std::cout << "(Total elements: " << num_output_elements << ")" <<
std::endl;

        std::vector<float> host_output_data(num_output_elements);

        // 10. Transfer output from device to host
        PJRT_Buffer_ToHostBuffer_Args bth_args;
        bth_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
        bth_args.extension_start = nullptr;
        bth_args.src = output_buffer_from_execute;
        bth_args.dst = host_output_data.data();
        bth_args.dst_size = host_output_data.size() * sizeof(float);
        bth_args.host_layout = nullptr; // Use default/current layout
        // Output field:
        bth_args.event = nullptr;

        CheckError(api->PJRT_Buffer_ToHostBuffer(&bth_args), api,
"PJRT_Buffer_ToHostBuffer (output)"); PJRT_Event* output_transfer_event =
bth_args.event;

        AwaitAndDestroyEvent(output_transfer_event, api, "output D2H transfer");
        std::cout << "Output buffer transferred to host." << std::endl;

        // 11. Process/Print Output Data
        std::cout << "Output data from device: ";
        for (size_t i = 0; i < host_output_data.size(); ++i) {
            std::cout << host_output_data[i] << (i == host_output_data.size() -
1 ? "" : ", ");
        }
        std::cout << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        // Note: Resources might not be fully cleaned up here if error occurs
mid-setup
    }

    // 12. Cleanup
    std::cout << "Cleaning up resources..." << std::endl;
    if (input_buffer) {
        PJRT_Buffer_Destroy_Args destroy_buf_args;
        destroy_buf_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        destroy_buf_args.extension_start = nullptr;
        destroy_buf_args.buffer = input_buffer;
        CheckError(api->PJRT_Buffer_Destroy(&destroy_buf_args), api,
"PJRT_Buffer_Destroy (input)"); std::cout << "Input buffer destroyed." <<
std::endl;
    }
    if (output_buffer_from_execute) {
        PJRT_Buffer_Destroy_Args destroy_buf_args;
        destroy_buf_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        destroy_buf_args.extension_start = nullptr;
        destroy_buf_args.buffer = output_buffer_from_execute;
        CheckError(api->PJRT_Buffer_Destroy(&destroy_buf_args), api,
"PJRT_Buffer_Destroy (output)"); std::cout << "Output buffer destroyed." <<
std::endl;
    }
    if (underlying_executable) {
        PJRT_Executable_Destroy_Args destroy_exec_args;
        destroy_exec_args.struct_size =
PJRT_Executable_Destroy_Args_STRUCT_SIZE; destroy_exec_args.extension_start =
nullptr; destroy_exec_args.executable = underlying_executable;
        CheckError(api->PJRT_Executable_Destroy(&destroy_exec_args), api,
"PJRT_Executable_Destroy (underlying)"); std::cout << "Underlying
PJRT_Executable object destroyed." << std::endl;
    }
    if (loaded_executable) {
        PJRT_LoadedExecutable_Destroy_Args destroy_loaded_exec_args;
        destroy_loaded_exec_args.struct_size =
PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_loaded_exec_args.extension_start = nullptr;
        destroy_loaded_exec_args.executable = loaded_executable;
        CheckError(api->PJRT_LoadedExecutable_Destroy(&destroy_loaded_exec_args),
api, "PJRT_LoadedExecutable_Destroy"); std::cout << "Loaded executable
destroyed." << std::endl;
    }
    if (client) {
        PJRT_Client_Destroy_Args destroy_client_args;
        destroy_client_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_client_args.extension_start = nullptr;
        destroy_client_args.client = client;
        CheckError(api->PJRT_Client_Destroy(&destroy_client_args), api,
"PJRT_Client_Destroy"); std::cout << "Client destroyed." << std::endl;
    }

    std::cout << "Execution finished successfully." << std::endl;
    return 0;
}
*/
