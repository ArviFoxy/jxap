
#include "jxap/pjrt_plugin_runner.h"

#include <cstring>  // For strlen
#include <fstream>
#include <iostream>
#include <source_location>
#include <stdexcept>  // For std::runtime_error
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "jxap/mlir_pipeline.h"
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

#define RETURN_IF_PJRT_ERROR(error_expr, api)               \
  {                                                         \
    PJRT_Error* error = error_expr;                         \
    if (error != nullptr) {                                 \
      absl::Status status_ = ErrorToStatus(error, api);     \
      DestroyError(error, api);                             \
      constexpr auto loc = std::source_location::current(); \
      return AppendLocToStatus(status_, loc);               \
    }                                                       \
  }

absl::Status PJRTAwaitAndDestroyEvent(PJRT_Event* event, const PJRT_Api* api) {
  if (event == nullptr) return absl::OkStatus();

  PJRT_Event_Await_Args await_args;
  await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  await_args.extension_start = nullptr;
  await_args.event = event;
  RETURN_IF_PJRT_ERROR(api->PJRT_Event_Await(&await_args), api);

  PJRT_Event_Destroy_Args destroy_event_args;
  destroy_event_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  destroy_event_args.extension_start = nullptr;
  destroy_event_args.event = event;
  RETURN_IF_PJRT_ERROR(api->PJRT_Event_Destroy(&destroy_event_args), api);

  return absl::OkStatus();
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

class PJRTBuffer {
 protected:
  PJRT_Buffer* buffer_;
  const PJRT_Api* api_;
  // Optional: host-owned buffer data. Only used for zero-copy buffer semantics.
  Buffer buffer_data_;
  // Signals that the host buffer is free to use again.
  PJRT_Event* done_with_host_buffer_ = nullptr;

  // Zero-copy variant. The buffer data will be owned by this object.
  PJRTBuffer(PJRT_Buffer* buffer, const PJRT_Api* api, Buffer&& buffer_data,
             PJRT_Event* done_with_host_buffer)
      : buffer_(buffer),
        api_(api),
        buffer_data_(std::move(buffer_data)),
        done_with_host_buffer_(done_with_host_buffer) {}

 public:
  // Copying variant, PJRT_Buffer owns the buffer data.
  PJRTBuffer(PJRT_Buffer* buffer, const PJRT_Api* api) : buffer_(buffer), api_(api) {}

  ~PJRTBuffer() {
    if (buffer_ != nullptr) {
      PJRT_Buffer_Destroy_Args destroy_args;
      destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
      destroy_args.extension_start = nullptr;
      destroy_args.buffer = buffer_;
      LogAndDestroyError(api_->PJRT_Buffer_Destroy(&destroy_args), api_);
    }
    auto status = AwaitDoneWithHostBuffer();
    if (!status.ok()) {
      LOG(ERROR) << "AwaitDoneWithHostBuffer failed: " << status.message();
    }
  }

  absl::Status AwaitDoneWithHostBuffer() {
    if (done_with_host_buffer_ == nullptr) return absl::OkStatus();
    auto status = PJRTAwaitAndDestroyEvent(done_with_host_buffer_, api_);
    done_with_host_buffer_ = nullptr;
    return status;
  }

  // Movable but not copyable.
  PJRTBuffer(PJRTBuffer&& other) noexcept : buffer_(other.buffer_), api_(other.api_) {
    other.buffer_ = nullptr;
  }
  PJRTBuffer& operator=(PJRTBuffer&& other) noexcept {
    if (this != &other) {
      std::swap(buffer_, other.buffer_);
      std::swap(api_, other.api_);
    }
    return *this;
  }
  PJRTBuffer(const PJRTBuffer&) = delete;
  PJRTBuffer& operator=(const PJRTBuffer&) = delete;

  // Returns the underlying buffer.
  PJRT_Buffer* GetBuffer() { return buffer_; }

  static absl::StatusOr<size_t> TypeToElementSize(PJRT_Buffer_Type type) {
    switch (type) {
      case PJRT_Buffer_Type_PRED:
      case PJRT_Buffer_Type_S8:
      case PJRT_Buffer_Type_U8:
      // Truncated 8 bit floating-point formats.
      case PJRT_Buffer_Type_F8E4M3:
      case PJRT_Buffer_Type_F8E3M4:
      case PJRT_Buffer_Type_F8E8M0FNU:
      case PJRT_Buffer_Type_F8E5M2:
      case PJRT_Buffer_Type_F8E4M3FN:
      case PJRT_Buffer_Type_F8E4M3B11FNUZ:
      case PJRT_Buffer_Type_F8E5M2FNUZ:
      case PJRT_Buffer_Type_F8E4M3FNUZ:
        return 1;

      case PJRT_Buffer_Type_S16:
      case PJRT_Buffer_Type_U16:
      case PJRT_Buffer_Type_F16:
      case PJRT_Buffer_Type_BF16:
        return 2;

      case PJRT_Buffer_Type_S32:
      case PJRT_Buffer_Type_U32:
      case PJRT_Buffer_Type_F32:
        return 4;

      case PJRT_Buffer_Type_S64:
      case PJRT_Buffer_Type_U64:
      case PJRT_Buffer_Type_F64:
      case PJRT_Buffer_Type_C64:
        return 8;

      case PJRT_Buffer_Type_C128:
        return 16;

        // Formats with less than 8 bit or unknown size.
      case PJRT_Buffer_Type_S4:
      case PJRT_Buffer_Type_U4:
      case PJRT_Buffer_Type_S2:
      case PJRT_Buffer_Type_U2:
      case PJRT_Buffer_Type_F4E2M1FN:
      case PJRT_Buffer_Type_TOKEN:
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("TypeToElementSize: unsupported buffer type: ", type));
    }
  }

  static absl::StatusOr<std::string> TypeToMLIR(PJRT_Buffer_Type type) {
    switch (type) {
      case PJRT_Buffer_Type_S8:
        return "s8";

      case PJRT_Buffer_Type_U8:
        return "u8";

      case PJRT_Buffer_Type_S16:
        return "s16";

      case PJRT_Buffer_Type_U16:
        return "u16";

      case PJRT_Buffer_Type_F16:
        return "f16";

      case PJRT_Buffer_Type_S32:
        return "s32";

      case PJRT_Buffer_Type_U32:
        return "u32";

      case PJRT_Buffer_Type_F32:
        return "f32";

      case PJRT_Buffer_Type_S64:
        return "s64";

      case PJRT_Buffer_Type_U64:
        return "u64";

      case PJRT_Buffer_Type_F64:
        return "f64";

      case PJRT_Buffer_Type_C64:
        return "complex<f32>";

      case PJRT_Buffer_Type_C128:
        return "complex<f64>";

      default:
        return absl::InvalidArgumentError(
            absl::StrCat("TypeToMLIR: unsupported buffer type: ", type));
    }
  }

  static absl::Status VerifyBufferSize(const Buffer& buffer, PJRT_Buffer_Type type,
                                       const std::vector<int64_t>& dims) {
    auto status_or_element_size = TypeToElementSize(type);
    RETURN_IF_ERROR(status_or_element_size.status());
    size_t element_size = status_or_element_size.value();
    size_t expected_size = element_size;
    for (int64_t dim : dims) {
      expected_size *= dim;
    }
    if (buffer.size() != expected_size) {
      return absl::InvalidArgumentError(absl::StrCat("Buffer size mismatch: expected ",
                                                     expected_size, ", got ", buffer.size(), "."));
    }
    return absl::OkStatus();
  }

  static absl::StatusOr<PJRTBuffer> FromHostCopy(const Buffer& buffer, PJRT_Buffer_Type type,
                                                 const std::vector<int64_t>& dims,
                                                 PJRTContext* ctx) {
    RETURN_IF_ERROR(VerifyBufferSize(buffer, type, dims));

    PJRT_Client_BufferFromHostBuffer_Args args;
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.client = ctx->client;
    args.data = buffer.data();
    args.type = type;
    args.dims = dims.data();
    args.num_dims = dims.size();
    args.byte_strides = nullptr;  // Null for dense row-major
    args.num_byte_strides = 0;
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    args.device = ctx->device;
    args.memory = nullptr;         // Use default memory for the device
    args.device_layout = nullptr;  // Use default layout
    // Output fields:
    args.done_with_host_buffer = nullptr;
    args.buffer = nullptr;

    RETURN_IF_PJRT_ERROR(ctx->api->PJRT_Client_BufferFromHostBuffer(&args), ctx->api);
    return PJRTBuffer(args.buffer, ctx->api);
  }

  static absl::StatusOr<PJRTBuffer> FromHostZeroCopy(Buffer&& buffer, PJRT_Buffer_Type type,
                                                     const std::vector<int64_t>& dims,
                                                     PJRTContext* ctx) {
    RETURN_IF_ERROR(VerifyBufferSize(buffer, type, dims));

    PJRT_Client_BufferFromHostBuffer_Args args;
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.client = ctx->client;
    args.data = buffer.data();
    args.type = type;
    args.dims = dims.data();
    args.num_dims = dims.size();
    args.byte_strides = nullptr;  // Null for dense row-major
    args.num_byte_strides = 0;
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableZeroCopy;
    args.device = ctx->device;
    args.memory = nullptr;         // Use default memory for the device
    args.device_layout = nullptr;  // Use default layout
    // Output fields:
    args.done_with_host_buffer = nullptr;
    args.buffer = nullptr;

    RETURN_IF_PJRT_ERROR(ctx->api->PJRT_Client_BufferFromHostBuffer(&args), ctx->api);
    return PJRTBuffer(args.buffer, ctx->api, std::move(buffer), args.done_with_host_buffer);
  }

  /**
   * Calculates buffer size in bytes.
   */
  absl::StatusOr<size_t> Size() {
    PJRT_Buffer_Dimensions_Args out_dim_args;
    out_dim_args.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
    out_dim_args.extension_start = nullptr;
    out_dim_args.buffer = buffer_;
    RETURN_IF_PJRT_ERROR(api_->PJRT_Buffer_Dimensions(&out_dim_args), api_);

    PJRT_Buffer_ElementType_Args out_type_args;
    out_type_args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
    out_type_args.extension_start = nullptr;
    out_type_args.buffer = buffer_;
    RETURN_IF_PJRT_ERROR(api_->PJRT_Buffer_ElementType(&out_type_args), api_)

    auto element_size = TypeToElementSize(out_type_args.type);
    RETURN_IF_ERROR(element_size.status());

    size_t size = element_size.value();
    for (size_t i = 0; i < out_dim_args.num_dims; ++i) {
      size *= out_dim_args.dims[i];
    }
    return size;
  }

  absl::Status ToHostBuffer(Buffer* out_buffer) {
    PJRT_Buffer_ToHostBuffer_Args bth_args;
    bth_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    bth_args.extension_start = nullptr;
    bth_args.src = buffer_;
    bth_args.host_layout = nullptr;  // Use default/current layout
    bth_args.dst = nullptr;
    bth_args.dst_size = 0;
    // Output field:
    bth_args.event = nullptr;

    // First call: query size.
    RETURN_IF_PJRT_ERROR(api_->PJRT_Buffer_ToHostBuffer(&bth_args), api_);
    out_buffer->resize(bth_args.dst_size);
    bth_args.dst = out_buffer->data();
    DLOG(INFO) << "ToHostBuffer() transferring buffer of size: " << bth_args.dst_size;
    // Second call: transfer.
    RETURN_IF_PJRT_ERROR(api_->PJRT_Buffer_ToHostBuffer(&bth_args), api_);
    RETURN_IF_ERROR(PJRTAwaitAndDestroyEvent(bth_args.event, api_));
    return absl::OkStatus();
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

  absl::StatusOr<std::vector<PJRTBuffer>> Execute(std::vector<PJRTBuffer>&& input_buffers) {
    std::vector<PJRT_Buffer*> input_buffer_ptrs;
    input_buffer_ptrs.reserve(input_buffers.size());
    for (auto& buffer : input_buffers) {
      input_buffer_ptrs.push_back(buffer.GetBuffer());
    }
    PJRT_Buffer* const* input_buffer_array_ptr = input_buffer_ptrs.data();
    PJRT_Buffer* const* const* all_devices_arg_lists = &input_buffer_array_ptr;

    // Allocate array for the outputs.
    auto status_or_num_outputs = NumOutputs();
    RETURN_IF_ERROR(status_or_num_outputs.status());
    std::vector<PJRT_Buffer*> device_output_buffers_list(status_or_num_outputs.value());
    PJRT_Buffer** device_output_buffers_array_ptr = device_output_buffers_list.data();
    PJRT_Buffer** const* output_lists_for_execute_arg = &device_output_buffers_array_ptr;

    // Output field for completion event:
    PJRT_Event* execution_complete_event = nullptr;

    PJRT_ExecuteOptions execute_options;
    execute_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    execute_options.extension_start = nullptr;
    execute_options.launch_id = 0;  // Default
    execute_options.non_donatable_input_indices = nullptr;
    execute_options.num_non_donatable_input_indices = 0;
    execute_options.context = nullptr;
    execute_options.send_callbacks = nullptr;
    execute_options.recv_callbacks = nullptr;
    execute_options.num_send_ops = 0;
    execute_options.num_recv_ops = 0;

    PJRT_LoadedExecutable_Execute_Args execute_args;
    execute_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.extension_start = nullptr;
    execute_args.executable = loaded_executable;
    execute_args.options = &execute_options;
    execute_args.argument_lists = all_devices_arg_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = input_buffers.size();
    execute_args.execute_device = nullptr;
    execute_args.output_lists = output_lists_for_execute_arg;
    execute_args.device_complete_events = &execution_complete_event;

    DLOG(INFO) << "Calling PJRT_LoadedExecutable_Execute";
    RETURN_IF_PJRT_ERROR(api->PJRT_LoadedExecutable_Execute(&execute_args), api);
    RETURN_IF_ERROR(PJRTAwaitAndDestroyEvent(execution_complete_event, api));

    std::vector<PJRTBuffer> output_buffers;
    output_buffers.reserve(device_output_buffers_list.size());
    for (auto& buffer : device_output_buffers_list) {
      output_buffers.emplace_back(buffer, api);
    }
    return output_buffers;
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
    LOG(INFO) << "| Compilation statistics for:";
    LOG(INFO) << "|   plugin: " << path;
    LOG(INFO) << "|   method: " << method;
    LOG(INFO) << "| Generated code size (bytes) : " << stats_args.generated_code_size_in_bytes;
    LOG(INFO) << "| Argument size       (bytes) : " << stats_args.argument_size_in_bytes;
    LOG(INFO) << "| Output size         (bytes) : " << stats_args.output_size_in_bytes;
    LOG(INFO) << "| Alias size          (bytes) : " << stats_args.alias_size_in_bytes;
    LOG(INFO) << "| Temp size           (bytes): " << stats_args.temp_size_in_bytes;
    LOG(INFO) << "| Host generated code size (bytes): "
              << stats_args.host_generated_code_size_in_bytes;
    LOG(INFO) << "| Host argument size       (bytes): " << stats_args.host_argument_size_in_bytes;
    LOG(INFO) << "| Host output size         (bytes): " << stats_args.host_output_size_in_bytes;
    LOG(INFO) << "| Host alias size          (bytes): " << stats_args.host_alias_size_in_bytes;
    LOG(INFO) << "| Host temp size           (bytes): " << stats_args.host_temp_size_in_bytes;
    LOG(INFO) << "-------------------------------------------------------------"
                 "-------------------";
  }

  absl::StatusOr<size_t> SizeOfGeneratedCodeInBytes() const {
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args size_args;
    size_args.struct_size = PJRT_Executable_SizeOfGeneratedCodeInBytes_Args_STRUCT_SIZE;
    size_args.extension_start = nullptr;
    size_args.executable = executable;
    RETURN_IF_PJRT_ERROR(api->PJRT_Executable_SizeOfGeneratedCodeInBytes(&size_args), api);
    return size_args.size_in_bytes;
  }

  absl::StatusOr<int> NumOutputs() const {
    PJRT_Executable_NumOutputs_Args num_outputs_args;
    num_outputs_args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
    num_outputs_args.extension_start = nullptr;
    num_outputs_args.executable = executable;
    RETURN_IF_PJRT_ERROR(api->PJRT_Executable_NumOutputs(&num_outputs_args), api);
    size_t num_outputs = num_outputs_args.num_outputs;
    return num_outputs;
  }

  absl::StatusOr<std::vector<PJRT_Buffer_Type>> OutputElementTypes() const {
    PJRT_Executable_OutputElementTypes_Args types_args;
    types_args.struct_size = PJRT_Executable_OutputElementTypes_Args_STRUCT_SIZE;
    types_args.extension_start = nullptr;
    types_args.executable = executable;
    RETURN_IF_PJRT_ERROR(api->PJRT_Executable_OutputElementTypes(&types_args), api);
    return std::vector<PJRT_Buffer_Type>(types_args.output_types,
                                         types_args.output_types + types_args.num_output_types);
  }

  absl::StatusOr<std::vector<std::vector<int64_t>>> OutputDimensions() const {
    PJRT_Executable_OutputDimensions_Args dims_args;
    dims_args.struct_size = PJRT_Executable_OutputDimensions_Args_STRUCT_SIZE;
    dims_args.extension_start = nullptr;
    dims_args.executable = executable;
    RETURN_IF_PJRT_ERROR(api->PJRT_Executable_OutputDimensions(&dims_args), api);
    std::vector<std::vector<int64_t>> dims;
    size_t offset = 0;
    for (size_t i = 0; i < dims_args.num_outputs; ++i) {
      dims.push_back(std::vector<int64_t>(dims_args.dims + offset,
                                          dims_args.dims + offset + dims_args.dim_sizes[i]));
      offset += dims_args.dim_sizes[i];
    }
    return dims;
  }
};

PJRTCompiledPlugin::~PJRTCompiledPlugin() {}

absl::StatusOr<PluginState> PJRTCompiledPlugin::Init(const std::vector<Buffer>& inputs) const {
  if (inputs.size() != input_buffer_names_.size()) {
    return absl::InvalidArgumentError(
        "PJRTCompiledPlugin::Init() inputs size doesn't match expected buffer number.");
  }

  std::vector<PJRTBuffer> input_buffers;
  input_buffers.reserve(inputs.size() + 1);
  // First argument: the platform.
  Buffer platform_buffer(sizeof(int32_t));
  reinterpret_cast<int32_t*>(platform_buffer.data())[0] = 0;
  auto status_or_platform_buffer =
      PJRTBuffer::FromHostCopy(platform_buffer, PJRT_Buffer_Type_S32, {1}, ctx_);
  RETURN_IF_ERROR(status_or_platform_buffer.status());
  input_buffers.push_back(std::move(status_or_platform_buffer.value()));
  // Audio input buffer arguments.
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto status_or_buffer = PJRTBuffer::FromHostCopy(inputs[i], /*type=*/audio_buffer_type_,
                                                     /*dims=*/{audio_buffer_size_}, ctx_);
    RETURN_IF_ERROR(status_or_buffer.status());
    input_buffers.push_back(std::move(status_or_buffer.value()));
  }

  auto status_or_output_buffers = init_fn_->Execute(std::move(input_buffers));
  RETURN_IF_ERROR(status_or_output_buffers.status());
  auto& output_buffers = status_or_output_buffers.value();
  if (output_buffers.size() != state_size_) {
    return absl::InternalError(
        absl::StrCat("PJRTCompuiledPlugin::Init() output size does not match state_size_."));
  }

  std::vector<Buffer> state(state_size_);
  for (size_t i = 0; i < state.size(); ++i) {
    RETURN_IF_ERROR(output_buffers[i].ToHostBuffer(&state[i]));
  }
  LOG(INFO) << "Plugin state initialized: " << state.size() << " elements.";
  return state;
}

absl::Status PJRTCompiledPlugin::Update(const std::vector<Buffer>& inputs, PluginState* state,
                                        std::vector<Buffer>* outputs) const {
  if (inputs.size() != input_buffer_names_.size()) {
    return absl::InvalidArgumentError(
        "PJRTCompiledPlugin::Update() inputs size doesn't match expected buffer number.");
  }
  if (outputs->size() != output_buffer_names_.size()) {
    return absl::InvalidArgumentError(
        "PJRTCompiledPlugin::Update() outputs size doesn't match expected buffer number.");
  }

  DLOG(INFO) << "Update() preparing buffers";
  std::vector<PJRTBuffer> input_buffers;
  input_buffers.reserve(inputs.size() + state_size_ + 1);
  // First argument: the platform.
  Buffer platform_buffer(sizeof(int32_t));
  reinterpret_cast<int32_t*>(platform_buffer.data())[0] = 0;
  auto status_or_platform_buffer =
      PJRTBuffer::FromHostCopy(platform_buffer, PJRT_Buffer_Type_S32, {1}, ctx_);
  RETURN_IF_ERROR(status_or_platform_buffer.status());
  input_buffers.push_back(std::move(status_or_platform_buffer.value()));
  // State.
  for (size_t i = 0; i < state->size(); ++i) {
    auto status_or_buffer = PJRTBuffer::FromHostCopy(state->at(i), /*type=*/state_types_[i],
                                                     /*dims=*/state_dimensions_[i], ctx_);
    RETURN_IF_ERROR(status_or_buffer.status());
    input_buffers.push_back(std::move(status_or_buffer.value()));
  }
  // Audio input buffer arguments.
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto status_or_buffer = PJRTBuffer::FromHostCopy(inputs[i], /*type=*/audio_buffer_type_,
                                                     /*dims=*/{audio_buffer_size_}, ctx_);
    RETURN_IF_ERROR(status_or_buffer.status());
    input_buffers.push_back(std::move(status_or_buffer.value()));
  }

  DLOG(INFO) << "Update() buffers done. Calling Execute.";
  auto status_or_output_buffers = update_fn_->Execute(std::move(input_buffers));
  RETURN_IF_ERROR(status_or_output_buffers.status());
  auto& output_buffers = status_or_output_buffers.value();
  if (output_buffers.size() != output_buffer_names_.size() + state_size_) {
    return absl::InternalError(absl::StrCat(
        "PJRTCompuiledPlugin::Update() output size does not match output buffers + state_size_."));
  }
  DLOG(INFO) << "Execute finished.";

  // Get the states.
  for (size_t i = 0; i < state->size(); ++i) {
    RETURN_IF_ERROR(output_buffers[i].ToHostBuffer(&state->at(i)));
  }
  DLOG(INFO) << "Copied state buffers.";
  // Get the output buffers.
  for (size_t i = 0; i < outputs->size(); ++i) {
    RETURN_IF_ERROR(output_buffers[i + state_size_].ToHostBuffer(&outputs->at(i)));
  }
  DLOG(INFO) << "Copied outputs.";
  return absl::OkStatus();
}

PJRTPluginRunner::~PJRTPluginRunner() {}

absl::StatusOr<std::unique_ptr<PJRTPluginRunner>> PJRTPluginRunner::LoadPlugin(
    absl::string_view path) {
  std::unique_ptr<PJRTPluginRunner> plugin(new PJRTPluginRunner());
  plugin->path_ = path;

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
  compiled_plugin->ctx_ = ctx_.get();
  compiled_plugin->input_buffer_names_ = input_buffers;
  compiled_plugin->output_buffer_names_ = output_buffers;
  compiled_plugin->audio_buffer_size_ = buffer_size;
  compiled_plugin->sample_rate_ = sample_rate;

  std::vector<ArgumentTransform> transforms;
  transforms.push_back(RefineType(MlirTensorType({}, "i32")));  // platform index
  for (const std::string& _ : input_buffers) {
    transforms.push_back(RefineType(MlirTensorType({buffer_size}, "f32")));
  }
  transforms.push_back(ReplaceWithConstant(sample_rate));  // sample rate

  std::map<std::string, ScalarValue> global_to_value;
  global_to_value["BufferSize"] = buffer_size;
  // TODO: support platforms other than CPU.
  global_to_value["_platform_index"] = 0;

  auto status_or_init_mlir = MlirPipeline(init_fn_mlir_, transforms, global_to_value);
  RETURN_IF_ERROR(status_or_init_mlir.status());
  std::string init_mlir = status_or_init_mlir.value();
  LOG(INFO) << "Compiling plugin init method MLIR:\n" << init_mlir;

  auto status_or_init_excutable = PJRTExecutable::Load(init_mlir, ctx_.get());
  RETURN_IF_ERROR(status_or_init_excutable.status());
  compiled_plugin->init_fn_ = std::move(status_or_init_excutable.value());
  compiled_plugin->init_fn_->PrintStats(path_, "init");

  // TODO: verify plugin inputs.

  auto status_or_num_outputs = compiled_plugin->init_fn_->NumOutputs();
  RETURN_IF_ERROR(status_or_num_outputs.status());
  size_t num_outputs = status_or_num_outputs.value();
  LOG(INFO) << "Plugin state size: " << num_outputs;

  // Get the shapes and dtypes of the plugin state.
  auto status_or_element_types = compiled_plugin->init_fn_->OutputElementTypes();
  RETURN_IF_ERROR(status_or_element_types.status());
  const auto& element_types = status_or_element_types.value();
  auto status_or_dimensions = compiled_plugin->init_fn_->OutputDimensions();
  RETURN_IF_ERROR(status_or_dimensions.status());
  const auto& dimensions = status_or_dimensions.value();
  for (size_t i = 0; i < num_outputs; ++i) {
    std::vector<std::string> dim_strs;
    for (int64_t dim : dimensions[i]) {
      dim_strs.push_back(std::to_string(dim));
    }
    LOG(INFO) << "State tensor " << i << " dtype: " << element_types[i] << " shape: ["
              << absl::StrJoin(dim_strs, ",") << "]";
  }
  compiled_plugin->state_size_ = num_outputs;
  compiled_plugin->state_types_ = element_types;
  compiled_plugin->state_dimensions_ = dimensions;

  transforms.clear();
  transforms.push_back(RefineType(MlirTensorType({}, "i32")));  // platform index
  for (size_t i = 0; i < num_outputs; ++i) {
    auto status_or_mlir_type = PJRTBuffer::TypeToMLIR(element_types[i]);
    RETURN_IF_ERROR(status_or_mlir_type.status());
    transforms.push_back(RefineType(MlirTensorType(dimensions[i], status_or_mlir_type.value())));
  }
  for (const std::string& _ : input_buffers) {
    transforms.push_back(RefineType(MlirTensorType({buffer_size}, "f32")));
  }
  transforms.push_back(ReplaceWithConstant(sample_rate));  // sample rate

  auto status_or_update_mlir = MlirPipeline(update_fn_mlir_, transforms, global_to_value);
  RETURN_IF_ERROR(status_or_init_mlir.status());
  std::string update_mlir = status_or_update_mlir.value();
  LOG(INFO) << "Compiling plugin update method MLIR:\n" << update_mlir;

  auto status_or_update_excutable = PJRTExecutable::Load(update_mlir, ctx_.get());
  RETURN_IF_ERROR(status_or_update_excutable.status());
  compiled_plugin->update_fn_ = std::move(status_or_update_excutable.value());
  compiled_plugin->update_fn_->PrintStats(path_, "update");

  return compiled_plugin;
}

}  // namespace jxap
