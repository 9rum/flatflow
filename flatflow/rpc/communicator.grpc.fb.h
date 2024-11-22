// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: communicator
#ifndef GRPC_communicator__INCLUDED
#define GRPC_communicator__INCLUDED

#include "flatflow/rpc/communicator_generated.h"
#include "flatbuffers/grpc.h"

#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/stub_options.h>
#include <grpcpp/impl/codegen/sync_stream.h>

namespace grpc {
class CompletionQueue;
class Channel;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace flatflow {

class Communicator final {
 public:
  static constexpr char const* service_full_name() {
    return "flatflow.Communicator";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status Init(::grpc::ClientContext* context, const flatbuffers::grpc::Message<InitRequest>& request, flatbuffers::grpc::Message<Empty>* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<Empty>>> AsyncInit(::grpc::ClientContext* context, const flatbuffers::grpc::Message<InitRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<Empty>>>(AsyncInitRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<Empty>>> PrepareAsyncInit(::grpc::ClientContext* context, const flatbuffers::grpc::Message<InitRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<Empty>>>(PrepareAsyncInitRaw(context, request, cq));
    }
    virtual ::grpc::Status Broadcast(::grpc::ClientContext* context, const flatbuffers::grpc::Message<BroadcastRequest>& request, flatbuffers::grpc::Message<BroadcastResponse>* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<BroadcastResponse>>> AsyncBroadcast(::grpc::ClientContext* context, const flatbuffers::grpc::Message<BroadcastRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<BroadcastResponse>>>(AsyncBroadcastRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<BroadcastResponse>>> PrepareAsyncBroadcast(::grpc::ClientContext* context, const flatbuffers::grpc::Message<BroadcastRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<BroadcastResponse>>>(PrepareAsyncBroadcastRaw(context, request, cq));
    }
    virtual ::grpc::Status Finalize(::grpc::ClientContext* context, const flatbuffers::grpc::Message<Empty>& request, flatbuffers::grpc::Message<Empty>* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<Empty>>> AsyncFinalize(::grpc::ClientContext* context, const flatbuffers::grpc::Message<Empty>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<Empty>>>(AsyncFinalizeRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<Empty>>> PrepareAsyncFinalize(::grpc::ClientContext* context, const flatbuffers::grpc::Message<Empty>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<Empty>>>(PrepareAsyncFinalizeRaw(context, request, cq));
    }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<Empty>>* AsyncInitRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<InitRequest>& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<Empty>>* PrepareAsyncInitRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<InitRequest>& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<BroadcastResponse>>* AsyncBroadcastRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<BroadcastRequest>& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<BroadcastResponse>>* PrepareAsyncBroadcastRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<BroadcastRequest>& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<Empty>>* AsyncFinalizeRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<Empty>& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<Empty>>* PrepareAsyncFinalizeRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<Empty>& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status Init(::grpc::ClientContext* context, const flatbuffers::grpc::Message<InitRequest>& request, flatbuffers::grpc::Message<Empty>* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<Empty>>> AsyncInit(::grpc::ClientContext* context, const flatbuffers::grpc::Message<InitRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<Empty>>>(AsyncInitRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<Empty>>> PrepareAsyncInit(::grpc::ClientContext* context, const flatbuffers::grpc::Message<InitRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<Empty>>>(PrepareAsyncInitRaw(context, request, cq));
    }
    ::grpc::Status Broadcast(::grpc::ClientContext* context, const flatbuffers::grpc::Message<BroadcastRequest>& request, flatbuffers::grpc::Message<BroadcastResponse>* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<BroadcastResponse>>> AsyncBroadcast(::grpc::ClientContext* context, const flatbuffers::grpc::Message<BroadcastRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<BroadcastResponse>>>(AsyncBroadcastRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<BroadcastResponse>>> PrepareAsyncBroadcast(::grpc::ClientContext* context, const flatbuffers::grpc::Message<BroadcastRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<BroadcastResponse>>>(PrepareAsyncBroadcastRaw(context, request, cq));
    }
    ::grpc::Status Finalize(::grpc::ClientContext* context, const flatbuffers::grpc::Message<Empty>& request, flatbuffers::grpc::Message<Empty>* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<Empty>>> AsyncFinalize(::grpc::ClientContext* context, const flatbuffers::grpc::Message<Empty>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<Empty>>>(AsyncFinalizeRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<Empty>>> PrepareAsyncFinalize(::grpc::ClientContext* context, const flatbuffers::grpc::Message<Empty>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<Empty>>>(PrepareAsyncFinalizeRaw(context, request, cq));
    }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<Empty>>* AsyncInitRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<InitRequest>& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<Empty>>* PrepareAsyncInitRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<InitRequest>& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<BroadcastResponse>>* AsyncBroadcastRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<BroadcastRequest>& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<BroadcastResponse>>* PrepareAsyncBroadcastRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<BroadcastRequest>& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<Empty>>* AsyncFinalizeRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<Empty>& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<Empty>>* PrepareAsyncFinalizeRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<Empty>& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_Init_;
    const ::grpc::internal::RpcMethod rpcmethod_Broadcast_;
    const ::grpc::internal::RpcMethod rpcmethod_Finalize_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status Init(::grpc::ServerContext* context, const flatbuffers::grpc::Message<InitRequest>* request, flatbuffers::grpc::Message<Empty>* response);
    virtual ::grpc::Status Broadcast(::grpc::ServerContext* context, const flatbuffers::grpc::Message<BroadcastRequest>* request, flatbuffers::grpc::Message<BroadcastResponse>* response);
    virtual ::grpc::Status Finalize(::grpc::ServerContext* context, const flatbuffers::grpc::Message<Empty>* request, flatbuffers::grpc::Message<Empty>* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_Init : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service */*service*/) {}
   public:
    WithAsyncMethod_Init() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_Init() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Init(::grpc::ServerContext* /*context*/, const flatbuffers::grpc::Message<InitRequest>* /*request*/, flatbuffers::grpc::Message<Empty>* /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestInit(::grpc::ServerContext* context, flatbuffers::grpc::Message<InitRequest>* request, ::grpc::ServerAsyncResponseWriter< flatbuffers::grpc::Message<Empty>>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_Broadcast : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service */*service*/) {}
   public:
    WithAsyncMethod_Broadcast() {
      ::grpc::Service::MarkMethodAsync(1);
    }
    ~WithAsyncMethod_Broadcast() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Broadcast(::grpc::ServerContext* /*context*/, const flatbuffers::grpc::Message<BroadcastRequest>* /*request*/, flatbuffers::grpc::Message<BroadcastResponse>* /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestBroadcast(::grpc::ServerContext* context, flatbuffers::grpc::Message<BroadcastRequest>* request, ::grpc::ServerAsyncResponseWriter< flatbuffers::grpc::Message<BroadcastResponse>>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_Finalize : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service */*service*/) {}
   public:
    WithAsyncMethod_Finalize() {
      ::grpc::Service::MarkMethodAsync(2);
    }
    ~WithAsyncMethod_Finalize() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Finalize(::grpc::ServerContext* /*context*/, const flatbuffers::grpc::Message<Empty>* /*request*/, flatbuffers::grpc::Message<Empty>* /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestFinalize(::grpc::ServerContext* context, flatbuffers::grpc::Message<Empty>* request, ::grpc::ServerAsyncResponseWriter< flatbuffers::grpc::Message<Empty>>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(2, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef   WithAsyncMethod_Init<  WithAsyncMethod_Broadcast<  WithAsyncMethod_Finalize<  Service   >   >   >   AsyncService;
  template <class BaseClass>
  class WithGenericMethod_Init : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service */*service*/) {}
   public:
    WithGenericMethod_Init() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_Init() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Init(::grpc::ServerContext* /*context*/, const flatbuffers::grpc::Message<InitRequest>* /*request*/, flatbuffers::grpc::Message<Empty>* /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_Broadcast : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service */*service*/) {}
   public:
    WithGenericMethod_Broadcast() {
      ::grpc::Service::MarkMethodGeneric(1);
    }
    ~WithGenericMethod_Broadcast() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Broadcast(::grpc::ServerContext* /*context*/, const flatbuffers::grpc::Message<BroadcastRequest>* /*request*/, flatbuffers::grpc::Message<BroadcastResponse>* /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_Finalize : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service */*service*/) {}
   public:
    WithGenericMethod_Finalize() {
      ::grpc::Service::MarkMethodGeneric(2);
    }
    ~WithGenericMethod_Finalize() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Finalize(::grpc::ServerContext* /*context*/, const flatbuffers::grpc::Message<Empty>* /*request*/, flatbuffers::grpc::Message<Empty>* /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_Init : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service */*service*/) {}
   public:
    WithStreamedUnaryMethod_Init() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler< flatbuffers::grpc::Message<InitRequest>, flatbuffers::grpc::Message<Empty>>(std::bind(&WithStreamedUnaryMethod_Init<BaseClass>::StreamedInit, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_Init() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Init(::grpc::ServerContext* /*context*/, const flatbuffers::grpc::Message<InitRequest>* /*request*/, flatbuffers::grpc::Message<Empty>* /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedInit(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< flatbuffers::grpc::Message<InitRequest>,flatbuffers::grpc::Message<Empty>>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_Broadcast : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service */*service*/) {}
   public:
    WithStreamedUnaryMethod_Broadcast() {
      ::grpc::Service::MarkMethodStreamed(1,
        new ::grpc::internal::StreamedUnaryHandler< flatbuffers::grpc::Message<BroadcastRequest>, flatbuffers::grpc::Message<BroadcastResponse>>(std::bind(&WithStreamedUnaryMethod_Broadcast<BaseClass>::StreamedBroadcast, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_Broadcast() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Broadcast(::grpc::ServerContext* /*context*/, const flatbuffers::grpc::Message<BroadcastRequest>* /*request*/, flatbuffers::grpc::Message<BroadcastResponse>* /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedBroadcast(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< flatbuffers::grpc::Message<BroadcastRequest>,flatbuffers::grpc::Message<BroadcastResponse>>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_Finalize : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service */*service*/) {}
   public:
    WithStreamedUnaryMethod_Finalize() {
      ::grpc::Service::MarkMethodStreamed(2,
        new ::grpc::internal::StreamedUnaryHandler< flatbuffers::grpc::Message<Empty>, flatbuffers::grpc::Message<Empty>>(std::bind(&WithStreamedUnaryMethod_Finalize<BaseClass>::StreamedFinalize, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_Finalize() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Finalize(::grpc::ServerContext* /*context*/, const flatbuffers::grpc::Message<Empty>* /*request*/, flatbuffers::grpc::Message<Empty>* /*response*/) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedFinalize(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< flatbuffers::grpc::Message<Empty>,flatbuffers::grpc::Message<Empty>>* server_unary_streamer) = 0;
  };
  typedef   WithStreamedUnaryMethod_Init<  WithStreamedUnaryMethod_Broadcast<  WithStreamedUnaryMethod_Finalize<  Service   >   >   >   StreamedUnaryService;
  typedef   Service   SplitStreamedService;
  typedef   WithStreamedUnaryMethod_Init<  WithStreamedUnaryMethod_Broadcast<  WithStreamedUnaryMethod_Finalize<  Service   >   >   >   StreamedService;
};

}  // namespace flatflow


#endif  // GRPC_communicator__INCLUDED
