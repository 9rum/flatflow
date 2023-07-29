// Copyright 2022 Sogang University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The messages and services in this file describe the definitions for
// communicating with Chronica's scheduler.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.30.0
// 	protoc        v3.12.4
// source: communicator.proto

package communicator

import (
	empty "github.com/golang/protobuf/ptypes/empty"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type Schedule int32

const (
	Schedule_STATIC  Schedule = 0
	Schedule_DYNAMIC Schedule = 1
	Schedule_GUIDED  Schedule = 2
)

// Enum value maps for Schedule.
var (
	Schedule_name = map[int32]string{
		0: "STATIC",
		1: "DYNAMIC",
		2: "GUIDED",
	}
	Schedule_value = map[string]int32{
		"STATIC":  0,
		"DYNAMIC": 1,
		"GUIDED":  2,
	}
)

func (x Schedule) Enum() *Schedule {
	p := new(Schedule)
	*p = x
	return p
}

func (x Schedule) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (Schedule) Descriptor() protoreflect.EnumDescriptor {
	return file_communicator_proto_enumTypes[0].Descriptor()
}

func (Schedule) Type() protoreflect.EnumType {
	return &file_communicator_proto_enumTypes[0]
}

func (x Schedule) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use Schedule.Descriptor instead.
func (Schedule) EnumDescriptor() ([]byte, []int) {
	return file_communicator_proto_rawDescGZIP(), []int{0}
}

type InitRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Rank      int64     `protobuf:"varint,1,opt,name=rank,proto3" json:"rank,omitempty"`
	BatchSize int64     `protobuf:"varint,2,opt,name=batch_size,json=batchSize,proto3" json:"batch_size,omitempty"`
	Sizes     []int64   `protobuf:"varint,3,rep,packed,name=sizes,proto3" json:"sizes,omitempty"`
	Groups    []int64   `protobuf:"varint,4,rep,packed,name=groups,proto3" json:"groups,omitempty"`
	Partition *bool     `protobuf:"varint,5,opt,name=partition,proto3,oneof" json:"partition,omitempty"`
	Clause    *Schedule `protobuf:"varint,6,opt,name=clause,proto3,enum=communicator.Schedule,oneof" json:"clause,omitempty"`
}

func (x *InitRequest) Reset() {
	*x = InitRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_communicator_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *InitRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InitRequest) ProtoMessage() {}

func (x *InitRequest) ProtoReflect() protoreflect.Message {
	mi := &file_communicator_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InitRequest.ProtoReflect.Descriptor instead.
func (*InitRequest) Descriptor() ([]byte, []int) {
	return file_communicator_proto_rawDescGZIP(), []int{0}
}

func (x *InitRequest) GetRank() int64 {
	if x != nil {
		return x.Rank
	}
	return 0
}

func (x *InitRequest) GetBatchSize() int64 {
	if x != nil {
		return x.BatchSize
	}
	return 0
}

func (x *InitRequest) GetSizes() []int64 {
	if x != nil {
		return x.Sizes
	}
	return nil
}

func (x *InitRequest) GetGroups() []int64 {
	if x != nil {
		return x.Groups
	}
	return nil
}

func (x *InitRequest) GetPartition() bool {
	if x != nil && x.Partition != nil {
		return *x.Partition
	}
	return false
}

func (x *InitRequest) GetClause() Schedule {
	if x != nil && x.Clause != nil {
		return *x.Clause
	}
	return Schedule_STATIC
}

type BcastRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Epoch       int64    `protobuf:"varint,1,opt,name=epoch,proto3" json:"epoch,omitempty"`
	Rank        int64    `protobuf:"varint,2,opt,name=rank,proto3" json:"rank,omitempty"`
	Coefficient *float64 `protobuf:"fixed64,3,opt,name=coefficient,proto3,oneof" json:"coefficient,omitempty"`
	Intercept   *float64 `protobuf:"fixed64,4,opt,name=intercept,proto3,oneof" json:"intercept,omitempty"`
}

func (x *BcastRequest) Reset() {
	*x = BcastRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_communicator_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *BcastRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*BcastRequest) ProtoMessage() {}

func (x *BcastRequest) ProtoReflect() protoreflect.Message {
	mi := &file_communicator_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use BcastRequest.ProtoReflect.Descriptor instead.
func (*BcastRequest) Descriptor() ([]byte, []int) {
	return file_communicator_proto_rawDescGZIP(), []int{1}
}

func (x *BcastRequest) GetEpoch() int64 {
	if x != nil {
		return x.Epoch
	}
	return 0
}

func (x *BcastRequest) GetRank() int64 {
	if x != nil {
		return x.Rank
	}
	return 0
}

func (x *BcastRequest) GetCoefficient() float64 {
	if x != nil && x.Coefficient != nil {
		return *x.Coefficient
	}
	return 0
}

func (x *BcastRequest) GetIntercept() float64 {
	if x != nil && x.Intercept != nil {
		return *x.Intercept
	}
	return 0
}

type BcastResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Indices []int64 `protobuf:"varint,1,rep,packed,name=indices,proto3" json:"indices,omitempty"`
}

func (x *BcastResponse) Reset() {
	*x = BcastResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_communicator_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *BcastResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*BcastResponse) ProtoMessage() {}

func (x *BcastResponse) ProtoReflect() protoreflect.Message {
	mi := &file_communicator_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use BcastResponse.ProtoReflect.Descriptor instead.
func (*BcastResponse) Descriptor() ([]byte, []int) {
	return file_communicator_proto_rawDescGZIP(), []int{2}
}

func (x *BcastResponse) GetIndices() []int64 {
	if x != nil {
		return x.Indices
	}
	return nil
}

var File_communicator_proto protoreflect.FileDescriptor

var file_communicator_proto_rawDesc = []byte{
	0x0a, 0x12, 0x63, 0x6f, 0x6d, 0x6d, 0x75, 0x6e, 0x69, 0x63, 0x61, 0x74, 0x6f, 0x72, 0x2e, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0c, 0x63, 0x6f, 0x6d, 0x6d, 0x75, 0x6e, 0x69, 0x63, 0x61, 0x74,
	0x6f, 0x72, 0x1a, 0x1b, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x62, 0x75, 0x66, 0x2f, 0x65, 0x6d, 0x70, 0x74, 0x79, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22,
	0xdf, 0x01, 0x0a, 0x0b, 0x49, 0x6e, 0x69, 0x74, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12,
	0x12, 0x0a, 0x04, 0x72, 0x61, 0x6e, 0x6b, 0x18, 0x01, 0x20, 0x01, 0x28, 0x03, 0x52, 0x04, 0x72,
	0x61, 0x6e, 0x6b, 0x12, 0x1d, 0x0a, 0x0a, 0x62, 0x61, 0x74, 0x63, 0x68, 0x5f, 0x73, 0x69, 0x7a,
	0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x03, 0x52, 0x09, 0x62, 0x61, 0x74, 0x63, 0x68, 0x53, 0x69,
	0x7a, 0x65, 0x12, 0x14, 0x0a, 0x05, 0x73, 0x69, 0x7a, 0x65, 0x73, 0x18, 0x03, 0x20, 0x03, 0x28,
	0x03, 0x52, 0x05, 0x73, 0x69, 0x7a, 0x65, 0x73, 0x12, 0x16, 0x0a, 0x06, 0x67, 0x72, 0x6f, 0x75,
	0x70, 0x73, 0x18, 0x04, 0x20, 0x03, 0x28, 0x03, 0x52, 0x06, 0x67, 0x72, 0x6f, 0x75, 0x70, 0x73,
	0x12, 0x21, 0x0a, 0x09, 0x70, 0x61, 0x72, 0x74, 0x69, 0x74, 0x69, 0x6f, 0x6e, 0x18, 0x05, 0x20,
	0x01, 0x28, 0x08, 0x48, 0x00, 0x52, 0x09, 0x70, 0x61, 0x72, 0x74, 0x69, 0x74, 0x69, 0x6f, 0x6e,
	0x88, 0x01, 0x01, 0x12, 0x33, 0x0a, 0x06, 0x63, 0x6c, 0x61, 0x75, 0x73, 0x65, 0x18, 0x06, 0x20,
	0x01, 0x28, 0x0e, 0x32, 0x16, 0x2e, 0x63, 0x6f, 0x6d, 0x6d, 0x75, 0x6e, 0x69, 0x63, 0x61, 0x74,
	0x6f, 0x72, 0x2e, 0x53, 0x63, 0x68, 0x65, 0x64, 0x75, 0x6c, 0x65, 0x48, 0x01, 0x52, 0x06, 0x63,
	0x6c, 0x61, 0x75, 0x73, 0x65, 0x88, 0x01, 0x01, 0x42, 0x0c, 0x0a, 0x0a, 0x5f, 0x70, 0x61, 0x72,
	0x74, 0x69, 0x74, 0x69, 0x6f, 0x6e, 0x42, 0x09, 0x0a, 0x07, 0x5f, 0x63, 0x6c, 0x61, 0x75, 0x73,
	0x65, 0x22, 0xa0, 0x01, 0x0a, 0x0c, 0x42, 0x63, 0x61, 0x73, 0x74, 0x52, 0x65, 0x71, 0x75, 0x65,
	0x73, 0x74, 0x12, 0x14, 0x0a, 0x05, 0x65, 0x70, 0x6f, 0x63, 0x68, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x03, 0x52, 0x05, 0x65, 0x70, 0x6f, 0x63, 0x68, 0x12, 0x12, 0x0a, 0x04, 0x72, 0x61, 0x6e, 0x6b,
	0x18, 0x02, 0x20, 0x01, 0x28, 0x03, 0x52, 0x04, 0x72, 0x61, 0x6e, 0x6b, 0x12, 0x25, 0x0a, 0x0b,
	0x63, 0x6f, 0x65, 0x66, 0x66, 0x69, 0x63, 0x69, 0x65, 0x6e, 0x74, 0x18, 0x03, 0x20, 0x01, 0x28,
	0x01, 0x48, 0x00, 0x52, 0x0b, 0x63, 0x6f, 0x65, 0x66, 0x66, 0x69, 0x63, 0x69, 0x65, 0x6e, 0x74,
	0x88, 0x01, 0x01, 0x12, 0x21, 0x0a, 0x09, 0x69, 0x6e, 0x74, 0x65, 0x72, 0x63, 0x65, 0x70, 0x74,
	0x18, 0x04, 0x20, 0x01, 0x28, 0x01, 0x48, 0x01, 0x52, 0x09, 0x69, 0x6e, 0x74, 0x65, 0x72, 0x63,
	0x65, 0x70, 0x74, 0x88, 0x01, 0x01, 0x42, 0x0e, 0x0a, 0x0c, 0x5f, 0x63, 0x6f, 0x65, 0x66, 0x66,
	0x69, 0x63, 0x69, 0x65, 0x6e, 0x74, 0x42, 0x0c, 0x0a, 0x0a, 0x5f, 0x69, 0x6e, 0x74, 0x65, 0x72,
	0x63, 0x65, 0x70, 0x74, 0x22, 0x29, 0x0a, 0x0d, 0x42, 0x63, 0x61, 0x73, 0x74, 0x52, 0x65, 0x73,
	0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x18, 0x0a, 0x07, 0x69, 0x6e, 0x64, 0x69, 0x63, 0x65, 0x73,
	0x18, 0x01, 0x20, 0x03, 0x28, 0x03, 0x52, 0x07, 0x69, 0x6e, 0x64, 0x69, 0x63, 0x65, 0x73, 0x2a,
	0x2f, 0x0a, 0x08, 0x53, 0x63, 0x68, 0x65, 0x64, 0x75, 0x6c, 0x65, 0x12, 0x0a, 0x0a, 0x06, 0x53,
	0x54, 0x41, 0x54, 0x49, 0x43, 0x10, 0x00, 0x12, 0x0b, 0x0a, 0x07, 0x44, 0x59, 0x4e, 0x41, 0x4d,
	0x49, 0x43, 0x10, 0x01, 0x12, 0x0a, 0x0a, 0x06, 0x47, 0x55, 0x49, 0x44, 0x45, 0x44, 0x10, 0x02,
	0x32, 0xcd, 0x01, 0x0a, 0x0c, 0x43, 0x6f, 0x6d, 0x6d, 0x75, 0x6e, 0x69, 0x63, 0x61, 0x74, 0x6f,
	0x72, 0x12, 0x3b, 0x0a, 0x04, 0x49, 0x6e, 0x69, 0x74, 0x12, 0x19, 0x2e, 0x63, 0x6f, 0x6d, 0x6d,
	0x75, 0x6e, 0x69, 0x63, 0x61, 0x74, 0x6f, 0x72, 0x2e, 0x49, 0x6e, 0x69, 0x74, 0x52, 0x65, 0x71,
	0x75, 0x65, 0x73, 0x74, 0x1a, 0x16, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x45, 0x6d, 0x70, 0x74, 0x79, 0x22, 0x00, 0x12, 0x42,
	0x0a, 0x05, 0x42, 0x63, 0x61, 0x73, 0x74, 0x12, 0x1a, 0x2e, 0x63, 0x6f, 0x6d, 0x6d, 0x75, 0x6e,
	0x69, 0x63, 0x61, 0x74, 0x6f, 0x72, 0x2e, 0x42, 0x63, 0x61, 0x73, 0x74, 0x52, 0x65, 0x71, 0x75,
	0x65, 0x73, 0x74, 0x1a, 0x1b, 0x2e, 0x63, 0x6f, 0x6d, 0x6d, 0x75, 0x6e, 0x69, 0x63, 0x61, 0x74,
	0x6f, 0x72, 0x2e, 0x42, 0x63, 0x61, 0x73, 0x74, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65,
	0x22, 0x00, 0x12, 0x3c, 0x0a, 0x08, 0x46, 0x69, 0x6e, 0x61, 0x6c, 0x69, 0x7a, 0x65, 0x12, 0x16,
	0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66,
	0x2e, 0x45, 0x6d, 0x70, 0x74, 0x79, 0x1a, 0x16, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x45, 0x6d, 0x70, 0x74, 0x79, 0x22, 0x00,
	0x42, 0x27, 0x5a, 0x25, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x39,
	0x72, 0x75, 0x6d, 0x2f, 0x63, 0x68, 0x72, 0x6f, 0x6e, 0x69, 0x63, 0x61, 0x2f, 0x63, 0x6f, 0x6d,
	0x6d, 0x75, 0x6e, 0x69, 0x63, 0x61, 0x74, 0x6f, 0x72, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x33,
}

var (
	file_communicator_proto_rawDescOnce sync.Once
	file_communicator_proto_rawDescData = file_communicator_proto_rawDesc
)

func file_communicator_proto_rawDescGZIP() []byte {
	file_communicator_proto_rawDescOnce.Do(func() {
		file_communicator_proto_rawDescData = protoimpl.X.CompressGZIP(file_communicator_proto_rawDescData)
	})
	return file_communicator_proto_rawDescData
}

var file_communicator_proto_enumTypes = make([]protoimpl.EnumInfo, 1)
var file_communicator_proto_msgTypes = make([]protoimpl.MessageInfo, 3)
var file_communicator_proto_goTypes = []interface{}{
	(Schedule)(0),         // 0: communicator.Schedule
	(*InitRequest)(nil),   // 1: communicator.InitRequest
	(*BcastRequest)(nil),  // 2: communicator.BcastRequest
	(*BcastResponse)(nil), // 3: communicator.BcastResponse
	(*empty.Empty)(nil),   // 4: google.protobuf.Empty
}
var file_communicator_proto_depIdxs = []int32{
	0, // 0: communicator.InitRequest.clause:type_name -> communicator.Schedule
	1, // 1: communicator.Communicator.Init:input_type -> communicator.InitRequest
	2, // 2: communicator.Communicator.Bcast:input_type -> communicator.BcastRequest
	4, // 3: communicator.Communicator.Finalize:input_type -> google.protobuf.Empty
	4, // 4: communicator.Communicator.Init:output_type -> google.protobuf.Empty
	3, // 5: communicator.Communicator.Bcast:output_type -> communicator.BcastResponse
	4, // 6: communicator.Communicator.Finalize:output_type -> google.protobuf.Empty
	4, // [4:7] is the sub-list for method output_type
	1, // [1:4] is the sub-list for method input_type
	1, // [1:1] is the sub-list for extension type_name
	1, // [1:1] is the sub-list for extension extendee
	0, // [0:1] is the sub-list for field type_name
}

func init() { file_communicator_proto_init() }
func file_communicator_proto_init() {
	if File_communicator_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_communicator_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*InitRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_communicator_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*BcastRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_communicator_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*BcastResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	file_communicator_proto_msgTypes[0].OneofWrappers = []interface{}{}
	file_communicator_proto_msgTypes[1].OneofWrappers = []interface{}{}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_communicator_proto_rawDesc,
			NumEnums:      1,
			NumMessages:   3,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_communicator_proto_goTypes,
		DependencyIndexes: file_communicator_proto_depIdxs,
		EnumInfos:         file_communicator_proto_enumTypes,
		MessageInfos:      file_communicator_proto_msgTypes,
	}.Build()
	File_communicator_proto = out.File
	file_communicator_proto_rawDesc = nil
	file_communicator_proto_goTypes = nil
	file_communicator_proto_depIdxs = nil
}
