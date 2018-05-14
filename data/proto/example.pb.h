// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: example.proto

#ifndef PROTOBUF_example_2eproto__INCLUDED
#define PROTOBUF_example_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3005000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3005001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace protobuf_example_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[4];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
void InitDefaultsSlotInfoImpl();
void InitDefaultsSlotInfo();
void InitDefaultsExampleInfoImpl();
void InitDefaultsExampleInfo();
void InitDefaultsSlotImpl();
void InitDefaultsSlot();
void InitDefaultsExampleImpl();
void InitDefaultsExample();
inline void InitDefaults() {
  InitDefaultsSlotInfo();
  InitDefaultsExampleInfo();
  InitDefaultsSlot();
  InitDefaultsExample();
}
}  // namespace protobuf_example_2eproto
namespace mltools {
class Example;
class ExampleDefaultTypeInternal;
extern ExampleDefaultTypeInternal _Example_default_instance_;
class ExampleInfo;
class ExampleInfoDefaultTypeInternal;
extern ExampleInfoDefaultTypeInternal _ExampleInfo_default_instance_;
class Slot;
class SlotDefaultTypeInternal;
extern SlotDefaultTypeInternal _Slot_default_instance_;
class SlotInfo;
class SlotInfoDefaultTypeInternal;
extern SlotInfoDefaultTypeInternal _SlotInfo_default_instance_;
}  // namespace mltools
namespace mltools {

enum SlotInfo_Format {
  SlotInfo_Format_DENSE = 1,
  SlotInfo_Format_SPARSE = 2,
  SlotInfo_Format_SPARSE_BINARY = 3
};
bool SlotInfo_Format_IsValid(int value);
const SlotInfo_Format SlotInfo_Format_Format_MIN = SlotInfo_Format_DENSE;
const SlotInfo_Format SlotInfo_Format_Format_MAX = SlotInfo_Format_SPARSE_BINARY;
const int SlotInfo_Format_Format_ARRAYSIZE = SlotInfo_Format_Format_MAX + 1;

const ::google::protobuf::EnumDescriptor* SlotInfo_Format_descriptor();
inline const ::std::string& SlotInfo_Format_Name(SlotInfo_Format value) {
  return ::google::protobuf::internal::NameOfEnum(
    SlotInfo_Format_descriptor(), value);
}
inline bool SlotInfo_Format_Parse(
    const ::std::string& name, SlotInfo_Format* value) {
  return ::google::protobuf::internal::ParseNamedEnum<SlotInfo_Format>(
    SlotInfo_Format_descriptor(), name, value);
}
// ===================================================================

class SlotInfo : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:mltools.SlotInfo) */ {
 public:
  SlotInfo();
  virtual ~SlotInfo();

  SlotInfo(const SlotInfo& from);

  inline SlotInfo& operator=(const SlotInfo& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  SlotInfo(SlotInfo&& from) noexcept
    : SlotInfo() {
    *this = ::std::move(from);
  }

  inline SlotInfo& operator=(SlotInfo&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const SlotInfo& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const SlotInfo* internal_default_instance() {
    return reinterpret_cast<const SlotInfo*>(
               &_SlotInfo_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(SlotInfo* other);
  friend void swap(SlotInfo& a, SlotInfo& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline SlotInfo* New() const PROTOBUF_FINAL { return New(NULL); }

  SlotInfo* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const SlotInfo& from);
  void MergeFrom(const SlotInfo& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(SlotInfo* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  typedef SlotInfo_Format Format;
  static const Format DENSE =
    SlotInfo_Format_DENSE;
  static const Format SPARSE =
    SlotInfo_Format_SPARSE;
  static const Format SPARSE_BINARY =
    SlotInfo_Format_SPARSE_BINARY;
  static inline bool Format_IsValid(int value) {
    return SlotInfo_Format_IsValid(value);
  }
  static const Format Format_MIN =
    SlotInfo_Format_Format_MIN;
  static const Format Format_MAX =
    SlotInfo_Format_Format_MAX;
  static const int Format_ARRAYSIZE =
    SlotInfo_Format_Format_ARRAYSIZE;
  static inline const ::google::protobuf::EnumDescriptor*
  Format_descriptor() {
    return SlotInfo_Format_descriptor();
  }
  static inline const ::std::string& Format_Name(Format value) {
    return SlotInfo_Format_Name(value);
  }
  static inline bool Format_Parse(const ::std::string& name,
      Format* value) {
    return SlotInfo_Format_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  // optional uint64 max_key = 4;
  bool has_max_key() const;
  void clear_max_key();
  static const int kMaxKeyFieldNumber = 4;
  ::google::protobuf::uint64 max_key() const;
  void set_max_key(::google::protobuf::uint64 value);

  // optional uint64 nnz_ele = 5;
  bool has_nnz_ele() const;
  void clear_nnz_ele();
  static const int kNnzEleFieldNumber = 5;
  ::google::protobuf::uint64 nnz_ele() const;
  void set_nnz_ele(::google::protobuf::uint64 value);

  // optional uint64 nnz_ex = 6;
  bool has_nnz_ex() const;
  void clear_nnz_ex();
  static const int kNnzExFieldNumber = 6;
  ::google::protobuf::uint64 nnz_ex() const;
  void set_nnz_ex(::google::protobuf::uint64 value);

  // optional int32 id = 2;
  bool has_id() const;
  void clear_id();
  static const int kIdFieldNumber = 2;
  ::google::protobuf::int32 id() const;
  void set_id(::google::protobuf::int32 value);

  // optional .mltools.SlotInfo.Format format = 1;
  bool has_format() const;
  void clear_format();
  static const int kFormatFieldNumber = 1;
  ::mltools::SlotInfo_Format format() const;
  void set_format(::mltools::SlotInfo_Format value);

  // optional uint64 min_key = 3 [default = 18446744073709551615];
  bool has_min_key() const;
  void clear_min_key();
  static const int kMinKeyFieldNumber = 3;
  ::google::protobuf::uint64 min_key() const;
  void set_min_key(::google::protobuf::uint64 value);

  // @@protoc_insertion_point(class_scope:mltools.SlotInfo)
 private:
  void set_has_format();
  void clear_has_format();
  void set_has_id();
  void clear_has_id();
  void set_has_min_key();
  void clear_has_min_key();
  void set_has_max_key();
  void clear_has_max_key();
  void set_has_nnz_ele();
  void clear_has_nnz_ele();
  void set_has_nnz_ex();
  void clear_has_nnz_ex();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  ::google::protobuf::uint64 max_key_;
  ::google::protobuf::uint64 nnz_ele_;
  ::google::protobuf::uint64 nnz_ex_;
  ::google::protobuf::int32 id_;
  int format_;
  ::google::protobuf::uint64 min_key_;
  friend struct ::protobuf_example_2eproto::TableStruct;
  friend void ::protobuf_example_2eproto::InitDefaultsSlotInfoImpl();
};
// -------------------------------------------------------------------

class ExampleInfo : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:mltools.ExampleInfo) */ {
 public:
  ExampleInfo();
  virtual ~ExampleInfo();

  ExampleInfo(const ExampleInfo& from);

  inline ExampleInfo& operator=(const ExampleInfo& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  ExampleInfo(ExampleInfo&& from) noexcept
    : ExampleInfo() {
    *this = ::std::move(from);
  }

  inline ExampleInfo& operator=(ExampleInfo&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const ExampleInfo& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const ExampleInfo* internal_default_instance() {
    return reinterpret_cast<const ExampleInfo*>(
               &_ExampleInfo_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    1;

  void Swap(ExampleInfo* other);
  friend void swap(ExampleInfo& a, ExampleInfo& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline ExampleInfo* New() const PROTOBUF_FINAL { return New(NULL); }

  ExampleInfo* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const ExampleInfo& from);
  void MergeFrom(const ExampleInfo& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(ExampleInfo* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .mltools.SlotInfo slot = 1;
  int slot_size() const;
  void clear_slot();
  static const int kSlotFieldNumber = 1;
  const ::mltools::SlotInfo& slot(int index) const;
  ::mltools::SlotInfo* mutable_slot(int index);
  ::mltools::SlotInfo* add_slot();
  ::google::protobuf::RepeatedPtrField< ::mltools::SlotInfo >*
      mutable_slot();
  const ::google::protobuf::RepeatedPtrField< ::mltools::SlotInfo >&
      slot() const;

  // optional uint64 num_ex = 2;
  bool has_num_ex() const;
  void clear_num_ex();
  static const int kNumExFieldNumber = 2;
  ::google::protobuf::uint64 num_ex() const;
  void set_num_ex(::google::protobuf::uint64 value);

  // @@protoc_insertion_point(class_scope:mltools.ExampleInfo)
 private:
  void set_has_num_ex();
  void clear_has_num_ex();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  ::google::protobuf::RepeatedPtrField< ::mltools::SlotInfo > slot_;
  ::google::protobuf::uint64 num_ex_;
  friend struct ::protobuf_example_2eproto::TableStruct;
  friend void ::protobuf_example_2eproto::InitDefaultsExampleInfoImpl();
};
// -------------------------------------------------------------------

class Slot : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:mltools.Slot) */ {
 public:
  Slot();
  virtual ~Slot();

  Slot(const Slot& from);

  inline Slot& operator=(const Slot& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  Slot(Slot&& from) noexcept
    : Slot() {
    *this = ::std::move(from);
  }

  inline Slot& operator=(Slot&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Slot& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const Slot* internal_default_instance() {
    return reinterpret_cast<const Slot*>(
               &_Slot_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    2;

  void Swap(Slot* other);
  friend void swap(Slot& a, Slot& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline Slot* New() const PROTOBUF_FINAL { return New(NULL); }

  Slot* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Slot& from);
  void MergeFrom(const Slot& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Slot* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated uint64 key = 2 [packed = true];
  int key_size() const;
  void clear_key();
  static const int kKeyFieldNumber = 2;
  ::google::protobuf::uint64 key(int index) const;
  void set_key(int index, ::google::protobuf::uint64 value);
  void add_key(::google::protobuf::uint64 value);
  const ::google::protobuf::RepeatedField< ::google::protobuf::uint64 >&
      key() const;
  ::google::protobuf::RepeatedField< ::google::protobuf::uint64 >*
      mutable_key();

  // repeated float val = 3 [packed = true];
  int val_size() const;
  void clear_val();
  static const int kValFieldNumber = 3;
  float val(int index) const;
  void set_val(int index, float value);
  void add_val(float value);
  const ::google::protobuf::RepeatedField< float >&
      val() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_val();

  // optional int32 id = 1;
  bool has_id() const;
  void clear_id();
  static const int kIdFieldNumber = 1;
  ::google::protobuf::int32 id() const;
  void set_id(::google::protobuf::int32 value);

  // optional float impt = 4 [default = 1];
  bool has_impt() const;
  void clear_impt();
  static const int kImptFieldNumber = 4;
  float impt() const;
  void set_impt(float value);

  // @@protoc_insertion_point(class_scope:mltools.Slot)
 private:
  void set_has_id();
  void clear_has_id();
  void set_has_impt();
  void clear_has_impt();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  ::google::protobuf::RepeatedField< ::google::protobuf::uint64 > key_;
  mutable int _key_cached_byte_size_;
  ::google::protobuf::RepeatedField< float > val_;
  mutable int _val_cached_byte_size_;
  ::google::protobuf::int32 id_;
  float impt_;
  friend struct ::protobuf_example_2eproto::TableStruct;
  friend void ::protobuf_example_2eproto::InitDefaultsSlotImpl();
};
// -------------------------------------------------------------------

class Example : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:mltools.Example) */ {
 public:
  Example();
  virtual ~Example();

  Example(const Example& from);

  inline Example& operator=(const Example& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  Example(Example&& from) noexcept
    : Example() {
    *this = ::std::move(from);
  }

  inline Example& operator=(Example&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Example& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const Example* internal_default_instance() {
    return reinterpret_cast<const Example*>(
               &_Example_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    3;

  void Swap(Example* other);
  friend void swap(Example& a, Example& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline Example* New() const PROTOBUF_FINAL { return New(NULL); }

  Example* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Example& from);
  void MergeFrom(const Example& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Example* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .mltools.Slot slot = 1;
  int slot_size() const;
  void clear_slot();
  static const int kSlotFieldNumber = 1;
  const ::mltools::Slot& slot(int index) const;
  ::mltools::Slot* mutable_slot(int index);
  ::mltools::Slot* add_slot();
  ::google::protobuf::RepeatedPtrField< ::mltools::Slot >*
      mutable_slot();
  const ::google::protobuf::RepeatedPtrField< ::mltools::Slot >&
      slot() const;

  // optional float eximpt = 2 [default = 1];
  bool has_eximpt() const;
  void clear_eximpt();
  static const int kEximptFieldNumber = 2;
  float eximpt() const;
  void set_eximpt(float value);

  // @@protoc_insertion_point(class_scope:mltools.Example)
 private:
  void set_has_eximpt();
  void clear_has_eximpt();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  ::google::protobuf::RepeatedPtrField< ::mltools::Slot > slot_;
  float eximpt_;
  friend struct ::protobuf_example_2eproto::TableStruct;
  friend void ::protobuf_example_2eproto::InitDefaultsExampleImpl();
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// SlotInfo

// optional .mltools.SlotInfo.Format format = 1;
inline bool SlotInfo::has_format() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void SlotInfo::set_has_format() {
  _has_bits_[0] |= 0x00000010u;
}
inline void SlotInfo::clear_has_format() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void SlotInfo::clear_format() {
  format_ = 1;
  clear_has_format();
}
inline ::mltools::SlotInfo_Format SlotInfo::format() const {
  // @@protoc_insertion_point(field_get:mltools.SlotInfo.format)
  return static_cast< ::mltools::SlotInfo_Format >(format_);
}
inline void SlotInfo::set_format(::mltools::SlotInfo_Format value) {
  assert(::mltools::SlotInfo_Format_IsValid(value));
  set_has_format();
  format_ = value;
  // @@protoc_insertion_point(field_set:mltools.SlotInfo.format)
}

// optional int32 id = 2;
inline bool SlotInfo::has_id() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void SlotInfo::set_has_id() {
  _has_bits_[0] |= 0x00000008u;
}
inline void SlotInfo::clear_has_id() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void SlotInfo::clear_id() {
  id_ = 0;
  clear_has_id();
}
inline ::google::protobuf::int32 SlotInfo::id() const {
  // @@protoc_insertion_point(field_get:mltools.SlotInfo.id)
  return id_;
}
inline void SlotInfo::set_id(::google::protobuf::int32 value) {
  set_has_id();
  id_ = value;
  // @@protoc_insertion_point(field_set:mltools.SlotInfo.id)
}

// optional uint64 min_key = 3 [default = 18446744073709551615];
inline bool SlotInfo::has_min_key() const {
  return (_has_bits_[0] & 0x00000020u) != 0;
}
inline void SlotInfo::set_has_min_key() {
  _has_bits_[0] |= 0x00000020u;
}
inline void SlotInfo::clear_has_min_key() {
  _has_bits_[0] &= ~0x00000020u;
}
inline void SlotInfo::clear_min_key() {
  min_key_ = GOOGLE_ULONGLONG(18446744073709551615);
  clear_has_min_key();
}
inline ::google::protobuf::uint64 SlotInfo::min_key() const {
  // @@protoc_insertion_point(field_get:mltools.SlotInfo.min_key)
  return min_key_;
}
inline void SlotInfo::set_min_key(::google::protobuf::uint64 value) {
  set_has_min_key();
  min_key_ = value;
  // @@protoc_insertion_point(field_set:mltools.SlotInfo.min_key)
}

// optional uint64 max_key = 4;
inline bool SlotInfo::has_max_key() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void SlotInfo::set_has_max_key() {
  _has_bits_[0] |= 0x00000001u;
}
inline void SlotInfo::clear_has_max_key() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void SlotInfo::clear_max_key() {
  max_key_ = GOOGLE_ULONGLONG(0);
  clear_has_max_key();
}
inline ::google::protobuf::uint64 SlotInfo::max_key() const {
  // @@protoc_insertion_point(field_get:mltools.SlotInfo.max_key)
  return max_key_;
}
inline void SlotInfo::set_max_key(::google::protobuf::uint64 value) {
  set_has_max_key();
  max_key_ = value;
  // @@protoc_insertion_point(field_set:mltools.SlotInfo.max_key)
}

// optional uint64 nnz_ele = 5;
inline bool SlotInfo::has_nnz_ele() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void SlotInfo::set_has_nnz_ele() {
  _has_bits_[0] |= 0x00000002u;
}
inline void SlotInfo::clear_has_nnz_ele() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void SlotInfo::clear_nnz_ele() {
  nnz_ele_ = GOOGLE_ULONGLONG(0);
  clear_has_nnz_ele();
}
inline ::google::protobuf::uint64 SlotInfo::nnz_ele() const {
  // @@protoc_insertion_point(field_get:mltools.SlotInfo.nnz_ele)
  return nnz_ele_;
}
inline void SlotInfo::set_nnz_ele(::google::protobuf::uint64 value) {
  set_has_nnz_ele();
  nnz_ele_ = value;
  // @@protoc_insertion_point(field_set:mltools.SlotInfo.nnz_ele)
}

// optional uint64 nnz_ex = 6;
inline bool SlotInfo::has_nnz_ex() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void SlotInfo::set_has_nnz_ex() {
  _has_bits_[0] |= 0x00000004u;
}
inline void SlotInfo::clear_has_nnz_ex() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void SlotInfo::clear_nnz_ex() {
  nnz_ex_ = GOOGLE_ULONGLONG(0);
  clear_has_nnz_ex();
}
inline ::google::protobuf::uint64 SlotInfo::nnz_ex() const {
  // @@protoc_insertion_point(field_get:mltools.SlotInfo.nnz_ex)
  return nnz_ex_;
}
inline void SlotInfo::set_nnz_ex(::google::protobuf::uint64 value) {
  set_has_nnz_ex();
  nnz_ex_ = value;
  // @@protoc_insertion_point(field_set:mltools.SlotInfo.nnz_ex)
}

// -------------------------------------------------------------------

// ExampleInfo

// repeated .mltools.SlotInfo slot = 1;
inline int ExampleInfo::slot_size() const {
  return slot_.size();
}
inline void ExampleInfo::clear_slot() {
  slot_.Clear();
}
inline const ::mltools::SlotInfo& ExampleInfo::slot(int index) const {
  // @@protoc_insertion_point(field_get:mltools.ExampleInfo.slot)
  return slot_.Get(index);
}
inline ::mltools::SlotInfo* ExampleInfo::mutable_slot(int index) {
  // @@protoc_insertion_point(field_mutable:mltools.ExampleInfo.slot)
  return slot_.Mutable(index);
}
inline ::mltools::SlotInfo* ExampleInfo::add_slot() {
  // @@protoc_insertion_point(field_add:mltools.ExampleInfo.slot)
  return slot_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::mltools::SlotInfo >*
ExampleInfo::mutable_slot() {
  // @@protoc_insertion_point(field_mutable_list:mltools.ExampleInfo.slot)
  return &slot_;
}
inline const ::google::protobuf::RepeatedPtrField< ::mltools::SlotInfo >&
ExampleInfo::slot() const {
  // @@protoc_insertion_point(field_list:mltools.ExampleInfo.slot)
  return slot_;
}

// optional uint64 num_ex = 2;
inline bool ExampleInfo::has_num_ex() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void ExampleInfo::set_has_num_ex() {
  _has_bits_[0] |= 0x00000001u;
}
inline void ExampleInfo::clear_has_num_ex() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void ExampleInfo::clear_num_ex() {
  num_ex_ = GOOGLE_ULONGLONG(0);
  clear_has_num_ex();
}
inline ::google::protobuf::uint64 ExampleInfo::num_ex() const {
  // @@protoc_insertion_point(field_get:mltools.ExampleInfo.num_ex)
  return num_ex_;
}
inline void ExampleInfo::set_num_ex(::google::protobuf::uint64 value) {
  set_has_num_ex();
  num_ex_ = value;
  // @@protoc_insertion_point(field_set:mltools.ExampleInfo.num_ex)
}

// -------------------------------------------------------------------

// Slot

// optional int32 id = 1;
inline bool Slot::has_id() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void Slot::set_has_id() {
  _has_bits_[0] |= 0x00000001u;
}
inline void Slot::clear_has_id() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void Slot::clear_id() {
  id_ = 0;
  clear_has_id();
}
inline ::google::protobuf::int32 Slot::id() const {
  // @@protoc_insertion_point(field_get:mltools.Slot.id)
  return id_;
}
inline void Slot::set_id(::google::protobuf::int32 value) {
  set_has_id();
  id_ = value;
  // @@protoc_insertion_point(field_set:mltools.Slot.id)
}

// repeated uint64 key = 2 [packed = true];
inline int Slot::key_size() const {
  return key_.size();
}
inline void Slot::clear_key() {
  key_.Clear();
}
inline ::google::protobuf::uint64 Slot::key(int index) const {
  // @@protoc_insertion_point(field_get:mltools.Slot.key)
  return key_.Get(index);
}
inline void Slot::set_key(int index, ::google::protobuf::uint64 value) {
  key_.Set(index, value);
  // @@protoc_insertion_point(field_set:mltools.Slot.key)
}
inline void Slot::add_key(::google::protobuf::uint64 value) {
  key_.Add(value);
  // @@protoc_insertion_point(field_add:mltools.Slot.key)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::uint64 >&
Slot::key() const {
  // @@protoc_insertion_point(field_list:mltools.Slot.key)
  return key_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::uint64 >*
Slot::mutable_key() {
  // @@protoc_insertion_point(field_mutable_list:mltools.Slot.key)
  return &key_;
}

// repeated float val = 3 [packed = true];
inline int Slot::val_size() const {
  return val_.size();
}
inline void Slot::clear_val() {
  val_.Clear();
}
inline float Slot::val(int index) const {
  // @@protoc_insertion_point(field_get:mltools.Slot.val)
  return val_.Get(index);
}
inline void Slot::set_val(int index, float value) {
  val_.Set(index, value);
  // @@protoc_insertion_point(field_set:mltools.Slot.val)
}
inline void Slot::add_val(float value) {
  val_.Add(value);
  // @@protoc_insertion_point(field_add:mltools.Slot.val)
}
inline const ::google::protobuf::RepeatedField< float >&
Slot::val() const {
  // @@protoc_insertion_point(field_list:mltools.Slot.val)
  return val_;
}
inline ::google::protobuf::RepeatedField< float >*
Slot::mutable_val() {
  // @@protoc_insertion_point(field_mutable_list:mltools.Slot.val)
  return &val_;
}

// optional float impt = 4 [default = 1];
inline bool Slot::has_impt() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void Slot::set_has_impt() {
  _has_bits_[0] |= 0x00000002u;
}
inline void Slot::clear_has_impt() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void Slot::clear_impt() {
  impt_ = 1;
  clear_has_impt();
}
inline float Slot::impt() const {
  // @@protoc_insertion_point(field_get:mltools.Slot.impt)
  return impt_;
}
inline void Slot::set_impt(float value) {
  set_has_impt();
  impt_ = value;
  // @@protoc_insertion_point(field_set:mltools.Slot.impt)
}

// -------------------------------------------------------------------

// Example

// repeated .mltools.Slot slot = 1;
inline int Example::slot_size() const {
  return slot_.size();
}
inline void Example::clear_slot() {
  slot_.Clear();
}
inline const ::mltools::Slot& Example::slot(int index) const {
  // @@protoc_insertion_point(field_get:mltools.Example.slot)
  return slot_.Get(index);
}
inline ::mltools::Slot* Example::mutable_slot(int index) {
  // @@protoc_insertion_point(field_mutable:mltools.Example.slot)
  return slot_.Mutable(index);
}
inline ::mltools::Slot* Example::add_slot() {
  // @@protoc_insertion_point(field_add:mltools.Example.slot)
  return slot_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::mltools::Slot >*
Example::mutable_slot() {
  // @@protoc_insertion_point(field_mutable_list:mltools.Example.slot)
  return &slot_;
}
inline const ::google::protobuf::RepeatedPtrField< ::mltools::Slot >&
Example::slot() const {
  // @@protoc_insertion_point(field_list:mltools.Example.slot)
  return slot_;
}

// optional float eximpt = 2 [default = 1];
inline bool Example::has_eximpt() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void Example::set_has_eximpt() {
  _has_bits_[0] |= 0x00000001u;
}
inline void Example::clear_has_eximpt() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void Example::clear_eximpt() {
  eximpt_ = 1;
  clear_has_eximpt();
}
inline float Example::eximpt() const {
  // @@protoc_insertion_point(field_get:mltools.Example.eximpt)
  return eximpt_;
}
inline void Example::set_eximpt(float value) {
  set_has_eximpt();
  eximpt_ = value;
  // @@protoc_insertion_point(field_set:mltools.Example.eximpt)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace mltools

namespace google {
namespace protobuf {

template <> struct is_proto_enum< ::mltools::SlotInfo_Format> : ::google::protobuf::internal::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::mltools::SlotInfo_Format>() {
  return ::mltools::SlotInfo_Format_descriptor();
}

}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_example_2eproto__INCLUDED
