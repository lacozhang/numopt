// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: range.proto

#ifndef PROTOBUF_range_2eproto__INCLUDED
#define PROTOBUF_range_2eproto__INCLUDED

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
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace protobuf_range_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[1];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
void InitDefaultsPbRangeImpl();
void InitDefaultsPbRange();
inline void InitDefaults() {
  InitDefaultsPbRange();
}
}  // namespace protobuf_range_2eproto
namespace mltools {
class PbRange;
class PbRangeDefaultTypeInternal;
extern PbRangeDefaultTypeInternal _PbRange_default_instance_;
}  // namespace mltools
namespace mltools {

// ===================================================================

class PbRange : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:mltools.PbRange) */ {
 public:
  PbRange();
  virtual ~PbRange();

  PbRange(const PbRange& from);

  inline PbRange& operator=(const PbRange& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  PbRange(PbRange&& from) noexcept
    : PbRange() {
    *this = ::std::move(from);
  }

  inline PbRange& operator=(PbRange&& from) noexcept {
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
  static const PbRange& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const PbRange* internal_default_instance() {
    return reinterpret_cast<const PbRange*>(
               &_PbRange_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(PbRange* other);
  friend void swap(PbRange& a, PbRange& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline PbRange* New() const PROTOBUF_FINAL { return New(NULL); }

  PbRange* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const PbRange& from);
  void MergeFrom(const PbRange& from);
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
  void InternalSwap(PbRange* other);
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

  // required uint64 begin = 1;
  bool has_begin() const;
  void clear_begin();
  static const int kBeginFieldNumber = 1;
  ::google::protobuf::uint64 begin() const;
  void set_begin(::google::protobuf::uint64 value);

  // required uint64 end = 2;
  bool has_end() const;
  void clear_end();
  static const int kEndFieldNumber = 2;
  ::google::protobuf::uint64 end() const;
  void set_end(::google::protobuf::uint64 value);

  // @@protoc_insertion_point(class_scope:mltools.PbRange)
 private:
  void set_has_begin();
  void clear_has_begin();
  void set_has_end();
  void clear_has_end();

  // helper for ByteSizeLong()
  size_t RequiredFieldsByteSizeFallback() const;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  ::google::protobuf::uint64 begin_;
  ::google::protobuf::uint64 end_;
  friend struct ::protobuf_range_2eproto::TableStruct;
  friend void ::protobuf_range_2eproto::InitDefaultsPbRangeImpl();
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// PbRange

// required uint64 begin = 1;
inline bool PbRange::has_begin() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void PbRange::set_has_begin() {
  _has_bits_[0] |= 0x00000001u;
}
inline void PbRange::clear_has_begin() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void PbRange::clear_begin() {
  begin_ = GOOGLE_ULONGLONG(0);
  clear_has_begin();
}
inline ::google::protobuf::uint64 PbRange::begin() const {
  // @@protoc_insertion_point(field_get:mltools.PbRange.begin)
  return begin_;
}
inline void PbRange::set_begin(::google::protobuf::uint64 value) {
  set_has_begin();
  begin_ = value;
  // @@protoc_insertion_point(field_set:mltools.PbRange.begin)
}

// required uint64 end = 2;
inline bool PbRange::has_end() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void PbRange::set_has_end() {
  _has_bits_[0] |= 0x00000002u;
}
inline void PbRange::clear_has_end() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void PbRange::clear_end() {
  end_ = GOOGLE_ULONGLONG(0);
  clear_has_end();
}
inline ::google::protobuf::uint64 PbRange::end() const {
  // @@protoc_insertion_point(field_get:mltools.PbRange.end)
  return end_;
}
inline void PbRange::set_end(::google::protobuf::uint64 value) {
  set_has_end();
  end_ = value;
  // @@protoc_insertion_point(field_set:mltools.PbRange.end)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mltools

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_range_2eproto__INCLUDED