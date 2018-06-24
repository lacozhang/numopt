#include "data/info_parser.h"
#include <algorithm>
#include <glog/logging.h>

namespace mltools {

InfoParser::InfoParser() {
  info_.Clear();
  for (int i = 0; i < kSlotIdMax; ++i) {
    slotsInfo_[i].Clear();
  }
  num_ex_ = 0;
}

InfoParser::~InfoParser() {}

bool InfoParser::add(Example &ex) {
  for (int i = 0; i < ex.slot_size(); ++i) {
    auto &slot = ex.slot(i);
    if (slot.id() >= kSlotIdMax) {
      LOG(WARNING) << "Slot id " << slot.id() << " larger than kSlotIdMax";
      return false;
    }
    auto &slotInfo = slotsInfo_[slot.id()];
    for (int idx = 0; idx < slot.key_size(); ++idx) {
      uint64 key = slot.key(idx);
      slotInfo.set_max_key(std::max((uint64)slotInfo.max_key(), key));
      slotInfo.set_min_key(std::min((uint64)slotInfo.min_key(), key));
    }

    if (slot.key_size() > 0) {
      if (slot.key_size() == slot.val_size()) {
        slotInfo.set_format(mltools::SlotInfo::SPARSE);
      } else {
        slotInfo.set_format(mltools::SlotInfo::SPARSE_BINARY);
      }
    } else if (slot.val_size() > 0) {
      slotInfo.set_format(mltools::SlotInfo::DENSE);
    }

    slotInfo.set_nnz_ex(slotInfo.nnz_ex() + 1);
    slotInfo.set_nnz_ele(slotInfo.nnz_ele() +
                         std::max(slot.key_size(), slot.val_size()));
  }

  ++num_ex_;
  return true;
}

} // namespace mltools
