#include "data/info_parser.h"
#include <algorithm>
#include <glog/logging.h>

namespace mltools {

InfoParser::InfoParser() {
  info_.Clear();
  for (int i = 0; i < static_cast<int>(FeatureConstants::kSlotIdMax); ++i) {
    slotsInfo_[i].Clear();
  }
  num_ex_ = 0;
}

InfoParser::~InfoParser() {}

bool InfoParser::add(Example &ex) {
  for (int i = 0; i < ex.slot_size(); ++i) {
    auto &slot = ex.slot(i);
    if (slot.id() >= static_cast<int>(FeatureConstants::kSlotIdMax)) {
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

bool InfoParser::clear() {
  info_.Clear();
  num_ex_ = 0;
  for (auto &slot : slotsInfo_) {
    slot.Clear();
  }
  return true;
}

const ExampleInfo &InfoParser::info() {
  info_.set_num_ex(num_ex_);
  info_.clear_slot();
  for (int i = 0; i < static_cast<int>(FeatureConstants::kSlotIdMax); ++i) {
    auto &slot = slotsInfo_[i];
    if (!slot.nnz_ele()) {
      continue;
    }
    slot.set_id(i);
    if (i == 0) {
      slot.set_min_key(0);
      slot.set_max_key(1);
    }
    *(info_.add_slot()) = slot;
  }
  return info_;
}

} // namespace mltools
