#include "data/info_parser.h"

namespace mltools {

InfoParser::InfoParser() {
  info_.Clear();
  for (int i = 0; i < kSlotIdMax; ++i) {
    slotsInfo_[i].Clear();
  }
}

InfoParser::~InfoParser() {}

} // namespace mltools
