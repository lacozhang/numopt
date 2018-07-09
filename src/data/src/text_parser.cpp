/*
 * =====================================================================================
 *
 *       Filename:  text_parser.cpp
 *
 *    Description:  implementation of parser
 *
 *        Version:  1.0
 *        Created:  05/07/2018 19:57:40
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "data/text_parser.h"
#include "data/common.h"
#include "util/murmurhash3.h"
#include "util/strtonum.h"
#include <glog/logging.h>
#include <string.h>

namespace mltools {

void ExampleParser::init(TextFormat format, bool ignore_feag_grp) {
  ignore_feat_grp_ = ignore_feag_grp;
  switch (format) {
  case DataConfig::LIBSVM: {
#ifdef XDEBUG
    LOG(INFO) << "LIB SVM format";
#endif // XDEBUG
    parser_ = [this](char *line, Example *ex) -> bool {
      return parseLibsvm(line, ex);
    };
  } break;
  case DataConfig::VW: {
#ifdef XDEBUG
    LOG(INFO) << "VW Format";
#endif
    parser_ = [this](char *line, Example *ex) -> bool {
      return parseVw(line, ex);
    };
  } break;
  case DataConfig::CRITEO: {
#ifdef XDEBUG
    LOG(INFO) << "CRITEO Format";
#endif
    parser_ = [this](char *line, Example *ex) -> bool {
      return parseCriteo(line, ex);
    };
  } break;
  case DataConfig::ADFEA:
  case DataConfig::TERAFEA:
  case DataConfig::DENSE:
  case DataConfig::SPARSE:
  case DataConfig::SPARSE_BINARY: {
    LOG(FATAL) << "unsupported text format" << std::endl;
    std::abort();
  } break;
  }
}

bool ExampleParser::toProto(char *line, Example *ex) {
  ex->Clear();
  return parser_(line, ex);
}

// libsvm format
// label feature_d:weight feature_id:weight
bool ExampleParser::parseLibsvm(char *line, Example *ex) {
#ifdef XDEBUG
  LOG(INFO) << "Start parsing LibSVM";
#endif

  char *saveptr;
  // label
  float label = 0;
  char *pch = strtok_r(line, " \t\r\n", &saveptr);
  if (!strtofloat(pch, &label)) {
    LOG(ERROR) << "text sample format error " << line;
    return false;
  }
  auto lbl_slot = ex->add_slot();
  lbl_slot->set_id(0);
  lbl_slot->add_val(label);

  pch = strtok_r(NULL, " \t\r\n", &saveptr);
  auto feat_slot = ex->add_slot();
  feat_slot->set_id(1);
  uint64 idx = 0, last_idx = 0;
  float val = 0;
  bool dfltval = false;
  while (pch != NULL) {
    char *it = pch;
    val = 0;
    dfltval = false;
    while (*it != ':' && *it != '\0') {
      ++it;
    }
    if (*it == ':') {
      *it = '\0';
    } else {
      dfltval = true;
    }

    if (!strtou64(pch, &idx)) {
#ifdef XDEBUG
      LOG(INFO) << "Libsvm feature index incorrect: " << pch;
#endif
      return false;
    }
    if (!dfltval && !strtofloat(it + 1, &val)) {
#ifdef XDEBUG
      LOG(INFO) << "Libsvm feature value incorrect: " << (it + 1);
#endif
      return false;
    } else if (dfltval) {
      val = 1.0;
    }

    if (last_idx > idx) {
#ifdef XDEBUG
      LOG(INFO) << "Libsvm feature index must follow ascending order";
#endif
      return false;
    }
    last_idx = idx;

    feat_slot->add_key(idx);
    feat_slot->add_val(val);
    pch = strtok_r(NULL, " \t\r\n", &saveptr);
  }
  return true;
}

// parser training sample in Vw format
// [Label] [Importance]|Namespace Features |Namespace Features ...
// |Namespace Features where Namespace=String[:Value],
// Features=(String[:Value])*
// [Importance] is a float value
bool ExampleParser::parseVw(char *line, Example *ex) {
  if (!line || !ex)
    return false;

  char *end, *end2;
  char *pch = nullptr, *pch2 = nullptr;
  pch = strtok_r(line, "|", &end);
  if (!pch) {
    LOG(ERROR) << "Format error " << line;
    return false;
  }

  int label_fields = 0;
  auto slot = ex->add_slot();
  slot->set_id(0);
  pch2 = strtok_r(pch, " \t\r", &end2);
  while (pch2 != NULL) {
    ++label_fields;
    float val = 0;
    if (!strtofloat(pch, &val)) {
      LOG(ERROR) << "label field format error " << pch;
      return false;
    }
    switch (label_fields) {
    case 1:
      slot->add_val(val);
      break;
    case 2:
      ex->set_wgt(val);
      break;
    default: {
      LOG(ERROR) << "too much fields in label";
      return false;
    } break;
    }
    pch2 = strtok_r(NULL, "\t\r", &end2);
  }

  // parse each namespace
  int feat_grp_id = 1;
  if (ignore_feat_grp_) {
    slot = ex->add_slot();
    slot->set_id(feat_grp_id);
  }
  pch = strtok_r(NULL, "|", &end);
  if (pch == NULL) {
    LOG(ERROR) << "no feature values" << line;
    return false;
  }

  uint64 murmur_out[2] = {0};
  while (pch != NULL) {
    // pass the namespace specification
    pch2 = strtok_r(pch, " \t\r", &end2);

    if (pch2 == NULL) {
      pch = strtok_r(NULL, "|", &end);
      continue;
    }

    if (!ignore_feat_grp_) {
      if (strlen(pch2) == 0) {
        MurmurHash3_x64_128(" ", 1, 512927377, murmur_out);
        feat_grp_id = (murmur_out[0] ^ murmur_out[1]) %
                      static_cast<int>(FeatureConstants::kSlotIdMax);
      } else {
        // TODO: support feature group scaling factor in the future
        MurmurHash3_x64_128(pch2, strlen(pch2), 512927377, murmur_out);
        feat_grp_id = (murmur_out[0] ^ murmur_out[1]) %
                      static_cast<int>(FeatureConstants::kSlotIdMax);
      }
    }

    pch2 = strtok_r(NULL, " \t\r", &end2);
    if (pch2 != NULL && !ignore_feat_grp_) {
      slot = ex->add_slot();
      slot->set_id(feat_grp_id);
    }

    while (pch2 != NULL) {
      if (strlen(pch2) == 0) {
        pch2 = strtok_r(NULL, " \t\r", &end2);
        continue;
      }

      char *ptr = pch2;
      while (*ptr != ':' && *ptr != '\0')
        ++ptr;
      if (*ptr == ':')
        *ptr = '\0', ++ptr;

      uint64 feat_idx = 0;
      float feat_val = 1.0;
      if (!strtou64(pch2, &feat_idx)) {
        MurmurHash3_x64_128(pch2, strlen(pch2), 512927377, murmur_out);
        feat_idx = (murmur_out[0] ^ murmur_out[1]);
      }
      slot->add_key(feat_idx);

      if (*ptr != '\0') {
        if (!strtofloat(ptr, &feat_val)) {
          LOG(ERROR) << "data format error" << line;
          return false;
        }
      }
      slot->add_val(feat_val);
      pch2 = strtok_r(NULL, "\t\r", &end2);
    }

    pch = strtok_r(NULL, "|", &end);
  }
  return true;
}

// criteo ctr dataset:
// The columns are tab separeted with the following schema:
// <label> <integer feature 1> ... <integer feature 13> <categorical feature 1>
// ... <categorical feature 26>
bool ExampleParser::parseCriteo(char *line, Example *ex) {
  if (!line || !ex)
    return false;
  char *p = line;
  char *pp = strchr(line, '\t');
  if (pp == NULL)
    return false;
  *pp = '\0';
  float label;
  if (!strtofloat(p, &label))
    return false;

  auto slot = ex->add_slot();
  slot->set_id(0);
  slot->add_val(label > 0 ? +1 : -1);

  p = pp + 1;
  for (int i = 0; i < 13; ++i) {
    pp = strchr(p, '\t');
    if (pp == NULL)
      return false;
    *pp = '\0';

    int32 cnt = 0;
    if (strtoi32(p, &cnt)) {
      if (i == 0 || !ignore_feat_grp_) {
        slot = ex->add_slot();
        slot->set_id(i + 1);
      }
      uint64 key = kuint64max / 13 * i + cnt;
      slot->add_key(key);
    }
    p = pp + 1;
  }

  uint64 murmur_out[2];
  for (int i = 0; i < 26; ++i) {
    pp = strchr(p, '\t');
    if (pp == NULL) {
      if (i != 25)
        return false;
    } else {
      *pp = '\0';
    }

    int n = strlen(p);
    if (n > 4) {
      if (!ignore_feat_grp_) {
        slot = ex->add_slot();
        slot->set_id(i + 14);
      }

      MurmurHash3_x64_128(p, n, 512927377, murmur_out);
      uint64 key = (murmur_out[0] ^ murmur_out[1]);
      slot->add_key(key);
    }
    p = pp + 1;
  }
  return true;
}
} // namespace mltools
