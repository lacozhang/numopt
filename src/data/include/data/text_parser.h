/*
 * =====================================================================================
 *
 *       Filename:  text_parser.h
 *
 *    Description:  The main interface for read training samples in various text
 * format (libsvm, vw, criteo etc.)
 *
 *        Version:  1.0
 *        Created:  05/07/2018 19:41:58
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#include "proto/dataconfig.pb.h"
#include "proto/example.pb.h"
#include <functional>

#ifndef __TEXT_PARSER_H__
namespace mltools {

// read data in txt format, convert to Proto Buffer Example
class ExampleParser {
public:
  typedef DataConfig::TextFormat TextFormat;
  void init(TextFormat format, bool ignore_feat_grp = false);
  bool toProto(char *, Example *);

private:
  bool parseLibsvm(char *, Example *);
  bool parseVw(char *, Example *);
  bool parseCriteo(char *, Example *);
  bool ignore_feat_grp_;
  std::function<bool(char *, Example *)> parser_;
};
} // namespace mltools
#endif // __TEXT_PARSER_H__
