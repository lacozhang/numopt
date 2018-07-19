/*
 * =====================================================================================
 *
 *       Filename:  assigner.h
 *
 *    Description:  Not implemented yet
 *
 *        Version:  1.0
 *        Created:  07/15/2018 20:22:49
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
#include "proto/node.pb.h"
#include "util/common.h"
#include "util/range.h"

namespace mltools {
/**
 * @brief assign rank to server/worker. Assign key range of parameters to
 * servers.
 */
class NodeAssigner {
public:
  NodeAssigner(int numServers, Range<Key> paramKeyRange) {
    numServers_ = numServers;
    paramkeyRange_ = paramKeyRange;
  }
  ~NodeAssigner() {}

  void assign(Node *node) {
    Range<Key> kr = paramkeyRange_;
    int rank = 0;
    if (node->role() == Node::SERVER) {
      kr = paramkeyRange_.evenDivide(numServers_, serverRank_);
      rank = serverRank_++;
    } else if (node->role() == Node::WORKER) {
      rank = workerRank_++;
    }
    node->set_rank(rank);
    kr.to(node->mutable_key());
  }

  void remove(const Node &node) {}

protected:
  int numServers_ = 0;
  int serverRank_ = 0;
  int workerRank_ = 0;
  Range<Key> paramkeyRange_;
};

/// @brief distribute the data
class DataAssigner {
public:
  DataAssigner() {}
  DataAssigner(const DataConfig &data, int num, bool local) {
    set(data, num, local);
  }
  ~DataAssigner() {}

  void set(const DataConfig &data, int num, bool local);
  bool next(DataConfig *data);

  int currIndex() { return currIdx_; }
  int size() { return parts_.size(); }

private:
  std::vector<DataConfig> parts_;
  int currIdx_ = 0;
};
} // namespace mltools
