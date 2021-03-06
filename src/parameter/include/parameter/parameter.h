/*
 * =====================================================================================
 *
 *       Filename:  parameter.h
 *
 *    Description:  interface for shared parameter
 *
 *        Version:  1.0
 *        Created:  07/22/2018 21:20:08
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once

#include "proto/param.pb.h"
#include "system/customer.h"

namespace mltools {

/// @brief the base class of shared parameters. The parameters will be shared
/// between server and worker. For servers, each server will only serve part of
/// the parameters. For workers, each worker will only get parameters needed.
/// Operate based on **ParamCall**.
class Parameter : public Customer {
public:
  Parameter(int id) : Customer(id) {}
  virtual ~Parameter() {}

  typedef std::initializer_list<int> Timestamps;
  typedef ::google::protobuf::RepeatedPtrField<FilterConfig> Filters;

  /// @brief create a request task general for parameters.
  /// Parameters are specified by timestamp, channel, other dependencies.
  static Task request(int channel, int ts = Message::kInvalidTime,
                      const Timestamps &wait = {},
                      const Filters &filters = Filters(),
                      const Range<Key> &kr = Range<Key>::all()) {
    Task req;
    req.set_request(true);
    req.set_key_channel(channel);
    if (ts > Message::kInvalidTime) {
      req.set_time(ts);
    }
    for (auto &t : wait) {
      req.add_wait_time(t);
    }
    for (auto &f : filters) {
      req.add_filter()->CopyFrom(f);
    }
    kr.to(req.mutable_key_range());
    return req;
  }

  /// @brief submit a specific **param** **push**.
  inline int push(Message *msg) {
    msg->task_.mutable_param()->set_push(true);
    return submit(msg);
  }

  /// @brief submit a specific **param** **pull** message.
  inline int pull(Message *msg) {
    msg->task_.mutable_param()->set_push(false);
    return submit(msg);
  }

  virtual void writeToFile(std::string filepath) {}

  /// @brief override the Customer::processRequest
  virtual void processRequest(Message *request) override;

  /// @brief override the Customer::processResponse
  virtual void processResponse(Message *response) override;

protected:
  /// @brief `msg->sender_` request to get values from receiver.
  ///   msg->value_[0][0] = myVal_[msg->key_[0]];
  virtual void getValue(Message *msg) = 0;

  /// @brief `msg->sender_` set the values to receiver.
  ///   myVal_[msg->key_[0]] = msg->value_[0][0];
  virtual void setValue(const Message *msg) = 0;

  /// @brief *msg* contains k-v pairs that need to be backed up by current node.
  virtual void setReplica(const Message *msg) {}

  /// @brief new master node to retrieve the backed up data.
  virtual void getReplica(Message *msg) {}

  /// @brief new master node to recover data from replica node.
  virtual void recover(Message *msg) {}
};
} // namespace mltools
