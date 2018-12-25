/*
 * =====================================================================================
 *
 *       Filename:  customer.h
 *
 *    Description:  Main application interface, anything need to inherit it for
 * ps style.
 *
 *        Version:  1.0
 *        Created:  07/18/2018 21:13:07
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "system/executor.h"
#include "system/message.h"
#include "system/postoffice.h"
#include "util/common.h"

namespace mltools {
/**
 * \brief The common inference of interaction objects, such as application or
 * parameters. Customer has a interface for asynchronous RPC-like interface. A
 * customer can only interact with other customer with same customer-id.
 *
 * How it works:
 * customer A in Node NA can send a request to customer B at node NB if A and B
 * have the same Customer::id. The request message contains the type of action,
 * essential information is specified through Message::task_, other additional
 * information like Message::key_/Message::value_. The customer B first process
 * the message using Customer::processRequest if the prequesite conditions are
 * satisfied; and then send back response to A through Customer::reply. Once A
 * get the response, A will invoke Customer::processResponse.
 *
 * It's an asynchronous interface. 'submit' will return immediately only queued
 * into PostOffice. For each message, there is an associated timestamp for
 * synchronization. For example, customer B want use timestamp to call
 * Customer::waitReceivedRequest. while A can wait on timestamp when it's
 * response is available. Further more, a customer can send a request to
 * customer group, like the server group. Customer::slice will be called to
 * split the message to multiple individual nodes.
 *
 * There are also user-defined filters to encode & decode the message. Mostly to
 * reduce the traffic between nodes.
 */
class Customer {
public:
  /**
   * @brief Id used to identify customer between nodes.
   */
  Customer(int id) : id_(id), sys_(PostOffice::getInstance()), exec_(*this) {
    sys_.manager().addCustomer(this);
  }
  virtual ~Customer() { sys_.manager().removeCustomer(id_); }

  ////// as a request sender ///////
  /**
   * brief Submit a request to a customer with the same id in node **recver**.
   *
   * Sample usage: send a request to all worker node and wait until the response
   * of this request has been received from all the worker nodes. (This is a
   * synchronous call).
   *
   *  Task tk; tk.mutable_sgd()->set_cmd(SGDCall::UPDATE_MODEL);
   *  int ts = submit(tk, kWorkerGroup);
   *  wait(ts); // key point, make this call synchronous.
   *  foo();
   *
   * @param request the arguments of RPC-call.
   * @param recver. Id of a node, can be a node group like kServerGroup
   *
   * @return the timestamp of this request.
   */
  inline int submit(const Task &request, const NodeID &recver) {
    Message msg(request, recver);
    return submit(&msg);
  }

  /**
   * @brief Submit a request message to a remote node.
   *
   * Sample usage: same functionality as function above.
   *  Message req(task, kWorkerGroup);
   *  req.callback_ = [this]() { Foo(); } // callback function will be invoked
   * when response of request is received. wait(submit(&req));
   * @param request contains all the essential information: task_, keys_,
   * value_, callback_ when response received.
   * @return the timestamp of this request.
   */
  inline int submit(Message *msg) { return exec_.submit(msg); }

  /**
   * @brief blocked until the response of request(marked by timestamp) received.
   *
   * If the receiver of this request is a single node, this function will be
   * blocked until a reply message with the same timestamp has been received
   * from this receiver, or the reciver is dead. Otherwise, this function is
   * blocked until all the response from each alive node in node group is
   * received.
   *
   * @param the timestamp of send out request.
   */
  inline void wait(int timestamp) { exec_.waitSentReq(timestamp); }

  /**
   * @brief slice a message into multiple parts according to ranges specified.
   *
   * This function is a factory method, it need to be customized. This function
   * will be called during **submit** called.
   *
   * @param request original request which will be splitted.
   * @param krs a list of key ranges. Single node will be lenght of 1, otherwise
   * it will contain the key ranges of the node group. the key ranges is
   * ordered.
   * @param requests output of sliced message. each message has been initialized
   * properly before call this method.
   */
  virtual void slice(const Message &request, const std::vector<Range<Key>> &krs,
                     std::vector<Message *> *requests) { }

  //////// as a receiver ///////
  /**
   * @brief Function need subclasses to process request from "request->sender".
   *
   * This function will be invoked by exector's processing thread to handle the
   * message that satisfies all the conditions.
   *
   * @param request the received request from other customers.
   */
  virtual void processRequest(Message *request) {}

  /// @brief last recieved request from other customers.
  inline std::shared_ptr<Message> lastRequest() { return exec_.lastRequest(); }

  /**
   * @brief A function need to be subclassed, which will be used to process
   * response from "msg->sender".
   *
   * called by executor's processing thread.
   * @param msg the received response.
   */
  virtual void processResponse(Message *msg) {}

  /// @brief the last response.
  inline std::shared_ptr<Message> lastResponse() {
    return exec_.lastResponse();
  }

  /**
   * @brief blocked until the received request is processed at this node or the
   * sender is dead. If the sender is a node group, then wait for each alive
   * node in this node group.
   *
   * @param timestamp
   * @param sender name of sender.
   */
  inline void waitReceivedRequest(int timestamp, const NodeID &sender) {
    exec_.waitRecvReq(timestamp, sender);
  }

  /**
   * @brief Set the request from specific sender as finished.
   *
   * Set the request with timestamp from sender as finished. Usually, this is
   * done by executor. But sometimes need this to implement synchronization.
   * Sample usage: data aggregation at server nodes. 1) workers push the data to
   * server nodes. 2) each server aggregate the data from pushed server. 3)
   * worker pull the aggregated from server.
   *
   * Implementation:
   * Worker: submit a **push** **request** with timestamp t via
   * `push_task.set_time(t)`; then submit a pull request with timestamp t+2 wich
   * depends timestamp t+1, namely `pull_task.set_time(t+2);
   * pull_task.add_wait_time(t+1);`. Server: wait for pushed data via
   * `waitReceivedReq(t, kWorkerGroup)`; aggregate data; mark the virtual
   * request t+1 as finished via `finishReceivedRequest`. Then all blocked pull
   * requests will be executed by the DAG engine.
   *
   * @param timestamp
   * @param sender
   */
  inline void finishReceivedRequest(int timestamp, const NodeID &sender) {
    exec_.finishRecvReq(timestamp, sender);
  }

  /// @brief #messages finished on timestamp this node received.
  inline int numDoneReceivedRequest(int timestamp, const NodeID &sender) {
    return exec_.queryRecvReq(timestamp, sender);
  }

  /**
   * @brief replies the request message with a response. In default, response is
   * empty ack message.
   */
  void reply(Message *request, Task response = Task()) {
    Message *msg = new Message(response);
    reply(request, msg);
  }

  void reply(Message *request, Message *response) {
    exec_.reply(request, response);
  }

  int id() const { return id_; }

  Executor *executor() { return &exec_; }

protected:
  int id_; /// @brief customer id.
  PostOffice &sys_;
  Executor exec_;

private:
  DISALLOW_COPY_AND_ASSIGN(Customer);
};

/// @brief Base class of application which will be runned by main thread.
class App : public Customer {
public:
  App() : Customer(PostOffice::getInstance().manager().nextCustomerID()) {}
  virtual ~App() {}

  /**
   * @brief the factory method an application must implement.
   *
   * This method will be used whenever main thread want to init the whole
   * workflow.
   * @param conf conf string.
   */
  static App *Create(const std::string &conf);

  /// @brief This function will be executed by main thread immediately after the
  /// instination of App object.
  virtual void run() {
    sys_.manager().waitWorkersReady();
    sys_.manager().waitServersReady();
  }
};
} // namespace mltools
