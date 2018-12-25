/*
 * =====================================================================================
 *
 *       Filename:  kv_map_ps.cpp
 *
 *    Description:  testing key-value map
 *
 *        Version:  1.0
 *        Created:  12/24/2018 21:35:28
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "system/sysutil.h"

namespace mltools {
class Server : public App {
public:
  virtual void processRequest(Message *req) override {
    std::cout << MyNodeID() << " : processing request " << req->task_.time()
              << " from " << req->sender_ << std::endl;
  }

  virtual void slice(const Message &request, const std::vector<Range<Key>> &krs,
                     std::vector<Message *> *requests) override {
    std::cout << MyNodeID() << " : invoked by server" << std::endl;
  }
};

class Worker : public App {
  virtual void processResponse(Message *res) override {
    std::cout << MyNodeID() << ": received response " << res->task_.time()
              << " from " << res->sender_ << std::endl;
  }

  virtual void slice(const Message &request, const std::vector<Range<Key>> &krs,
                     std::vector<Message *> *requests) override {
    std::cout << MyNodeID() << " : invoked by worker" << std::endl;
  }

  virtual void run() override {
    int ts = submit(Task(), kServerGroup);
    wait(ts);

    ts = submit(Task(), kServerGroup);
    wait(ts);

    Message req;
    req.recver_ = kServerGroup;
    req.callback = [this]() {
      std::cout << MyNodeID() << ": request " << lastResponse()->task_.time()
                << " is finished" << std::endl;
    };
    wait(submit(&req));
  }
};

class Scheduler : public App {
  virtual void processResponse(Message *res) override {
    std::cout << MyNodeID() << ": received response " << res->task_.time()
              << " from " << res->sender_ << std::endl;
  }

  virtual void processRequest(Message *req) override {
    std::cout << MyNodeID() << " : processing request " << req->task_.time()
              << " from " << req->sender_ << std::endl;
  }

  virtual void slice(const Message &request, const std::vector<Range<Key>> &krs,
                     std::vector<Message *> *requests) override {
    std::cout << MyNodeID() << " : invoked by scheduler" << std::endl;
  }

  virtual void run() override {
    std::cout << "running from scheduler";
    sys_.manager().waitServersReady();
    sys_.manager().waitWorkersReady();
  }
};

App *App::Create(const std::string &conf) {
  if (IsWorker()) {
    return new Worker();
  }
  if (IsServer()) {
    return new Server();
  }

  if (IsScheduler()) {
    return new Scheduler();
  }

  LOG(FATAL) << "Unknow role type " << MyNode().DebugString();
}
} // namespace mltools

int main(int argc, char *argv[]) { return mltools::RunSystem(argc, argv); }
