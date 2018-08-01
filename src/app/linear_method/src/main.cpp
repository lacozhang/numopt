/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  application interface
 *
 *        Version:  1.0
 *        Created:  07/29/2018 10:04:10
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "linear_method/async_sgd.h"
#include "linear_method/darlin.h"
#include "linear_method/model_evaluation.h"

namespace mltools {
App *App::Create(const std::string &confStr) {
  using namespace linear;
  Config conf;
  CHECK(google::protobuf::TextFormat::ParseFromString(confStr, &conf))
      << "failed to parse conf: " << confStr;
  auto role = MyRole();
  App *app = nullptr;
  if (conf.has_darlin()) {
    switch (role) {
    case Node::SCHEDULER:
      break;
    case Node::WORKER:
      break;
    case Node::SERVER:
      break;
    default:
      break;
    }
  } else if (conf.has_async_sgd()) {
    typedef float Real;
    switch (role) {
    case Node::SCHEDULER:
      app = new ASyncSGDScheduler(conf);
      break;
    case Node::WORKER:
      app = new ASyncSGDWorker<Real>(conf);
      break;
    case Node::SERVER:
      app = new ASyncSGDServer<Real>(conf);
      break;
    default:
      CHECK(false) << "wrong role type" << confStr;
      break;
    }
  } else if (conf.has_validation_data()) {
    app = new ModelEvaluation(conf);
  }

  CHECK(app) << "fail to create " << conf.ShortDebugString() << "at "
             << MyNode().ShortDebugString();
  return app;
}
} // namespace mltools

int main(int argc, char *argv[]) {
  mltools::RunSystem(argc, argv);
  return 0;
}
