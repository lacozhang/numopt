/*
 * =====================================================================================
 *
 *       Filename:  parameter.cpp
 *
 *    Description:  implementation of parameter.h
 *
 *        Version:  1.0
 *        Created:  07/22/2018 21:20:33
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "parameter/parameter.h"
#include "parameter/kv_layer.h"
#include "parameter/kv_map.h"
#include "parameter/kv_vector.h"
#include "system/sysutil.h"

namespace mltools {
void Parameter::processRequest(mltools::Message *request) {
  const auto &call = request->task_.param();
  Message *resp = nullptr;
  bool push = call.push();

  if (!push) {
    // a pull request, need to return parameter values.
    resp = new Message(*request);
  }

  if (call.replica()) {
    if (push) {
      setReplica(request);
    } else {
      getReplica(resp);
    }
  } else {
    if (push) {
      setValue(request);
    } else {
      getValue(resp);
    }
  }

  if (resp) {
    reply(request, resp);
  }
}

void Parameter::processResponse(mltools::Message *response) {
  const auto &call = response->task_.param();
  bool push = call.push();
  if (call.replica()) {
    if (push) {
      return;
    }
    if (Range<Key>(response->task_.key_range()) == MyKeyRange()) {
      recover(response);
    } else {
      setReplica(response);
    }
  } else {
    if (!push) {
      setValue(response);
    }
  }
}
} // namespace mltools
