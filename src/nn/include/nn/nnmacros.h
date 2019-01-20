#pragma once

#include <glog/logging.h>
#ifndef __NN_MACROS_H__
#define __NN_MACROS_H__

#define NNForbidOperation                                                      \
  do {                                                                         \
    LOG(FATAL) << "Operation " << __FUNCTION__ << " Not Supported";            \
    LOG(FATAL) << "File " << __FILE__ << " line " << __LINE__;                 \
    std::abort();                                                              \
  } while (false);

#define NNForbidOperationMsg(msg)                                              \
  do {                                                                         \
    LOG(FATAL) << msg;                                                         \
    LOG(FATAL) << "code at " << __FUNCTION__ << " happended";                  \
    LOG(FATAL) << "File " << __FILE__ << " line " << __LINE__;                 \
    std::abort();                                                              \
  } while (false);

#endif // !__NN_MACROS_H__
