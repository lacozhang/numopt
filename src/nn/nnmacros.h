#pragma once

#include <boost/log/trivial.hpp>
#ifndef __NN_MACROS_H__
#define __NN_MACROS_H__

#define NNForbidOperation \
    do { \
        BOOST_LOG_TRIVIAL(fatal) << "Operation " <<  __FUNCTION__ << " Not Supported"; \
        BOOST_LOG_TRIVIAL(fatal) << "File " << __FILE__ << " line " << __LINE__; \
        std::abort();  \
    } while(false);

#define NNForbidOperationMsg(msg) \
    do {\
        BOOST_LOG_TRIVIAL(fatal) << msg; \
        BOOST_LOG_TRIVIAL(fatal) << "code at " << __FUNCTION__ <<" happended"; \
        BOOST_LOG_TRIVIAL(fatal) << "File " << __FILE__ << " line " << __LINE__; \
        std::abort(); \
    } while (false);

#endif // !__NN_MACROS_H__
