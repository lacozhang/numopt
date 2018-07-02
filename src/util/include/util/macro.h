/*
 * =====================================================================================
 *
 *       Filename:  macro.h
 *
 *    Description:  some macros
 *
 *        Version:  1.0
 *        Created:  07/01/2018 21:01:09
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#ifndef __MACROS_H__
#define __MACROS_H__

namespace mltools {

#define DISALLOW_COPY_AND_ASSIGN(TypeName)                                     \
  TypeName(const TypeName &);                                                  \
  TypeName &operator=(const TypeName &);

#define SINGLETON(TypeName)                                                    \
  static TypeName &getInstance() {                                             \
    static TypeName instance;                                                  \
    return instance;                                                           \
  }

} // namespace mltools

#endif // __MACROS_H__
