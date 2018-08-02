/*
 * =====================================================================================
 *
 *       Filename:  test_message.cpp
 *
 *    Description:  test code for message related components
 *
 *        Version:  1.0
 *        Created:  07/15/2018 10:27:46
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "system/message.h"
#include "gtest/gtest.h"

TEST(Message, init) {
  mltools::Message msg;
  msg.ShortDebugString();
  msg.DebugString();
}
