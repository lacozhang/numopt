/*
 * =====================================================================================
 *
 *       Filename:  test_van.cpp
 *
 *    Description:  test implementation of van
 *
 *        Version:  1.0
 *        Created:  07/15/2018 10:30:37
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "system/van.h"
using namespace mltools;

// follwing functions are faked for the testing purpose.

TEST(Van, init) {
  Van vanObj;
  vanObj.init();
}
