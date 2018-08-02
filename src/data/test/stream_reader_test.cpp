/*
 * =====================================================================================
 *
 *       Filename:  stream_reader_test.cpp
 *
 *    Description:  Test for stream reader
 *
 *        Version:  1.0
 *        Created:  07/08/2018 21:30:54
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "data/stream_reader.h"
#include "gtest/gtest.h"

TEST(streamreader, init) { mltools::StreamReader<float> test; }
