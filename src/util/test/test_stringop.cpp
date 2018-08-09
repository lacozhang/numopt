#include "util/stringop.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

// Test Split Function
TEST(util, stringsplitoperation) {
  std::string test1 = "i'm going to find a test  case  u";
  std::vector<std::string> res;
  Util::Split(test1, res, " ", true);
  EXPECT_EQ(res.size(), 8);
  EXPECT_EQ(res[3], "find");
  res.clear();
  Util::Split(test1, res, " ", false);
  res.clear();

  std::string test2 = "c \tb u\t\t\td";
  Util::Split(test2, res, "\t", false);
  EXPECT_EQ(res.size(), 5);
  res.clear();
  Util::Split(test2, res, "\t", true);
  EXPECT_EQ(res.size(), 3);
  res.clear();

  std::string test3 = "ax dy x\t\t\t";
  Util::Split(test3, res, "\t", false);
  EXPECT_EQ(res.size(), 4);
  Util::Split(test3, res, "\t", true);
  EXPECT_EQ(res.size(), 1);
  res.clear();

  std::string test4 = "abcd e d xe";
  Util::Split(test4, res, "\t", false);
  EXPECT_EQ(res.size(), 1);
  
  std::string test5 = "/test/file/path/a.cc";
  Util::Split(test5, res, "/", false);
  EXPECT_EQ(res.size(), 5);
  Util::Split(test5, res, "/", true);
  EXPECT_EQ(res.size(), 4);
}
