#define BOOST_TEST_MODULE "utilstringop"
#include <boost/test/unit_test.hpp>
#include "util/stringop.h"

// Test Split Function
BOOST_AUTO_TEST_CASE(StringSplitOperation) {
	std::string test1 = "i'm going to find a test  case  u";
	std::vector<std::string> res;
	Util::Split(test1, res, " ", true);
	res.clear();
	Util::Split(test1, res, " ", false);
	res.clear();

	std::string test2 = "c \tb u\t\t\td";
	Util::Split(test2, res, "\t", false);
	res.clear();
	Util::Split(test2, res, "\t", true);
	res.clear();

	std::string test3 = "ax dy x\t\t\t";
	Util::Split(test3, res, "\t", false);
	res.clear();

	std::string test4 = "abcd e d xe";
	Util::Split(test4, res, "\t", false);

}