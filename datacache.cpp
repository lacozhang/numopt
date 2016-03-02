#include <dlib/cmd_line_parser.h>
#include <dlib/logger.h>
#include <dlib/misc_api.h>
#include <climits>
#include <fstream>
#include <iostream>
#include "dataop.h"
#include "util.h"

dlib::logger dlog("datacache");

static char TempLineBuffer[UINT16_MAX] = {'\0'};

int main(int argc, char* argv[]) {
  dlog.set_level(dlib::LALL);
  dlog << dlib::LINFO << "Start Program " << argv[0];

  try {
    dlib::command_line_parser parser;
    parser.add_option("in", "input file name", 1);
    parser.add_option("out", "output file name", 1);
    parser.set_group_name("misc options");
    parser.add_option("h", "display help message");

    parser.parse(argc, argv);

    if (parser.option("h")) {
      parser.print_options();
      return 0;
    }

    std::string infile, outfile;
    if (parser.option("in")) {
      infile = parser.option("in").argument();
      dlog << dlib::LINFO << "Input file " << infile;
    } else {
      dlog << dlib::LERROR << "Input file is missing, must supply";
      parser.print_options();
      return -1;
    }

    if (parser.option("out")) {
      outfile = parser.option("out").argument();
      dlog << dlib::LINFO << "Output file " << outfile;
    } else {
      dlog << dlib::LERROR << "Output file is missing, must supply";
      parser.print_options();
      return -1;
    }

    std::ifstream src(infile.c_str(), std::ios_base::in);
    if (!src.is_open()) {
      dlog << dlib::LERROR << "Open file " << infile << " failed";
      return -1;
    }

    std::ofstream sink(outfile.c_str(),
                       std::ios_base::out | std::ios_base::binary);
    if (!sink.is_open()) {
      dlog << dlib::LERROR << "Create binary file " << outfile << " failed";
      return -1;
    }

    std::memset(TempLineBuffer, 0, sizeof(TempLineBuffer));

    int linecount = 0;
    src.getline(TempLineBuffer, sizeof(TempLineBuffer));
    std::vector<std::pair<size_t, double>> feats;
    int label = 0;
    while (src.good()) {
      parselibsvmline(TempLineBuffer, feats, label);
      ++linecount;
      // write label
      sink.write((const char*)&label, sizeof(label));
      int n = feats.size();
      sink.write((const char*)&n, sizeof(n));

      for (std::pair<size_t, double>& item : feats) {
        sink.write((const char*)&item.first, sizeof(size_t));
        sink.write((const char*)&item.second, sizeof(double));
      }

      src.getline(TempLineBuffer, sizeof(TempLineBuffer));

      if (linecount % 10000 == 0) {
        dlog << dlib::LINFO << "Line count " << linecount;
      }
    }
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}