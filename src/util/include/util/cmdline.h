
#ifndef __CMDLINE_H__
#define __CMDLINE_H__

#include "util/parameter.h"

bool cmd_line_parse(int argc, const char* argv[], Parameter& p);
LossFunc parselossfunc(const char *loss);
OptMethod parseopt(const char *opt);
ModelType parsemodel(const char* model);

#endif
