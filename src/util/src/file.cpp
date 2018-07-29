/*
 * =====================================================================================
 *
 *       Filename:  file.cpp
 *
 *    Description:  implementation of file.h interface
 *
 *        Version:  1.0
 *        Created:  07/07/2018 15:21:41
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#include "util/file.h"
#include "util/common.h"
#include "util/stringop.h"
#include <boost/signals2/detail/auto_buffer.hpp>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace mltools {

DECLARE_bool(verbose);

File *File::open(const std::string &filepath, const char *const flags) {
  File *f = nullptr;
  if (filepath == "stdin") {
    f = new File(stdin, filepath);
  } else if (filepath == "stdout") {
    f = new File(stdout, filepath);
  } else if (filepath == "stderr") {
    f = new File(stderr, filepath);
  } else if (gzfile(filepath)) {
    gzFile gz = gzopen(filepath.c_str(), flags);
    if (gz == NULL) {
      LOG(ERROR) << "Failed to open file " << filepath;
      return nullptr;
    }
    f = new File(gz, filepath);
  } else {
    FILE *cf = fopen(filepath.c_str(), flags);
    if (cf == NULL) {
      LOG(ERROR) << "Failed to open file " << filepath;
      return nullptr;
    }
    f = new File(cf, filepath);
  }
  return f;
}

File *File::openOrDie(const std::string &filepath, const char *const flags) {
  File *fileptr = open(filepath, flags);
  if (fileptr == NULL && fileptr->isOpen()) {
    LOG(FATAL) << "Failed to open " << filepath;
    std::exit(-1);
  }
  return fileptr;
}

File *File::open(const mltools::DataConfig &config, const char *const flags) {
  assert(config.file_size() == 1);
  auto filepath = config.file(0);
  if (config.has_hdfs()) {
    auto cmd = hadoopFS(config.hdfs()) + " -cat " + filepath;
    if (gzfile(filepath)) {
      cmd += " | gunzip";
    }

    auto despt = popen(cmd.c_str(), flags);
    if (despt == NULL) {
      LOG(ERROR) << "Failed to open " << config.DebugString();
      return nullptr;
    }
    auto f = new File(despt, filepath);
    return f;
  } else {
    return open(filepath, flags);
  }
}

File *File::openOrDie(const mltools::DataConfig &config,
                      const char *const flags) {
  auto fptr = open(config, flags);
  if ((fptr == NULL) || (!fptr->isOpen())) {
    LOG(FATAL) << "Failed to open file " << config.DebugString();
  }
  return fptr;
}

size_t File::size(const std::string &filepath) {
  if (gzfile(filepath)) {
    LOG(WARNING) << "Unable to get file size of compressed file";
    return 0;
  }
  struct stat fstat;
  stat(filepath.c_str(), &fstat);
  return fstat.st_size;
}

size_t File::size() const { return size(name_); }

bool File::flush() {
  return isgz_ ? (gzflush(gz_, Z_FINISH) == Z_OK) : (fflush(f_) == 0);
}

bool File::close() {
  bool succ = isgz_ ? (gzclose(gz_) == Z_OK) : (fclose(f_) == 0);
  gz_ = NULL;
  f_ = NULL;
  return succ;
}

size_t File::read(void *const buff, size_t size) {
  return isgz_ ? gzread(gz_, buff, size) : fread(buff, 1, size, f_);
}

size_t File::write(const void *const buff, size_t size) {
  return isgz_ ? gzwrite(gz_, buff, size) : fwrite(buff, 1, size, f_);
}

char *File::readLine(char *const output, uint64 maxLength) {
  return isgz_ ? gzgets(gz_, output, maxLength) : fgets(output, maxLength, f_);
}

bool File::seek(size_t pos) {
  return isgz_ ? gzseek(gz_, pos, SEEK_SET) == pos
               : fseek(f_, pos, SEEK_SET) == 0;
}

int64 File::readToString(std::string *const line, uint64 maxLength) {
  namespace detail = boost::signals2::detail;
  assert(line != nullptr);
  line->clear();
  if (maxLength <= 0) {
    return 0;
  }
  detail::auto_buffer<char, detail::store_n_objects<4096>> buffer;
  int64 readCnt = 0;
  while (maxLength > 0) {
    auto respReadCnt = read(buffer.data(), maxLength > 4096 ? 4096 : maxLength);
    if (respReadCnt == 0) {
      LOG(INFO) << "Failed to read specified bytes";
      break;
    }
    maxLength -= respReadCnt;
    readCnt += respReadCnt;
    line->append(buffer.data(), respReadCnt);
  }
  return static_cast<int64>(line->size());
}

size_t File::writeString(const std::string &line) {
  return write(line.c_str(), line.size());
}

File::~File() {
  if (f_ != NULL) {
    fclose(f_);
  }
  if (gz_ != NULL) {
    gzclose(gz_);
  }
}

bool readFileToString(const std::string &filepath, std::string *output) {
  File *file = File::open(filepath, "r");
  if (file == NULL) {
    return false;
  }
  auto fileSize = file->size();
  return (fileSize <= file->read(output, fileSize * 100));
}

bool writeStringToFile(const std::string &data, const std::string &filepath) {
  File *f = File::open(filepath, "w");
  if (f == NULL) {
    return false;
  }
  return (data.size() == f->writeString(data) && f->close());
}

namespace {
class NoOpErrorCollector : public google::protobuf::io::ErrorCollector {
public:
  virtual void AddError(int line, int column, const std::string &message) {}
};
} // namespace

bool readFileToProto(const std::string &filepath, GProto *proto) {
  DataConfig config;
  config.add_file(filepath);
  return readFileToProto(config, proto);
}

bool readFileToProto(const mltools::DataConfig &config, GProto *proto) {
  File *f = File::open(config, "r");
  if (f == NULL) {
    LOG(INFO) << "Failed to open file " << config.DebugString();
    return false;
  }
  size_t estSize = 2 << 20;
  if (!config.has_hdfs()) {
    estSize = f->size();
  }
  std::string content;
  f->readToString(&content, estSize * 100);
  NoOpErrorCollector noOpCollector;
  google::protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&noOpCollector);
  if (parser.ParseFromString(content, proto)) {
    return true;
  }
  if (proto->ParseFromString(content)) {
    return true;
  }

  google::protobuf::TextFormat::ParseFromString(content, proto);
  LOG(ERROR) << "Failed to parse the content " << config.DebugString();
  return false;
}

void readFileToProtoOrDie(const DataConfig &config, GProto *proto) {
  assert(readFileToProto(config, proto));
}

void readFileToProtoOrDie(const std::string &filepath, GProto *proto) {
  assert(readFileToProto(filepath, proto));
}

bool writeProtoToASCIIFile(const GProto &proto, const std::string &filepath) {
  std::string content;
  return google::protobuf::TextFormat::PrintToString(proto, &content) &&
         writeStringToFile(content, filepath);
}

void writeProtoToASCIIFileOrDie(const GProto &proto,
                                const std::string &filepath) {
  assert(writeProtoToASCIIFile(proto, filepath));
}

bool writeProtoToFile(const GProto &proto, const std::string &filepath) {
  std::string content;
  return proto.AppendToString(&content) && writeStringToFile(content, filepath);
}

void writeProtoToFileOrDie(const GProto &proto, const std::string &filepath) {
  assert(writeProtoToFile(proto, filepath));
}

std::string hadoopFS(const HDFSConfig &config) {
  string str = config.home() + "/bin/hadoop fs";
  if (config.has_namenode()) {
    str += " -D fs.default.name=" + config.namenode();
  }
  if (config.has_ugi()) {
    str += " -D hadoop.job.ugi=" + config.ugi();
  }
  return str;
}

std::vector<std::string> readFilenamesInDir(const std::string &dirpath) {
  std::vector<std::string> files;
  DIR *dir = opendir(dirpath.c_str());
  CHECK(dir != NULL) << " Failed to open directory " << dirpath;
  struct dirent *ent = nullptr;
  while ((ent = readdir(dir)) != NULL) {
    files.emplace_back(std::string(ent->d_name));
  }
  closedir(dir);
  return files;
}

std::vector<std::string> readFilenamesInDir(const DataConfig &config) {
  CHECK_EQ(config.file_size(), 1);
  return readFilenamesInDir(config.file(0));
}

std::string getFilename(const std::string &full) {
  std::vector<std::string> elems;
  Util::Split(full, elems, "/", true);
  return elems.empty() ? "" : elems.back();
}

std::string getPath(const std::string &full) {
  std::vector<std::string> elems;
  Util::Split(full, elems, "/", true);
  if (elems.size() <= 1) {
    return full;
  }
  elems.pop_back();
  return Util::join(elems, "/");
}

std::string removeExtension(const std::string &full) {
  std::vector<std::string> elems;
  Util::Split(full, elems, ".", false);
  if (elems.size() <= 1) {
    return full;
  }
  if (elems.size() > 2 && elems.back() == "gz") {
    elems.pop_back();
  }
  elems.pop_back();
  return Util::join(elems, ".");
}

bool dirExists(const std::string &path) {
  struct stat info;
  if (stat(path.c_str(), &info) != 0) {
    return false;
  }

  if (info.st_mode & S_IFDIR) {
    return true;
  }
  return true;
}

bool dirCreate(const std::string &path) {
  return mkdir(path.c_str(), 0755) == 0;
}
} // namespace mltools
