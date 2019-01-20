#include "dataop/vocabulary.h"
#include <boost/signals2/detail/auto_buffer.hpp>

const std::string Vocabulary::kUNK = "<UNK>";
const std::string Vocabulary::kBeginOfDoc = "<s>";
const std::string Vocabulary::kEndOfDoc = "</s>";

boost::shared_ptr<Vocabulary>
Vocabulary::BuildVocabForQuery(const std::string &filepath, size_t cutoff,
                               bool hasbos, bool haseos) {
  namespace signal = boost::signals2::detail;
  boost::shared_ptr<Vocabulary> vocab;
  signal::auto_buffer<char, signal::store_n_objects<4 * 1024>> buffer(4 * 1024,
                                                                      '\0');

  cedar::da<int> rawvocab;
  std::ifstream src(filepath);
  if (!src.is_open()) {
    LOG(FATAL) << "Faled to open file " << filepath << std::endl;
    return vocab;
  }

  vocab = boost::make_shared<Vocabulary>();
  if (!vocab.get()) {
    LOG(FATAL) << "Failed to allocate memory";
    return vocab;
  }

  src.getline(buffer.data(), buffer.size());
  std::vector<std::string> words;
  while (src.good()) {
    if (hasbos)
      rawvocab.update(Vocabulary::kBeginOfDoc.c_str(),
                      Vocabulary::kBeginOfDoc.size(), 1);

    words.clear();
    Util::Split((const unsigned char *)buffer.data(),
                std::strlen(buffer.data()), words, (const unsigned char *)" ",
                true);
    for (auto &word : words) {
      rawvocab.update(word.c_str(), word.size(), 1);
    }

    if (haseos)
      rawvocab.update(Vocabulary::kEndOfDoc.c_str(),
                      Vocabulary::kEndOfDoc.size(), 1);

    src.getline(buffer.data(), buffer.size());
  }

  if (!src.eof()) {
    LOG(WARNING) << "Unexpected EOF";
  }

  vocab->AddWord(Vocabulary::kUNK, Vocabulary::kUNKID);
  vocab->AddWord(Vocabulary::kBeginOfDoc, Vocabulary::kBOSID);
  vocab->AddWord(Vocabulary::kEndOfDoc, Vocabulary::kEOSID);

  // iterator over vocabulary
  size_t from(0), p(0);
  int featcnt;
  for (featcnt = rawvocab.begin(from, p);
       featcnt != cedar::da<int>::CEDAR_NO_PATH;
       featcnt = rawvocab.next(from, p)) {
    if (featcnt >= cutoff) {
      rawvocab.suffix(buffer.data(), p, from);
      std::string tword(buffer.data());
      int key = vocab->VocabSize();
      if (!vocab->AddWord(tword, key)) {
        LOG(INFO) << "Add word failed" << std::endl;
      }
    }
  }

  return vocab;
}

boost::shared_ptr<Vocabulary>
Vocabulary::BuildVocabForLabel(const std::string &filepath) {
  namespace signal = boost::signals2::detail;
  signal::auto_buffer<char, signal::store_n_objects<1024>> buffer(1024, '\0');
  boost::shared_ptr<Vocabulary> vocab;
  std::ifstream src(filepath);
  if (!src.is_open()) {
    LOG(ERROR) << "Failed to open file " << filepath << std::endl;
    return vocab;
  }

  vocab = boost::make_shared<Vocabulary>();
  if (vocab.get() == nullptr) {
    LOG(INFO) << "Allocate vocabulary failed";
    return vocab;
  }

  std::vector<std::string> labels;
  cedar::da<int> rawlabels;
  src.getline(buffer.data(), buffer.size());
  int linenumber = 0;
  while (src.good()) {
    ++linenumber;
    rawlabels.clear();
    labels.clear();
    Util::Split((const unsigned char *)buffer.data(),
                std::strlen(buffer.data()), labels,
                (const unsigned char *)" \t", true);
    if (labels.empty() || labels.size() > 1) {
      LOG(WARNING) << "Multiple labels appeared " << buffer.data() << " @"
                   << linenumber;
    } else {
      rawlabels.update(labels[0].c_str(), labels[0].size());
    }
    src.getline(buffer.data(), buffer.size());
  }

  if (!src.eof()) {
    LOG(WARNING) << "Unexpected EOF";
  }

  vocab->AddWord(Vocabulary::kUNK, 0);
  // iterator over vocabulary
  size_t from(0), p(0);
  int featcnt;
  for (featcnt = rawlabels.begin(from, p);
       featcnt != cedar::da<int>::CEDAR_NO_PATH;
       featcnt = rawlabels.next(from, p)) {
    rawlabels.suffix(buffer.data(), p, from);
    std::string tword(buffer.data());
    int key = vocab->VocabSize();
    if (!vocab->AddWord(tword, key)) {
      LOG(INFO) << "Label " << tword << " appear multiple times";
    }
  }

  return vocab;
}
