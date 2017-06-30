#include <boost/log/trivial.hpp>
#include "cnn.h"
#include "../util.h"

namespace NNModel {
    const char* CNNModel::kEmbeddingSizeOption = "cnn.embed";
    const char* CNNModel::kHiddenSizeOption = "cnn.hidden";
    const char* CNNModel::kConvFilterSizeOptioin = "cnn.convcnt";
    const char* CNNModel::kConvWindowSizeOption = "cnn.convsize";
    const char* CNNModel::kPoolingSizeOption = "cnn.pool";
    const char* CNNModel::kDropOutOption = "cnn.dropout";


    CNNModel::CNNModel(){
        optionsdesc_.add_options()
            (kEmbeddingSizeOption, boost::program_options::value<int>(&embedsize_)->default_value(50), "embedding size")
            (kHiddenSizeOption, boost::program_options::value<int>(&hiddensize_)->default_value(100), "hidden size of cnn")
            (kConvFilterSizeOptioin, boost::program_options::value<int>(&convfilters_)->default_value(50), "number of filters of cnn")
            (kConvWindowSizeOption, boost::program_options::value<int>(&convsize_)->default_value(3), "number of words for convolution")
            (kPoolingSizeOption, boost::program_options::value<int>(&poolstack_)->default_value(3), "max strides for convolution operation")
            (kDropOutOption, boost::program_options::value<double>(&dropout_)->default_value(0.5), "drop out");

        vocabsie_ = labelsize_ = 0;
    }

    CNNModel::~CNNModel(){
    }

    void CNNModel::InitFromCmd(int argc, const char * argv[]) {
        auto vm = ParseArgs(argc, argv, optionsdesc_, true);
        BOOST_LOG_TRIVIAL(info) << "Embedding Size " << embedsize_;
        BOOST_LOG_TRIVIAL(info) << "Hidden Size    " << hiddensize_;
        BOOST_LOG_TRIVIAL(info) << "#Filters       " << convfilters_;
        BOOST_LOG_TRIVIAL(info) << "#Window Size   " << convsize_;
        BOOST_LOG_TRIVIAL(info) << "#Stack Size    " << poolstack_;
        BOOST_LOG_TRIVIAL(info) << "DropOut        " << dropout_;
    }

    void CNNModel::InitFromData(DataIterator& iterator){
        auto data = iterator.GetAllData();
        auto label = iterator.GetAllLabel();
        vocab_ = data.GetVocabulary();
        vocabsie_ = data.GetVocabularySize();
        label_ = label.GetVocabulary();
        labelsize_ = label.GetLabelSize();
    }

    void CNNModel::Init() {
        int modelsize = 0;
        // aligned memory by 16 bytes
        // embedding size
        modelsize += vocabsie_ * embedsize_;
        // convolution size
        modelsize += convfilters_ * embedsize_ * convsize_;
        // from pooling to hidden
        modelsize += poolstack_ * embedsize_ * hiddensize_;
        // from query to hidden
        modelsize += hiddensize_ * vocabsie_;
        // from hidden to output
        modelsize += hiddensize_ * labelsize_;
        // from query to output
        modelsize += labelsize_ * vocabsie_;

        param_ = boost::make_shared<RealVector>(modelsize);
        grad_ = boost::make_shared<RealVector>(modelsize);
        if (!param_ || !grad_) {
            BOOST_LOG_TRIVIAL(error) << "Allocate memory failed";
            std::abort();
        }
        param_->setZero();
        grad_->setZero();

        const double * parambae = param_->data(), *gradbase = grad_->data();
        embedlayer_ = boost::make_shared<EmbeddingLayer>(const_cast<double*>(parambae), const_cast<double*>(gradbase),
            vocabsie_, embedsize_);
    }

}
