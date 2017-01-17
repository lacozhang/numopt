#include "DataIterator.h"
#include "typedef.h"
#include "util.h"


template<class SampleType, class LabelType>
const char* DataIteratorBase<SampleType, LabelType>::kBaseBatchSizeOption = "batch";

template<class SampleType, class LabelType>
const char* DataIteratorBase<SampleType, LabelType>::kBaseRandomSeedOption = "seed";

template <class SampleType, class LabelType>
DataIteratorBase<SampleType, LabelType>::DataIteratorBase() : batchdesc_("Data iteration options")
{

	batchsize_ = 0;
	dataindex_ = 0;
	realbatchsize_ = 0;

	std::srand(seed_);
	datapointers_.clear();
	data_.reset();

	ConstructCmdOptions();
}

template<class SampleType, class LabelType>
void DataIteratorBase<SampleType, LabelType>::InitFromCmd(int argc, const char * argv[])
{
	auto vm = ParseArgs(argc, argv, batchdesc_, true);
	batchsize_ = vm[kBaseBatchSizeOption].as<int>();
	seed_ = vm[kBaseRandomSeedOption].as<int>();
	BOOST_LOG_TRIVIAL(info) << "Batch size  : " << batchsize_;
	BOOST_LOG_TRIVIAL(info) << "Random seed : " << seed_;
}

template <class SampleType, class LabelType>
void DataIteratorBase<SampleType, LabelType>::SetDataSet(boost::shared_ptr<IndexDataType> dataset)
{
	if (dataset.get() == nullptr) {
		BOOST_LOG_TRIVIAL(warning) << "dataset is null";
		valid_ = false;
		return;
	}
	data_ = dataset;
	if (data_.get() != nullptr) {
		if (batchsize_ == -1) {
			realbatchsize_ = data_->SampleSize();
		}
		else {
			realbatchsize_ = batchsize_;
		}

		for (size_t i = 0; i < data_->SampleSize(); ++i) {
			datapointers_.push_back(i);
		}
	}
	valid_ = true;
}

template <class SampleType, class LabelType>
void DataIteratorBase<SampleType, LabelType>::ResetBatch()
{
	dataindex_ = 0;
	std::random_shuffle(datapointers_.begin(), datapointers_.end());
}

template <class SampleType, class LabelType>
bool DataIteratorBase<SampleType, LabelType>::GetNextBatch(SampleType& databatch, LabelType& labelbatch)
{
	if (dataindex_ >= data_->SampleSize()) {
		return false;
	}

	size_t endindex = dataindex_ + realbatchsize_;
	if (endindex > data_->SampleSize()) {
		endindex = data_->SampleSize();
	}

	data_->ResizeFeature(databatch, endindex - dataindex_);
	data_->ResizeLabel(labelbatch, endindex - dataindex_);

	if (data_.get() == nullptr) {
		BOOST_LOG_TRIVIAL(warning) << "retrieve data failed, because data is empty";
		return false;
	}

	for (size_t startindex = dataindex_; dataindex_ < endindex; ++dataindex_) {
		data_->FeatureCopyAtIndex(databatch, dataindex_ - startindex, datapointers_[dataindex_]);
		data_->LabelCopyAtIndex(labelbatch, dataindex_ - startindex, datapointers_[dataindex_]);
	}
	return true;
}

template<class SampleType, class LabelType>
void DataIteratorBase<SampleType, LabelType>::ConstructCmdOptions()
{
	batchdesc_.add_options()
		(kBaseBatchSizeOption, boost::program_options::value<int>()->default_value(1), "batch size for iteration, will be ignored for some optimization methods")
		(kBaseRandomSeedOption, boost::program_options::value<int>()->default_value(0), "random seed for date pertubation");
}


template class DataIteratorBase<DataSamples, LabelVector>;