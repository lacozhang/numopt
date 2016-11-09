#include "DataIterator.h"

DataIterator::DataIterator(int batchsize, int seed)
{

	BOOST_ASSERT_MSG(batchsize > 0, "Batchsize can't be 0");
	batchsize_ = batchsize;
	seed_ = seed;
	dataindex_ = 0;
	realbatchsize_ = 0;

	std::srand(seed_);
	datapointers_.clear();
	dataset_.reset();
}

void DataIterator::SetDataSet(boost::shared_ptr<DataSamples>& data, boost::shared_ptr<LabelVector>& label)
{
	if (data.get() == nullptr) {
		BOOST_LOG_TRIVIAL(warning) << "dataset is null";
		valid_ = false;
		return;
	}
	dataset_ = data;
	if (data.get() != nullptr) {
		if (batchsize_ == -1) {
			realbatchsize_ = dataset_->rows();
		}
		else {
			realbatchsize_ = batchsize_;
		}

		for (size_t i = 0; i < dataset_->rows(); ++i) {
			datapointers_.push_back(i);
		}
	}

	if (label.get() == nullptr) {
		BOOST_LOG_TRIVIAL(warning) << "label is null";
		valid_ = false;
		return;
	}
	labels_ = label;

	valid_ = true;
}

void DataIterator::ResetBatch()
{
	dataindex_ = 0;
	std::random_shuffle(datapointers_.begin(), datapointers_.end());
}

bool DataIterator::GetNextBatch(DataSamples& databatch, LabelVector& labelbatch)
{
	if (dataindex_ >= dataset_->rows()) {
		return false;
	}

	size_t endindex = dataindex_ + realbatchsize_;
	if (endindex > dataset_->rows()) {
		endindex = dataset_->rows();
	}

	databatch.resize(endindex - dataindex_, dataset_->cols());
	labelbatch.resize(endindex - dataindex_);

	if (dataset_.get() == nullptr || labels_.get() == nullptr) {
		BOOST_LOG_TRIVIAL(warning) << "retrieve data failed, because data is empty";
		return false;
	}

	for (size_t startindex = dataindex_; dataindex_ < endindex; ++dataindex_) {
		databatch.row(dataindex_ - startindex) = dataset_->row(datapointers_[dataindex_]);
		labelbatch.coeffRef(dataindex_ - startindex) = labels_->coeff(datapointers_[dataindex_]);
	}
	return true;
}
