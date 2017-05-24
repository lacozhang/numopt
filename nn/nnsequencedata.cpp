#include "nnsequencedata.h"

namespace NNModel {

	NNSequenceFeature::NNSequenceFeature() {
		features_.clear();
	}

	NNSequenceFeature::~NNSequenceFeature() {
	}


	void NNSequenceFeature::AppendSequenceFeature(boost::shared_ptr<SentenceFeature>& feat) {
		features_.push_back(feat);
	}

	void NNSequenceFeature::SetSequenceFeature(boost::shared_ptr<SentenceFeature>& feat, int index) {
		while (index >= features_.size()) {
			BOOST_LOG_TRIVIAL(info) << "increase the size of features";
			features_.resize(2 * features_.size());
		}

		features_[index] = feat;
	}

	boost::shared_ptr<SentenceFeature> NNSequenceFeature::GetSequenceFeature(int index) {
		if (index < features_.size()) {
			return features_[index];
		}
		BOOST_LOG_TRIVIAL(info) << "Access data index outside of array";
		return boost::shared_ptr<SentenceFeature>();
	}

	boost::shared_ptr<SentenceFeature> NNSequenceFeature::operator[](int index) {
		return features_[index];
	}

	void NNSequenceLabel::AppendSequenceLabel(boost::shared_ptr<SentenceLabel>& label) {
		labels_.push_back(label);
	}

	void NNSequenceLabel::SetSequenceLabel(boost::shared_ptr<SentenceLabel>& label, int idx) {
		while (idx >= labels_.size()) {
			BOOST_LOG_TRIVIAL(info) << "Increase the size of labels";
			labels_.resize(2 * labels_.size());
		}

		labels_[idx] = label;
	}


	boost::shared_ptr<SentenceLabel> NNSequenceLabel::GetSequenceLabel(int index) {
		if (index < labels_.size())
			return labels_[index];
		
		BOOST_LOG_TRIVIAL(info) << "Access data index outside of array";
		return boost::shared_ptr<SentenceLabel>();
	}


	boost::shared_ptr<SentenceLabel> NNSequenceLabel::operator[](int index) {
		return labels_[index];
	}
}

