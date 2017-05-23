#include "nnsequencedata.h"

namespace NNModel {

	void NNSequenceFeature::AppendSequenceFeature(boost::shared_ptr<SentenceFeature>& feat){
		features_.push_back(feat);
	}

	void NNSequenceFeature::SetSequenceFeature(boost::shared_ptr<SentenceFeature>& feat, int index){
		if (features_.size() < index + 1){
			BOOST_LOG_TRIVIAL(info) << "increase the size of features";
			features_.resize(index + 1);
		}
	}
}

