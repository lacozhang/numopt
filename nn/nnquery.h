
#ifndef __NN_QUERY_H__
#define __NN_QUERY_H__
#include <boost/shared_ptr.hpp>
#include <boost/log/trivial.hpp>
#include "../typedef.h"

namespace NNModel {
	class QueryFeature {
	public:
		QueryFeature(){}
		~QueryFeature(){}

		DataSamples& Feature(){
			return features_;
		}

	private:
		DataSamples features_;
	};

	class QueryLabel {
	public:
		QueryLabel(){}
		~QueryLabel(){}

		LabelVector& Label(){
			return labels_;
		}

	private:
		LabelVector labels_;
	};

	class NNQueryFeature {
	public:
		NNQueryFeature(){
			featdat_.clear();
		}
		~NNQueryFeature(){

		}

		std::vector<boost::shared_ptr<QueryFeature>>& Features(){
			return featdat_;
		}

		boost::shared_ptr<QueryFeature>& FeatureOfSample(int idx){
			if (idx >= featdat_.size()){
				BOOST_LOG_TRIVIAL(fatal) << "Access Sample Index out of bound";
			}
			return featdat_[idx];
		}

		size_t NumSamples(){
			return featdat_.size();
		}

	private:
		std::vector<boost::shared_ptr<QueryFeature>> featdat_;
	};

	class NNQueryLabel {
	public:
		NNQueryLabel(){
			labels_.clear();
		}
		~NNQueryLabel(){

		}

		std::vector<boost::shared_ptr<QueryLabel>>& Labels(){
			return labels_;
		}

		boost::shared_ptr<QueryLabel>& LabelOfSample(int idx){
			if (idx >= labels_.size()){
				BOOST_LOG_TRIVIAL(fatal) << "Access sample index out of bound";
			}
			return labels_[idx];
		}

		size_t NumSamples(){
			return labels_.size();
		}

	private:
		std::vector<boost::shared_ptr<QueryLabel>> labels_;
	};
}


#endif // !__NN_QUERY_H__
