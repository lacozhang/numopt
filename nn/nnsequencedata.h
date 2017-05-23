#pragma once

#ifndef __NNSEQUENCE_DATA_H__
#define __NNSEQUENCE_DATA_H__
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/log/trivial.hpp>
#include "../typedef.h"

namespace NNModel {

	class NNSequenceFeature;

	class SentenceFeature {
	public:
		SentenceFeature() {
		}

		~SentenceFeature() {

		}

		DataSamples& SparseBinaryFeature() {
			return sparsebinary_;
		}

		DataSamples::ConstRowXpr SparseBinaryFeature(int pos) const {
			return sparsebinary_.row(pos);
		}

		DataSamples& SparseFeature() {
			return sparse_;
		}

		DataSamples::ConstRowXpr SparseFeature(int pos) const {
			return sparse_.row(pos);
		}

		DenseMatrix& DenseFeature() {
			return dense_;
		}

		DenseMatrix::ConstRowXpr DenseFeature(int pos) const {
			return dense_.row(pos);
		}

	private:
		DataSamples sparsebinary_;
		DataSamples sparse_;
		DenseMatrix dense_;
	};

	class SentenceLabel {
	public:

		const static int UNKNOWN = -1;

		SentenceLabel() {
			labels_.resize(0);
		}

		explicit SentenceLabel(int size) {
			labels_.resize(size);
			for (int i = 0; i < labels_.size(); ++i) {
				labels_.coeffRef(i) = SentenceLabel::UNKNOWN;
			}
		}

		~SentenceLabel() {
		}

		int GetLabel(int idx) {
			return labels_[idx];
		}

		void SetLabel(int idx, int label) {
			labels_[idx] = label;
		}

		LabelVector& GetLabels() {
			return labels_;
		}

		void SetLabels(std::vector<int>& labels) {
			labels_.resize(labels.size());
			for (int i = 0; i < labels.size(); ++i)
				labels_.coeffRef(i) = labels[i];
		}

		void SetLabels(LabelVector& labels) {
			labels_ = labels;
		}

	private:
		LabelVector labels_;
	};

	class NNSequenceFeature {
	public:
		NNSequenceFeature();
		~NNSequenceFeature();

		void AppendSequenceFeature(boost::shared_ptr<SentenceFeature>& feat);
		void SetSequenceFeature(boost::shared_ptr<SentenceFeature>& feat, int idx);
		SentenceFeature& GetSequenceFeature(int index);
		SentenceFeature& operator[](int index);
		size_t NumSamples();

	private:
		std::vector<boost::shared_ptr<SentenceFeature>> features_;
	};

	class NNSequenceLabel {
	public:
		NNSequenceLabel();
		~NNSequenceLabel();

		void AppendSequenceLabel(boost::shared_ptr<SentenceLabel>& label);
		void SetSequenceLabel(boost::shared_ptr<SentenceLabel>& label, int idx);
		SentenceLabel& GetSequenceLabel(int index);
		SentenceLabel& operator[](int index);
		size_t NumSamples();

	private:
		std::vector<boost::shared_ptr<SentenceLabel>> labels_;
	};

}

#endif // !__NNSEQUENCE_DATA_H__
