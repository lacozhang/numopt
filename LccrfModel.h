#pragma once
#ifndef __LCCRF_MODEL_H__
#define __LCCRF_MODEL_H__

#include <unordered_set>
#include <boost/signals2/detail/auto_buffer.hpp>
#include "AbstractModel.h"

class LccrfModel : public AbstractModel<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector>
{
public:

	typedef AbstractModel<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector> BaseModelType;

	LccrfModel();
	~LccrfModel();

	virtual void InitFromCmd(int argc, const char* argv[]) override;
	virtual void InitFromData(DataIterator& iterator) override;

	virtual DenseVector& GetParameters() const override {
		return *param_;
	}

	virtual DenseVector& GetParameters() override {
		return *param_;
	}

	size_t FeatureDimension() const {
		return 0;
	}

	virtual bool LoadModel(std::string model) override;
	virtual bool SaveModel(std::string model) override;

	virtual void Learn(LccrfSamples& samples, LccrfLabels& labels, SparseVector& grad) override;
	virtual void Learn(LccrfSamples& samples, LccrfLabels& labels, DenseVector& grad) override;

	virtual void Inference(LccrfSamples& samples, LccrfLabels& labels) override;

	virtual void Evaluate(LccrfSamples& samples, LccrfLabels& labels, std::string& summary) override;

private:

	inline DenseVector::value_type GetUnigramWeight(int featid, int labelid) {
		if (featid < maxunigramid_) {
			int index = GetUnigramFeatureIndex(featid, labelid);
			return param_->coeff(index);
		}
		else
			return 0;
	}


	inline DenseVector::value_type GetBigramWeight(int featid, int fromlabelid, int tolabelid) {
		if (featid < maxbigramid_) {
			int index = GetBigramFeatureIndex(featid, fromlabelid, tolabelid);
			return param_->coeff(index);
		}
		else {
			return 0;
		}
	}


	inline int GetUnigramFeatureIndex(int featid, int labelid) {
		return featid * (maxlabelid_ + 1) + labelid;
	}

	inline int GetBigramFeatureIndex(int featid, int fromlabel, int tolabel) {
		return uniweightsize_ + fromlabel * (maxlabelid_ + 1) + tolabel;
	}

	void CalculateUnigramScore(DataSamples& features, DenseMatrix& val);
	void CalculateBigramScore(DataSamples& features, DenseMatrix& val);

	void ForwardPass(DenseMatrix& node, DenseMatrix& edge, int wordcount, DenseMatrix& alpha);
	void BackwardPass(DenseMatrix& node, DenseMatrix& edge, int wordcount, DenseMatrix& beta);
	void NodeProb(DenseMatrix& alpha, DenseMatrix& beta, int wordcount, DenseMatrix& nodeprob);
	void EdgeProb(DenseMatrix& alpha, DenseMatrix& beta, DenseMatrix& node, DenseMatrix& edge, int wordcount, DenseMatrix& edgeprob);
	void Viterbi1Best(DenseMatrix& node, DenseMatrix& edge, int wordcount, LabelVector& path);
	void GetNodeAndEdgeProb(DataSamples& unifeatues, DataSamples& bifeatures, DenseMatrix& nodeprob, DenseMatrix& edgeprob);

	double LogSumExp(Eigen::VectorXd& v);

	boost::shared_ptr<DenseVector> param_;

	// set data members for keys and hash table
	std::unordered_set<size_t> sparseunikeys_;
	std::unordered_set<size_t> sparsebikeys_;
	boost::shared_ptr<DenseVector> sparseparam_;
	std::vector<size_t> sparsekeys_;

	size_t maxunigramid_, maxbigramid_, maxlabelid_;
	size_t uniweightsize_;
	size_t modelsize_;
	int markovorder_;
	static const char* kOrderOption;
};

#endif // !__LCCRF_MODEL_H__
