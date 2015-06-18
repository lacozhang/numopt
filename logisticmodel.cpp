#include "logisticmodel.h"
#include "dataop.h"

LogisticModel::LogisticModel()
{
}

LogisticModel::~LogisticModel()
{
}

LogisticModel::LogisticModel(std::string dat){
	dataload(dat);
}

void LogisticModel::dataload(std::string dat){
	load_libsvm_data(dat, trainsamples_, trainlabels_);
}

void LogisticModel::startbatch(int batchsize){

	if (-1 == batchsize){
		epochbatch_ = 1;
		batchsize_ = trainsamples_->rows();
	}
	else {
		batchsize_ = batchsize;
		epochbatch_ = ((trainsamples_->rows() + batchsize_ - 1) / batchsize_);
	}

	sampleidx_ = 0;
	hypouts_.resize(batchsize_);
}

bool LogisticModel::nextbatch(){

	if (0 == epochbatch_){
		return false;
	}

	epochbatch_--;
	sampleidx_ += batchsize_;
	return true;
}

double LogisticModel::lossval(DenseVector& param){
	double vals = 0;
	hypouts_.clear();
	int start = sampleidx_;
	for (int i = 0; i < batchsize_; ++i){
		int iterIdx = start + i;
		hypouts_[i] = trainsamples_->row(iterIdx).dot(param);
		vals += loss_.loss(hypouts_[i], trainlabels_->coeff(iterIdx));
	}
	return vals;
}

double LogisticModel::funcval(DenseVector& param, SparseVector& sample){
	double hypout = sample.dot(param);
	return 1 / (1 + exp(-hypout));
}

void LogisticModel::grad(DenseVector& param, DenseVector& g){
	g.setZero();
	hypouts_.resize(batchsize_);
	for (int i = 0; i < batchsize_; ++i){
		int iterIdx = sampleidx_ + i;
		hypouts_[i] = trainsamples_->row(iterIdx).dot(param);
		double gradweight = loss_.dloss(hypouts_[i], trainlabels_->coeff(iterIdx));
		g += gradweight*trainsamples_->row(iterIdx).toDense();
	}
}

void LogisticModel::setparameter(DenseVector& param){
	param_ = param;
}

int LogisticModel::samplesize() const{
	return trainsamples_->rows();
}

int LogisticModel::featsize() const{
	return trainsamples_->cols();
}