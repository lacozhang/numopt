#include<iostream>
#include "logisticmodel.h"
#include "dataop.h"
#include "util.h"

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
	sampleidx_ = std::min(sampleidx_ + batchsize_, trainsamples_->rows());
	return true;
}

double LogisticModel::lossval(DenseVector& param){
	double vals = 0;
	double hypout = 0;
	for (int i = 0; i < trainsamples_->rows(); ++i){
		hypout = trainsamples_->row(i).dot(param);
		vals += loss_.loss(hypout, trainlabels_->coeff(i));
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
		if (iterIdx >= trainsamples_->rows()){
			break;
		}

		hypouts_[i] = trainsamples_->row(iterIdx).dot(param);
		double gradweight = loss_.dloss(hypouts_[i], trainlabels_->coeff(iterIdx));

		for (DataSamples::InnerIterator iter(*trainsamples_, iterIdx); iter; ++iter){
			g.coeffRef(iter.col()) += gradweight * iter.value();
		}
	}
}

void LogisticModel::grad(DenseVector& param, SparseVector& g){
	std::map<int, double> updates;
	hypouts_.resize(batchsize_);
	for (int i = 0; i < batchsize_; ++i){

		int iterIdx = sampleidx_ + i;
		if (iterIdx >= trainsamples_->rows()){
			break;
		}

		hypouts_[i] = trainsamples_->row(iterIdx).dot(param);
		double gradweight = loss_.dloss(hypouts_[i], trainlabels_->coeff(iterIdx));

		for (DataSamples::InnerIterator iter(*trainsamples_, iterIdx); iter; ++iter){
			updates[iter.col()] += gradweight * iter.value();
		}
	}

	g.setZero();
	for (std::map<int, double>::iterator iter = updates.begin(); iter != updates.end(); ++iter){
		g.coeffRef(iter->first) = iter->second;
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