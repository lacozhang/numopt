#include<iostream>
#include "linearmodel.h"
#include "dataop.h"
#include "util.h"

LinearModel::LinearModel(IOParameters& io, LossFunc loss)
{
	setio(io);
	setloss(loss);

	io_ = io;
	param_.reset(new DenseVector(featsize()));
}

LinearModel::~LinearModel()
{
}

void LinearModel::setio(IOParameters& io){
	loadtrain(io.train_);
	loadtest(io.test_);

	io_ = io;
	param_.reset(new DenseVector(featsize()));
}

void LinearModel::setloss(LossFunc loss){

	switch (loss){
	case LossFunc::Hinge:
		loss_.reset(new HingeLoss());
		break;
	case LossFunc::Logistic:
		loss_.reset(new LogLoss());
		break;
	case LossFunc::Squared:
		loss_.reset(new SquaredLoss());
		break;
	case LossFunc::SquaredHinge:
		loss_.reset(new SquaredHingeLoss());
		break;
	default:
		std::cerr << "Error" << std::endl;
	}
}


void LinearModel::loadtrain(std::string dat){
	load_libsvm_data(dat, trainsamples_, trainlabels_);
}

void LinearModel::loadtest(std::string dat){
	load_libsvm_data(dat, testsamples_, testlabels_);
}

void LinearModel::savemodel(std::string model){

}


DenseVector& LinearModel::param() const{
	return *param_;
}

void LinearModel::startbatch(int batchsize){

	if (-1 == batchsize){
		epochbatch_ = 1;
		batchsize_ = trainsamples_->rows();
	}
	else {
		batchsize_ = batchsize;
		epochbatch_ = ((trainsamples_->rows() + batchsize_ - 1) / batchsize_);
	}

	sampleidx_ = 0;
}

bool LinearModel::nextbatch(){

	if (0 == epochbatch_){
		return false;
	}

	epochbatch_--;
	sampleidx_ = std::min(sampleidx_ + batchsize_, trainsamples_->rows());
	return true;
}

double LinearModel::lossval(){
	double vals = 0;
	double hypout = 0;
	for (int i = 0; i < trainsamples_->rows(); ++i){
		hypout = trainsamples_->row(i).dot(*param_);
		vals += loss_->loss(hypout, trainlabels_->coeff(i));
	}
	return vals;
}

double LinearModel::funcval(SparseVector& sample){
	double hypout = sample.dot(*param_);
	return 1 / (1 + exp(-hypout));
}

void LinearModel::grad(DenseVector& g){
	double hypout;
	g.setZero();
	for (int i = 0; i < batchsize_; ++i){

		int iterIdx = sampleidx_ + i;
		if (iterIdx >= trainsamples_->rows()){
			break;
		}

		hypout = trainsamples_->row(iterIdx).dot(*param_);
		double gradweight = loss_->dloss(hypout, trainlabels_->coeff(iterIdx));

		for (DataSamples::InnerIterator iter(*trainsamples_, iterIdx); iter; ++iter){
			g.coeffRef(iter.col()) += gradweight * iter.value();
		}
	}
}

void LinearModel::grad(SparseVector& g){

	double hypout;
	std::map<int, double> updates;
	for (int i = 0; i < batchsize_; ++i){

		int iterIdx = sampleidx_ + i;
		if (iterIdx >= trainsamples_->rows()){
			break;
		}

		hypout = trainsamples_->row(iterIdx).dot(*param_);
		double gradweight = loss_->dloss(hypout, trainlabels_->coeff(iterIdx));

		for (DataSamples::InnerIterator iter(*trainsamples_, iterIdx); iter; ++iter){
			updates[iter.col()] += gradweight * iter.value();
		}
	}

	g.setZero();
	for (std::map<int, double>::iterator iter = updates.begin(); iter != updates.end(); ++iter){
		g.coeffRef(iter->first) = iter->second;
	}
}

void LinearModel::setparameter(DenseVector& param){
	*param_ = param;
}

int LinearModel::samplesize() const{
	return trainsamples_->rows();
}

int LinearModel::featsize() const{
	return trainsamples_->cols();
}