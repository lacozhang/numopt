#include "textconv.h"

namespace NNModel {

    TextConvLayer::TextConvLayer(double * parambase, double * gradbase, int numfilters, int windowsize, int convsize, int stride) :
        param_(parambase, numfilters, convsize*windowsize), grad_(gradbase, numfilters, convsize*windowsize) {
        numfilters_ = numfilters;
        windowsize_ = windowsize;
        convsize_ = convsize;
        stride_ = stride;
        output_ = boost::make_shared<RowRealMatrix>();
        inputgrad_ = boost::make_shared<RowRealMatrix>();
    }

    void TextConvLayer::Forward(const RowRealMatrix& input, boost::shared_ptr<RowRealMatrix>& output) {
        if (input.cols() != convsize_) {
            NNForbidOperationMsg("Current not support convolution size not equal to embedding size");
        }

        size_t inputrows = input.rows();
        size_t outputsize = outputrowsize(inputrows);
        output_->resize(outputsize, numfilters_);
        output_->setZero();
        size_t inidx = 0, outidx = 0;
        for (inidx = 0; inidx + windowsize_ <= inputrows; inidx += stride_, outidx++) {
            Eigen::Map<const Eigen::VectorXd> dat(input.middleRows(inidx, windowsize_).data(), windowsize_*convsize_, 1);
            for (int i = 0; i < numfilters_; ++i) {
                output_->coeffRef(outidx, i) = dat.dot(param_.row(i));
            }
        }

        if (inidx < inputrows) {
            size_t leftrows = inputrows - inidx;
            Eigen::Map<const Eigen::VectorXd> dat(input.middleRows(inidx, leftrows).data(), leftrows*convsize_, 1);
            for (int i = 0; i < numfilters_; ++i) {
                output_->coeffRef(outidx, i) = dat.dot(param_.row(i).head(leftrows*convsize_));
            }
        }
        output = output_;
    }

    void TextConvLayer::Backward(const RowRealMatrix& input, const boost::shared_ptr<RowRealMatrix>& gradin, boost::shared_ptr<RowRealMatrix>& gradout){
        inputgrad_->resizeLike(input);
        inputgrad_->setZero();
        if (!gradin) {
            NNForbidOperationMsg("gradient of output is empty");
        }
        if ((gradin->rows() != output_->rows()) || (gradin->cols() != output_->cols())) {
            NNForbidOperationMsg("gradient of output do not match size of output");
        }

        size_t inputrows = input.rows();
        size_t outputrows = outputrowsize(inputrows);
        size_t inidx = 0, outidx = 0;
        for (inidx = 0; inidx + windowsize_ <= inputrows; inidx += stride_, outidx++) {
            Eigen::Map<const Eigen::VectorXd> inputdat(input.middleRows(inidx, windowsize_).data(), windowsize_*convsize_, 1);
            Eigen::Map<Eigen::VectorXd> ingrad(inputgrad_->middleRows(inidx, windowsize_).data(), windowsize_*convsize_, 1);
            ingrad += gradin->row(outidx).asDiagonal() * param_;
            for (int i = 0; i < numfilters_; ++i) {
                if (gradin->coeff(outidx, i) == 0.0) continue;
                param_.row(i) += gradin->coeff(outidx, i) * 
            }
        }
    }
}