#pragma once
#ifndef __LOGISTIC_MODEL_H__
#define __LOGISTIC_MODEL_H__
#include "dataop/DataIterator.h"
#include "model/AbstractModel.h"
#include "util/lossfunc.h"
#include "util/parameter.h"
#include "util/typedef.h"
#include <boost/shared_ptr.hpp>

class BinaryLinearModel
: public AbstractModel<DenseVector, DataSamples, LabelVector, SparseVector,
DenseVector> {
    public:
    typedef AbstractModel<DenseVector, DataSamples, LabelVector, SparseVector,
    DenseVector>
    BaseModelType;
    
    BinaryLinearModel();
    ~BinaryLinearModel();
    
    void SetLoss(LossFunc loss);
    
    virtual void InitFromCmd(int argc, const char *argv[]) override;
    virtual void InitFromData(DataIterator &iterator) override;
    virtual void Init() override {}
    
    virtual DenseVector &GetParameters() const override { return *param_; }
    
    virtual DenseVector &GetParameters() override { return *param_; }
    
    size_t FeatureDimension() const override { return featdim_; }
    
    virtual bool LoadModel(std::string model) override;
    virtual bool SaveModel(std::string model) override;
    
    virtual double Learn(DataSamples &samples, LabelVector &labels,
                         SparseVector &grad) override;
    virtual double Learn(DataSamples &samples, LabelVector &labels,
                         DenseVector &grad) override;
    
    virtual void Inference(DataSamples &samples, LabelVector &labels) override;
    
    virtual double Evaluate(DataSamples &samples, LabelVector &labels,
                            std::string &summary) override;
    
    private:
    boost::shared_ptr<lossbase> loss_;
    boost::shared_ptr<DenseVector> param_;
    size_t featdim_;
    double bias_;
    
    static const char *kLossOption;
    static const char *kBiasOption;
};

#endif // __LOGISTIC_MODEL_H__
