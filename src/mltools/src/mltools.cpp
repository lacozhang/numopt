#include "RunModel.h"
#include "util/util.h"
#include <iostream>

void PrintGeneralUsage() {
  std::cout
      << "General Options linearmodel.exe [ModelType] [Optimization Method]"
      << std::endl
      << "                [ModelType]           : linear, lccrf, smcrf"
      << std::endl
      << "                [Optimization Method] : sgd, pgd, cg, lbfgs, cd, bcd"
      << std::endl;
  std::cout << "For more information on specific model & optimization method "
            << std::endl
            << "Plase use follow command linearmodel.exe linear sgd -h"
            << std::endl;
}

int main(int argc, const char *argv[]) {

  if (argc < 3) {
    PrintGeneralUsage();
    return 0;
  }

  boost::program_options::options_description helpdesc;
  helpdesc.add_options()("help,h", "print help");
  auto vm = ParseArgs(argc, argv, helpdesc, true);

  std::string modelstr(argv[1]);
  std::string optimstr(argv[2]);

  ModelType model = parsemodel(modelstr.c_str());

  if (vm.count("help") || (model == ModelType::None) || (optimstr.empty())) {

    if ((model == ModelType::None) || (optimstr.empty())) {
      PrintGeneralUsage();
    } else {

      std::cout << "Model     : " << modelstr << std::endl;
      std::cout << "Optimizer : " << optimstr << std::endl;
      switch (model) {
      case ModelType::Linear: {
        boost::shared_ptr<BinaryLinearModel> linearmodel;
        linearmodel.reset(new BinaryLinearModel());
        LinearFactory factory;
        RunModelHelp<TrainDataType::kLibSVM, DenseVector, DataSamples,
                     LabelVector, SparseVector, DenseVector>(
            optimstr, linearmodel, factory);
      } break;
      case ModelType::LCCRF: {
        boost::shared_ptr<LccrfModel> lccrf;
        lccrf.reset(new LccrfModel());
        LccrfFactory factory;
        RunModelHelp<TrainDataType::kLCCRF, DenseVector, LccrfSamples,
                     LccrfLabels, SparseVector, DenseVector>(optimstr, lccrf,
                                                             factory);
      } break;
      case ModelType::SMCRF:
        break;
      case ModelType::None: {
        LOG(FATAL) << "ModelType " << modelstr << " do no exist!!!";
      } break;
      }
    }
    return 0;
  }

  switch (model) {
  case ModelType::Linear: {
    boost::shared_ptr<BinaryLinearModel> linearmodel;
    linearmodel.reset(new BinaryLinearModel());
    LinearFactory factory;
    RunModel<TrainDataType::kLibSVM, DenseVector, DataSamples, LabelVector,
             SparseVector, DenseVector>(argc, argv, optimstr, linearmodel,
                                        factory);
  } break;
  case ModelType::LCCRF: {
    boost::shared_ptr<LccrfModel> lccrf;
    lccrf.reset(new LccrfModel());
    LccrfFactory factory;
    RunModel<TrainDataType::kLCCRF, DenseVector, LccrfSamples, LccrfLabels,
             SparseVector, DenseVector>(argc, argv, optimstr, lccrf, factory);
  } break;
  case ModelType::SMCRF:
    break;
  default:
    break;
  }
  return 0;
}
