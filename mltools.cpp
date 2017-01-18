#include <iostream>
#include "RunModel.h"
#include "util.h"

void PrintGeneralUsage() {
	std::cout << "General Options linearmodel.exe [ModelType] [Optimization Method]" << std::endl
		<< "                [ModelType]           : linear, lccrf, smcrf" << std::endl
		<< "                [Optimization Method] : sgd, pgd, cg, lbfgs, cd, bcd";
	std::cout << "For more information on specific model & optimization method " << std::endl
		<< "Plase use follow command linearmodel.exe linear sgd -h";
}

int main(int argc, const char *argv[]) {

	if (argc < 3) {
		PrintGeneralUsage();
		return 0;
	}

	boost::program_options::options_description helpdesc;
	helpdesc.add_options()
		("help,h", "print help");
	auto vm = ParseArgs(argc, argv, helpdesc, true);

	std::string modelstr(argv[1]);
	std::string optimstr(argv[2]);
	
	ModelType model = parsemodel(modelstr.c_str());
	OptMethod optim = parseopt(optimstr.c_str());

	if (vm.count("help") ||
		(model == ModelType::None) ||
		(optim == OptMethod::None)) {

		if ((model == ModelType::None) ||
			(optim == OptMethod::None)) {
			PrintGeneralUsage();
		}
		else {

			std::cout << "Model     : " << modelstr << std::endl;
			std::cout << "Optimizer : " << optimstr << std::endl;
			switch (model)
			{
			case ModelType::Linear:
			{
				boost::shared_ptr<BinaryLinearModel> linearmodel;
				linearmodel.reset(new BinaryLinearModel());
				RunModelHelp<TrainDataType::kLibSVM, DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>(optim, linearmodel);
			}
			break;
			case ModelType::LCCRF:
				break;
			case ModelType::SMCRF:
				break;
			case ModelType::None:
			{
				BOOST_LOG_TRIVIAL(fatal) << "ModelType " << modelstr << " do no exist!!!";
			}
			break;
			}
		}
		return 0;
	}

	switch (model)
	{
	case ModelType::Linear:
	{
		boost::shared_ptr<BinaryLinearModel> linearmodel;
		linearmodel.reset(new BinaryLinearModel());
		RunModel<TrainDataType::kLibSVM, DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>(argc, argv, optim, linearmodel);
	}
	break;
	case ModelType::LCCRF:
		break;
	case ModelType::SMCRF:
		break;
	default:
		break;
	}
	return 0;
}