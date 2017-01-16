#include <iostream>
#include "RunModel.h"

void PrintGeneralUsage() {
	std::cout << "General Options linearmodel.exe [ModelType] [Optimization Method]"
		<< "                [ModelType]           : linear, lccrf, smcrf" << std::endl
		<< "                [Optimization Method] : sgd, pgd, cg, lbfgs, cd, bcd";
	std::cout << "For more information on specific model & optimization method "
		<< "Plase use follow command linearmodel.exe linear sgd -h";
}

int main(int argc, const char *argv[]) {

	boost::program_options::options_description desc("General command line options");
	boost::program_options::positional_options_description posdesc;
	posdesc.add("model", 1);
	posdesc.add("opt", 1);
	desc.add_options()
		("help,h", "display help message")
		("opt", boost::program_options::value<std::string>()->default_value(""), "optimization method want to use: sgd")
		("model", boost::program_options::value<std::string>()->default_value(""), "model type specification: linear,lccrf,smcrf");

	boost::program_options::variables_map vm;

	try
	{
		boost::program_options::store(
			boost::program_options::command_line_parser(argc, argv).options(desc).positional(posdesc).run(),
			vm);
		boost::program_options::notify(vm);
	}
	catch (boost::program_options::required_option& e)
	{
		std::cout << e.what() << std::endl;
		return 1;
	}
	catch (boost::program_options::error& e) {
		std::cout << e.what() << std::endl;
		return 1;
	}

	std::string modelstr, optimstr;
	if (vm.count("model")) {
		modelstr = vm["model"].as<std::string>();
	}
	else {
		modelstr = "";
	}
	optimstr = vm["opt"].as<std::string>();

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