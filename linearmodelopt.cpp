#include <string>
#include "model.h"
#include "lossfunc.h"
#include "typedef.h"

class logreg : modelbase {
public:
	logreg(std::string train, std::string model);


private:
	LogLoss loss_;
	std::string traing_;
	std::string modelname_;

	
};