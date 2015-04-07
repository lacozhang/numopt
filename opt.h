#ifndef __OPT_H__
#define __OPT_H__

// header file for optimization algorithm like GD, SGD, CG, LBFGS, Proximal SGD, GD

enum OptMethod {
	GD = 2,
	SGD,
	CG,
	LBFGS,
	PGD,
	CD, // coordinate descent
	BCD // block coordinate descent
};

#endif // __OPT_H__