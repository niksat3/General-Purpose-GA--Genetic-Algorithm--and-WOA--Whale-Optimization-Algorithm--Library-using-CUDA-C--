#pragma once
#include <GA_Lib.h>

class userClass : public GA_Lib<short> {

public:
	float* coord;

	userClass();
	void doInitialization();
	void doFitnessCheck(long chromosomeAmount);
	void randomChromosome(short* newChromosome);
};
