#pragma once
#include <GA_Lib.h>

class userClass : public GA_Lib<int>
{
public:
	int* problem;
	userClass();
	void doInitialization();
	void setGenChangedOption(bool* genChangedOption);
	void doFitnessCheck(long chromosomeAmount);
	void randomChromosome(int* newChromosome);
};

