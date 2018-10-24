#pragma once
#include <WOA_Lib.h>

class userClass : public WOA_Lib {

public:
	float* coord;

	userClass();
	void doInitialization();
	void generateSort(short* cityIndex, float* searchAgent, int startPoint);
	void generateSequencedValue(short* cityIndex);
	void doFitnessCheck(long chromosomeAmount);
	void randomSearchAgent(float* newChromosome);
};