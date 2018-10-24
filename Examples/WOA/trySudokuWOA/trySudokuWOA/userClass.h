#pragma once
#pragma once
#include <WOA_Lib.h>
#include "userFitness.h"

class userClass : public WOA_Lib {

public:
	float* problem;
	userClass();
	void doInitialization();
	void setGenChangedOption(bool* genChangedOption);
	void doFitnessCheck(long searchAgentAmount);
	void randomSearchAgent(float* randomSearchAgentValue);
};

