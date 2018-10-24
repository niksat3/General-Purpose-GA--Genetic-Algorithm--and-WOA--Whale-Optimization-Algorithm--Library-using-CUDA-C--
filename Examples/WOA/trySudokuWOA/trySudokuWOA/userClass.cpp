#include "userClass.h"
#include <string>
#include <sstream>
#include <fstream>

userClass::userClass() : WOA_Lib((long)1000, 81, 40) {
	run();
	float* leader = this->getLeader();
	printf("Best Search Agent:\n");
	for (int i = 0; i < getSize(); i++) {
		if (i % 9 == 0) printf("\n");
		printf("%i ", (abs((int)leader[i]) % 9));
	}
	printf("\n");
}

void userClass::doInitialization() {
	problem = new float[getSize()];
	ifstream file;
	file.open("..\\..\\..\\..\\Resources\\Sudoku File\\Sudoku Easy.txt");
	int i = 0;
	while (i < getSize() && file >> problem[i]) i++;
	file.close();
	float* newSearchAgent = this->getSearchAgent();
	for (int i = 0; i < getNumberOfSearchAgent(); i++)
	{
		float* a = new float[getSize()];
		for (int j = 0; j < 81; j++) a[j] = problem[j];
		randomSearchAgent(a);
		for (int j = 0; j < 81; j++) newSearchAgent[(i * 81) + j] = a[j];
		delete a;
	}
	setSearchAgent(newSearchAgent);
	delete problem;
}

void userClass::setGenChangedOption(bool* genChangedOption) {
	for (int i = 0; i < 81; i++) {
		if (problem[i] == 0) genChangedOption[i] = true;
		else genChangedOption[i] = false;
	}
}

void userClass::doFitnessCheck(long searchAgentAmount) {
	float* searchAgentTemp = this->getSearchAgent();
	float* fitnessTemp = this->getFitness();
	callFitnessCheckGPU_SudokuWOA(81, searchAgentTemp, fitnessTemp, searchAgentAmount);
};

void userClass::randomSearchAgent(float* randomSearchAgentValue) {
	for (int i = 0; i < 9; i++)
	{
		float temp[] = { 1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f };
		for (int j = 0; j < 100; j++)
		{
			int index1 = rand() % 9;
			int index2 = rand() % 9;
			int temp2 = temp[index1];
			temp[index1] = temp[index2];
			temp[index2] = temp2;
		}
		for (int j = 0; j < 9; j++)
		{
			if (randomSearchAgentValue[i * 9 + j] == 0) randomSearchAgentValue[i * 9 + j] = temp[j];
		}
	}
};