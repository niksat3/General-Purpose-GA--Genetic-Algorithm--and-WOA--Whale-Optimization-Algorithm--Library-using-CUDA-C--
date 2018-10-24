#include "userClass.h"
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>

userClass::userClass() : WOA_Lib((long)100, 38, 100) {
	coord = new float[getSize() * 2];
	run();
	float* leader = getLeader();
	short* cityIndex = new short[getSize()];
	generateSequencedValue(cityIndex);	
	generateSort(cityIndex, leader, 0);
	printf("Best Search Agent:\n");
	for (int i = 0; i < getSize(); i++) printf("%i ", cityIndex[i]);
	printf("\n");
	delete cityIndex;
}

void userClass::doInitialization() {
	float* newSearchAgent = this->getSearchAgent();

	//read file
	ifstream file("..\\..\\..\\..\\Resources\\TSP txt\\dj38.tsp.txt");
	string line;
	bool startRead = false;
	int i = 0;
	while (getline(file, line)) {
		stringstream linestream(line);
		if (startRead) {
			int temp;
			float c1, c2;
			linestream >> temp >> c1 >> c2;
			coord[i] = c1;
			coord[i + 1] = c2;
			i += 2;
		}
		if (line == "NODE_COORD_SECTION") startRead = true;
	}
	for (int i = 0; i < getNumberOfSearchAgent(); i++) {
		float* a = new float[getSize()];
		randomSearchAgent(a);
		for (int j = 0; j < getSize(); j++) {
			newSearchAgent[i * getSize() + j] = a[j];
		}
	}
	setSearchAgent(newSearchAgent);
}

void userClass::generateSort(short* cityIndex, float* searchAgent, int startPoint) {
	vector<pair<short, float>> vect;
	for (int i = 0; i < getSize(); i++) vect.push_back(make_pair(searchAgent[startPoint + i], cityIndex[i]));
	sort(vect.begin(), vect.end());
	for (int i = 0; i < getSize(); i++) cityIndex[i] = vect[i].second;
}

void userClass::generateSequencedValue(short* cityIndex) {
	for (short i = 0; i < getSize(); i++) cityIndex[i] = i;
}

void userClass::doFitnessCheck(long searchAgentAmount) {
	float* searchAgentTemp = this->getSearchAgent();
	short* cityIndex = new short[getSize()];
	float* fitnessTemp = this->getFitness();
	int size = getSize();
	for (int i = 0; i < searchAgentAmount; i++) {
		generateSequencedValue(cityIndex);
		generateSort(cityIndex, searchAgentTemp, i * size);
		float nilai = 0;
		for (int j = 1; j < getSize(); j++) {
			int xDist = coord[cityIndex[j] * 2] - coord[cityIndex[j - 1] * 2];
			int yDist = coord[(cityIndex[j] * 2) + 1] - coord[(cityIndex[(j - 1)] * 2) + 1];
			nilai += (sqrtf((xDist*xDist) + (yDist*yDist)));
		}
		int xDist = coord[cityIndex[0] * 2] - coord[cityIndex[(size - 1)] * 2];
		int yDist = coord[(cityIndex[0] * 2) + 1] - coord[(cityIndex[(size - 1)] * 2) + 1];
		nilai += (sqrtf((xDist*xDist) + (yDist*yDist)));
		fitnessTemp[i] = 99999.0f - nilai;
	}
	setFitness(fitnessTemp);
	delete cityIndex;
	//callFitnessCheckGPU(81, chromosomeTemp, fitnessTemp, chromosomeAmount);
};

void userClass::randomSearchAgent(float* newSearchAgent) {
	float* temp = new float[getSize()];
	for (int i = 0; i < getSize(); i++) temp[i] = i * 1.0f;
	for (int i = 0; i < 300; i++) {
		int index1 = randomInt(getSize() - 1), index2 = randomInt(getSize() - 1);
		int temp2 = temp[index1];
		temp[index1] = temp[index2];
		temp[index2] = temp2;
	}
	for (int i = 0; i < getSize(); i++) {
		newSearchAgent[i] = temp[i];
	}
	delete temp;
};
