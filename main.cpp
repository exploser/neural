#include <iostream>
#include "NeuralNetwork.hpp"

int main(int argc, char *argv[])
{
	const std::vector<double> groundTruth = {0.01, 0.99};

	neural::NeuralNetwork network({2, 2, 2});

	for (int i = 0; i < 100; ++i)
	{
		auto err = network.Train({0.05, 0.10}, groundTruth);
		std::cout << "Fwd pass " << i << ", total error " << err << std::endl;
	}

	std::getchar();
}
