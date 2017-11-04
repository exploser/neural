#include "NeuralNet.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
	using namespace neural;
	Network n({2, 2, 2});
	n.SetInputs({0.05, 0.10});
	std::cout << n.Dump() << std::endl;
}
