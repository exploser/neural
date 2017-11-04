#pragma once
#include <random>
#include <atomic>

namespace neural
{
struct Random
{
	static std::minstd_rand& Generator()
	{
		thread_local static std::minstd_rand generator(std::random_device{ } ());

		return generator;
	}

	static std::minstd_rand& Generator(int seed)
	{
		thread_local static std::minstd_rand generator(seed);

		return generator;
	}
};
}
