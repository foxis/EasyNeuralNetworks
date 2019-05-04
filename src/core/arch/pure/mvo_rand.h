#if !defined(ENN_MVO_RAND_H)
#define ENN_MVO_RAND_H

#include <stdlib.h>
#include <math.h>
#include <limits>

namespace EasyNeuralNetworks {

inline float random_flat(float mean, float scale) {
	return scale * ((float).5 - (rand() / (float)RAND_MAX) + mean);
}

inline float random_normal(float mean, float stddev) {
	//Box muller method(stackoverflow https://stackoverflow.com/questions/19944111/creating-a-gaussian-random-generator-with-a-mean-and-standard-deviation)
	static float n2 = 0.0;
	static int n2_cached = 0;
	if (!n2_cached)
	{
			float x, y, r;
			do
			{
					x = 2.0 * rand() / (float)RAND_MAX - 1;
					y = 2.0 * rand() / (float)RAND_MAX - 1;

					r = x * x + y * y;
			}
			while (r == 0.0 || r > 1.0);
			{
					float d = sqrt(-2.0 * log(r) / r);
					float n1 = x * d;
					n2 = y * d;
					float result = n1 * stddev + mean;
					n2_cached = 1;
					return result;
			}
	}
	else
	{
			n2_cached = 0;
			return n2 * stddev + mean;
	}
}

};

#endif
