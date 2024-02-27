#include "arithmetic.h"

long long CalculateSumOfArrayElements(long long arr[], long long startIndex, long long endIndex) {
	long long sum = 0;
	for (long long i = startIndex; i <= endIndex; i++) {
		sum += arr[i];
	}
	return sum;
}