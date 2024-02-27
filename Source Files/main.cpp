#include <iostream>
#include <omp.h>
#include "arithmetic.h"

int main()
{
	const int totalThreads = 4;
	const int vectorSize = 1000;
	const int rowSize = 96, columnSize = 96;
	omp_set_num_threads(totalThreads);
	double startTime, endTime;

//----------------Program to test time difference between parallel and sequential variant of sum of array elements-----------------
#pragma region sum_of_array_elements
	
	static const long long sArraySize = 100000;

	long long arr[sArraySize];
	for (long long i = 0; i < sArraySize; i++) {
		arr[i] = 1;
	}

	long long sumParallel = 0, sumSequential = 0;
	long long sectionSize = sArraySize / totalThreads;
	int threadId;

	startTime = omp_get_wtime();
	sumSequential = CalculateSumOfArrayElements(arr, 0, sArraySize - 1);
	endTime = omp_get_wtime();
	std::cout<<"In sequential variant sum is: "<< sumSequential<<" and time taken is: "<<(endTime - startTime)<<std::endl;


	startTime = omp_get_wtime();
	#pragma omp parallel default(shared) private(threadId)
	{
		long long startIndex, endIndex, tempSum = 0;
		threadId = omp_get_thread_num();
		startIndex = sectionSize * threadId;
		endIndex = sectionSize * (threadId + 1) - 1;
		if (endIndex >= sArraySize) endIndex = sArraySize - 1;
		
		#pragma omp critical
		sumParallel += CalculateSumOfArrayElements(arr, startIndex, endIndex);
	}
	endTime = omp_get_wtime();
	std::cout<<"In parallel variant-1 sum is: "<<sumParallel<<" and time taken is: "<<(endTime - startTime)<<std::endl;

	//Another parallel approach
	sumParallel = 0;
	startTime = omp_get_wtime();
	#pragma omp parallel default(shared) reduction(+: sumParallel)
	{
		#pragma omp for
		for (int i = 0; i < sArraySize; i++) {
			sumParallel += arr[i];
		}
	}
	endTime = omp_get_wtime();
	std::cout << "In parallel variant-2 sum is: " << sumParallel << " and time taken is: " << (endTime - startTime) << std::endl;

//------------------------------Result---------------------------------
//In sequential variant sum is : 100000 and time taken is : 0.0001812
//In parallel variant - 1 sum is : 100000 and time taken is : 0.0002678
//In parallel variant - 2 sum is : 100000 and time taken is : 8.16e-05
//----------------------------------------------------------------------

#pragma endregion
//---------------------------------------------------------------------------------------------------------------------------------

//----------------Program to test time difference between static and dynamic scheduling in executing dot product-------------------
#pragma region dot_product_of_vector
	/*
	int vec1[vectorSize];
	int vec2[vectorSize];
	for (int i = 0; i < vectorSize; i++) {
		vec1[i] = i;
		vec2[i] = i;
	}
	int dotProduct = 0;

	startTime = omp_get_wtime();
	#pragma omp parallel for schedule(static,100) reduction(+: dotProduct)
	for (int i = 0; i < vectorSize; i++) {
		dotProduct += vec1[i] * vec2[i];
	}
	endTime = omp_get_wtime();

	std::cout << "In static scheduling case dot product is: " << dotProduct << " and time taken is: " << (endTime - startTime) << std::endl;

	dotProduct = 0;
	startTime = omp_get_wtime();
	#pragma omp parallel for schedule(dynamic,100) reduction(+: dotProduct)
	for (int i = 0; i < vectorSize; i++) {
		dotProduct += vec1[i] * vec2[i];
	}
	endTime = omp_get_wtime();

	std::cout << "In dynamic scheduling case dot product is: " << dotProduct << " and time taken is: " << (endTime - startTime) << std::endl;
	
	dotProduct = 0;
	startTime = omp_get_wtime();
	for (int i = 0; i < vectorSize; i++) {
		dotProduct += vec1[i] * vec2[i];
	}
	endTime = omp_get_wtime();

	std::cout << "In sequential execution case dot product is: " << dotProduct << " and time taken is: " << (endTime - startTime) << std::endl;
	*/
//---------------------------------Result-----------------------------------------------
//In static scheduling case dot product is : 332833500 and time taken is : 0.0004011
//In dynamic scheduling case dot product is : 332833500 and time taken is : 6.79999e-06
//--------------------------------------------------------------------------------------

#pragma endregion
//---------------------------------------------------------------------------------------------------------------------------------

//------------Program to test time difference between sequential execution and dynamic scheduling in executing dot product----------
#pragma region matrix_vector_product
	/*
	int matrix[rowSize][columnSize];
	int vector[columnSize];
	int resultVectorParallel[columnSize] = { 0 };
	int resultVectorSequential[columnSize] = { 0 };
	for (int i = 0; i < rowSize; i++) {
		vector[i] = i;
		for (int j = 0; j < columnSize; j++) {
			matrix[i][j] = i + j;
		}
	}

	startTime = omp_get_wtime();
	#pragma omp parallel for schedule(dynamic,10) // 10 is the number of rows taken up by each thread
	for (int i = 0; i < rowSize;i++) {
		for (int j = 0;j < columnSize;j++) {
			resultVectorParallel[i] += matrix[i][j] * vector[j];
		}
	}
	endTime = omp_get_wtime();

	std::cout << "In parallel execution case the result vector is: ";
	for (int i = 0;i < columnSize;i++) {
		std::cout << resultVectorParallel[i] << " ";
	}
	std::cout << " and time taken for the operation is: " << (endTime - startTime) << std::endl;

	startTime = omp_get_wtime();
	for (int i = 0; i < rowSize;i++) {
		for (int j = 0;j < columnSize;j++) {
			resultVectorSequential[i] += matrix[i][j] * vector[j];
		}
	}
	endTime = omp_get_wtime();

	std::cout << "In sequential execution case the result vector is: ";
	for (int i = 0;i < columnSize;i++) {
		std::cout << resultVectorSequential[i] << " ";
	}
	std::cout << " and time taken for the operation is: " << (endTime - startTime) << std::endl;
	*/
#pragma endregion
//----------------------------------------------------------------------------------------------------------------------------------

//-----------------------------------Program to test various vzriants of matrix matrix multiplication-------------------------------
#pragma region matrix_matrix_multiplication

	int matrix1[rowSize][columnSize];
	int matrix2[rowSize][columnSize];
	int resultMatrix[rowSize][columnSize];
	for (int i = 0; i < rowSize; i++) {
		for (int j = 0; j < columnSize; j++) {
			matrix1[i][j] = i + j;
			matrix2[i][j] = i + j;
		}
	}

	for (int i = 0;i < rowSize;i++) {
		for (int j = 0;j < columnSize;j++) {
			resultMatrix[i][j] = 0;
		}
	}
	startTime = omp_get_wtime();
	#pragma omp parallel for schedule(dynamic, 8)
	for (int i = 0;i < rowSize;i++) {
		for (int j = 0;j < columnSize;j++) {
			for (int k = 0;k < columnSize;k++) {
				resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}
	endTime = omp_get_wtime();

	int sum = 0;
	for (int i = 0;i < rowSize;i++) {
		for (int j = 0;j < columnSize;j++) {
			sum += resultMatrix[i][j];
		}
	}
	std::cout << "In row wise parallelization case sum of all the entries are: " << sum << " and total time taken is : " << (endTime - startTime) << std::endl;

	for (int i = 0;i < rowSize;i++) {
		for (int j = 0;j < columnSize;j++) {
			resultMatrix[i][j] = 0;
		}
	}
	startTime = omp_get_wtime();
	#pragma omp parallel for collapse(2)
	for (int i = 0;i < rowSize;i++) {
		for (int j = 0;j < columnSize;j++) {
			for (int k = 0;k < columnSize;k++) {
				resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}
	endTime = omp_get_wtime();

	sum = 0;
	for (int i = 0;i < rowSize;i++) {
		for (int j = 0;j < columnSize;j++) {
			sum += resultMatrix[i][j];
		}
	}
	std::cout << "In the parallelization case using collapse(2) sum of all the entries are: " << sum << " and total time taken is: " << (endTime - startTime) << std::endl;

#pragma endregion
//----------------------------------------------------------------------------------------------------------------------------------

	return 0;
}