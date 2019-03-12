// LSTM.cpp : this file contain "main" func
//

#include <iostream>
#include "nnlayer.h"
int main()
{
	paratype data_input[500]; //input data
	paratype pre_result[500];// inference result
	for (int i = 0; i < 500; i++) 
	{
		data_input[i] = 1;
	}
        for(int iter=0;iter<100;iter++) //for test
       {
	 std::cout << iter <<"\n";
	 Spmodel(data_input, pre_result);
       }

	std::cout << "calc done";
	return 0;
}
