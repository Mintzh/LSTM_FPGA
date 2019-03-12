// LSTM.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "nnlayer.h"
int main()
{
	float data_input[500];
	float pre_result[500];
	for (int i = 0; i < 500; i++)
	{
		data_input[i] = 1;
	}
        for(int iter=0;iter<100;iter++)
       {
	std::cout << iter <<"\n";
	 Spmodel(data_input, pre_result);
       }
	std::cout << "calc done";
}
