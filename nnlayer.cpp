#include "nnlayer.h"
#include <stdio.h>
#include <cmath>
#define paratype float
const char* conv_1_w = "./wbin/W_in_1.bin";
const char* conv_1_b = "./wbin/b_in_1.bin";
const char* conv_2_w = "./wbin/W_in_2.bin";
const char* conv_2_b = "./wbin/b_in_2.bin";
const char* LSTM_0_k = "./wbin/RNNmulti_rnn_cellcell_0basic_lstm_cellkernel.bin";
const char* LSTM_0_b = "./wbin/RNNmulti_rnn_cellcell_0basic_lstm_cellbias.bin";
const char* LSTM_1_k = "./wbin/RNNmulti_rnn_cellcell_1basic_lstm_cellkernel.bin";
const char* LSTM_1_b = "./wbin/RNNmulti_rnn_cellcell_1basic_lstm_cellbias.bin";
const char* LSTM_2_k = "./wbin/RNNmulti_rnn_cellcell_2basic_lstm_cellkernel.bin";
const char* LSTM_2_b = "./wbin/RNNmulti_rnn_cellcell_2basic_lstm_cellbias.bin";
const char* LSTM_3_k = "./wbin/RNNmulti_rnn_cellcell_3basic_lstm_cellkernel.bin";
const char* LSTM_3_b = "./wbin/RNNmulti_rnn_cellcell_3basic_lstm_cellbias.bin";
const char* LSTM_4_k = "./wbin/RNNmulti_rnn_cellcell_4basic_lstm_cellkernel.bin";
const char* LSTM_4_b = "./wbin/RNNmulti_rnn_cellcell_4basic_lstm_cellbias.bin";
const char* MLP_1_w = "./wbin/softmax_w.bin";
const char* MLP_1_b = "./wbin/softmax_b.bin";
const char* MLP_2_w = "./wbin/softmax_w2.bin";
const char* MLP_2_b = "./wbin/softmax_b2.bin";

inline void conv1d(float data_input[input_length],float result[input_length], float kernel[conv1d_size], float bias) {
	for (int i = 0; i < input_length - 2; i++)
	{
	 result[i+1] = bias;
	 for (int j = 0; j < conv1d_size; j++) {
		 result[i+1] += data_input[i + j] * kernel[j];
	 }
	
	}
	//padding
	result[0] = bias;
	for (int j = 0; j < conv1d_size - 1; j++) {
		result[0] += data_input[j] * kernel[j+1];
	}

	result[input_length - 1] = bias;
	for (int j = 0; j < conv1d_size-1; j++) {
		result[input_length - 1] += data_input[input_length - 2 + j] * kernel[j];
	}
}

void MatMul(float* A, float* B, float* C,int row, int mulsize, int col) { 
	// C=A*B  A row*mulsize  B  mulsize*col
	for (int rr = 0; rr < row; rr++) {
		for (int cc = 0; cc < col; cc++) {
			C[rr * col + cc] = 0;
			for (int calc=0; calc < mulsize;calc++)
			{
				C[rr * col + cc] += A[rr * mulsize + calc] * B[calc * col + cc];
			}
		}
}
}

inline void MatMulAdd(float* A, float* B, float* C, int mulsize, int col,float* LSTMbias) {
	// C=A*B+bias  A row(1)*mulsize  B  mulsize*col
	for (int cc = 0; cc < col; cc++) {
		C[cc] = 0; 
	}
	for (int calc = 0; calc < mulsize; calc++) {
		for (int cc = 0; cc < col; cc++) {
			C[cc] += A[calc] * B[calc * col + cc];
		}
	}
	for (int cc = 0; cc < col; cc++) {
		C[cc] +=  LSTMbias[cc];
	}
}

void LSTM(float* input, float* state_c, float* state_h, float* LSTMkernel, float* LSTMbias,float* mulmid, float* concact){
// kernel 1000*2000
// concat --> MATMUL --> biasadd
	//Split
	float* i_gate = mulmid;
	float* a_gate = mulmid+500;
	float* f_gate = mulmid+1000;
	float* o_gate = mulmid+1500;
	//gate 1-4 i a f o from kernel
	for (int i = 0; i < 500; i++)
	{
		concact[i] = input[i];
	}
	for (int i = 0; i < 500; i++)
	{
		concact[i+500] = state_h[i];
	}
	MatMulAdd(concact, LSTMkernel, mulmid, 1000, 2000, LSTMbias);
	for (int i = 0; i < 500; i++)
	{
		f_gate[i] = 1 / (1 + expf(-f_gate[i]));  //sigmoid
	}
	for (int i = 0; i < 500; i++)
	{
		f_gate[i] = f_gate[i] * state_c[i];  // f^t * c^(t-1)
	}
	for (int i = 0; i < 500; i++)
	{
		i_gate[i] = 1/(1+expf(-i_gate[i]));  //sigmoid
	}
	for (int i = 0; i < 500; i++)
	{
		o_gate[i] = 1 / (1 + expf(-o_gate[i]));  //sigmoid
	}
	for (int i = 0; i < 500; i++)
	{
		a_gate[i] =tanhf(a_gate[i]);  //tanh
	}
	//ADD gate1 gate2
	for (int i = 0; i < 500; i++)
	{
		state_c[i] = f_gate[i] + i_gate[i] * a_gate[i];  //Add =¡· c^t
		if (state_c[i] > 3) //clip
		{
			state_c[i] = 3;
		}
	}
	for (int i = 0; i < 500; i++)
	{
		state_h[i] = tanhf(state_c[i]) * o_gate[i];  //o^t*Tanh
	}
}
void Spmodel(float* data_input,float* pre_result) {    
//make inference
	static int steps = 0;
	static float conv1d_1_kernel[3];
	static float conv1d_1_bias[1];
	static float conv1d_2_kernel[3];
	static float conv1d_2_bias[1];
	static float* conv_result_1;
	static float* conv_result_2;
	static float* state_c_0 ;
	static float* state_h_0 ;
	static float* state_c_1 ;
	static float* state_h_1 ;
	static float* state_c_2 ;
	static float* state_h_2 ;
	static float* state_c_3 ;
	static float* state_h_3 ;
	static float* state_c_4 ;
	static float* state_h_4 ;
	static float* LSTMkernel_0 ;
	static float* LSTMbias_0  ;
	static float* LSTMkernel_1;
	static float* LSTMbias_1  ;
	static float* LSTMkernel_2 ;
	static float* LSTMbias_2  ;
	static float* LSTMkernel_3 ;
	static float* LSTMbias_3  ;
	static float* LSTMkernel_4 ;
	static float* LSTMbias_4  ;
	static float* mlpweight_1 ;
	static float* mlpweight_2 ;
	static float* mlpbias_1 ;
	static float* mlpbias_2 ;
	static float* mlpresult_1 ;
	static float* mlpresult_2 ;
	static float* mulmid;
	static float* concact;
	static int firstread = 1;  //only read parameter once
       if(firstread)
       {
		  conv_result_1 = new float[input_length];
		  conv_result_2 = new float[input_length];
		  state_c_0 = new float[input_length];
		  state_h_0 = new float[input_length];
		  state_c_1 = new float[input_length];
		  state_h_1 = new float[input_length];
		  state_c_2 = new float[input_length];
		  state_h_2 = new float[input_length];
		  state_c_3 = new float[input_length];
		  state_h_3 = new float[input_length];
		  state_c_4 = new float[input_length];
		  state_h_4 = new float[input_length];
		  LSTMkernel_0 = new float[1000 * 2000];
		  LSTMbias_0 = new float[2000];
		  LSTMkernel_1 = new float[1000 * 2000];
		  LSTMbias_1 = new float[2000];
		  LSTMkernel_2 = new float[1000 * 2000];
		  LSTMbias_2 = new float[2000];
		  LSTMkernel_3 = new float[1000 * 2000];
		  LSTMbias_3 = new float[2000];
		  LSTMkernel_4 = new float[1000 * 2000];
		  LSTMbias_4 = new float[2000];
		  mlpweight_1 = new float[input_length * input_length];
		  mlpweight_2 = new float[input_length * input_length];
		  mlpbias_1 = new float[input_length];
		  mlpbias_2 = new float[input_length];
		  mlpresult_1 = new float[input_length];
		  mlpresult_2 = new float[input_length];
		  mulmid = new float[2000];
		  concact = new float[1000];

	FILE* fr = fopen(conv_1_w, "rb");
	fread(conv1d_1_kernel, sizeof(paratype), 3, fr);
	fclose(fr);

	fr = fopen(conv_1_b, "rb");
	fread(conv1d_1_bias, sizeof(paratype), 1, fr);
	fclose(fr);

	fr = fopen(conv_2_w, "rb");
	fread(conv1d_2_kernel, sizeof(paratype), 3 , fr);
	fclose(fr);

	fr = fopen(conv_2_b, "rb");
	fread(conv1d_2_bias, sizeof(paratype), 1, fr);
	fclose(fr);

	fr = fopen(conv_2_b, "rb");
	fread(conv1d_2_bias, sizeof(paratype), 1, fr);
	fclose(fr);

	fr = fopen(LSTM_0_k, "rb");
	fread(LSTMkernel_0, sizeof(paratype), 1000*2000, fr);
	fclose(fr);

	fr = fopen(LSTM_0_b, "rb");
	fread(LSTMbias_0, sizeof(paratype), 2000, fr);
	fclose(fr);

	fr = fopen(LSTM_1_k, "rb");
	fread(LSTMkernel_1, sizeof(paratype), 1000 * 2000, fr);
	fclose(fr);

	fr = fopen(LSTM_1_b, "rb");
	fread(LSTMbias_1, sizeof(paratype), 2000, fr);
	fclose(fr);

	fr = fopen(LSTM_2_k, "rb");
	fread(LSTMkernel_2, sizeof(paratype), 1000 * 2000, fr);
	fclose(fr);

	fr = fopen(LSTM_2_b, "rb");
	fread(LSTMbias_2, sizeof(paratype), 2000, fr);
	fclose(fr);

	fr = fopen(LSTM_3_k, "rb");
	fread(LSTMkernel_3, sizeof(paratype), 1000 * 2000, fr);
	fclose(fr);

	fr = fopen(LSTM_3_b, "rb");
	fread(LSTMbias_3, sizeof(paratype), 2000, fr);
	fclose(fr);

	fr = fopen(LSTM_4_k, "rb");
	fread(LSTMkernel_4, sizeof(paratype), 1000 * 2000, fr);
	fclose(fr);

	fr = fopen(LSTM_4_b, "rb");
	fread(LSTMbias_4, sizeof(paratype), 2000, fr);
	fclose(fr);

	fr = fopen(MLP_1_w, "rb");
	fread(mlpweight_1, sizeof(paratype), 500*500, fr);
	fclose(fr);

	fr = fopen(MLP_1_b, "rb");
	fread(mlpbias_1, sizeof(paratype), 500, fr);
	fclose(fr);

	fr = fopen(MLP_2_w, "rb");
	fread(mlpweight_2, sizeof(paratype), 500*500, fr);
	fclose(fr);

	fr = fopen(MLP_2_b, "rb");
	fread(mlpbias_2, sizeof(paratype), 500, fr);
	fclose(fr);
	
        firstread=0;
       }
	
  	//start calc
	//1dconv
	conv1d(data_input, conv_result_1, conv1d_1_kernel, *conv1d_1_bias);
	//doing elu
	for (int i = 0; i < input_length; i++)
	{
		if (conv_result_1[i] < 0)
		{
			conv_result_1[i] = expf(conv_result_1[i]) - 1;  
		}
	}
	conv1d(conv_result_1, conv_result_2, conv1d_2_kernel, *conv1d_2_bias);

	//LSTM
	//initialize LSTM state

	if (steps==0) //time step
	{
		for (int i = 0; i < input_length; i++)
		{
			state_c_0[i] = 0;
			state_h_0[i] = 0;
		}
		for (int i = 0; i < input_length; i++)
		{
			state_c_1[i] = 0;
			state_h_1[i] = 0;
		}
		for (int i = 0; i < input_length; i++)
		{
			state_c_2[i] = 0;
			state_h_2[i] = 0;
		}
		for (int i = 0; i < input_length; i++)
		{
			state_c_3[i] = 0;
			state_h_3[i] = 0;
		}
		for (int i = 0; i < input_length; i++)
		{
			state_c_4[i] = 0;
			state_h_4[i] = 0;
		}
	}
	steps++;
	if (steps == 9)
	{
		steps = 0;
	}
	//5 layers
	LSTM(conv_result_2, state_c_0, state_h_0,LSTMkernel_0, LSTMbias_0, mulmid,concact);
	
	LSTM(state_h_0, state_c_1, state_h_1 ,LSTMkernel_1, LSTMbias_1, mulmid, concact);

	LSTM(state_h_1, state_c_2, state_h_2, LSTMkernel_2, LSTMbias_2, mulmid, concact);

	LSTM(state_h_2, state_c_3, state_h_3, LSTMkernel_3, LSTMbias_3, mulmid, concact);

	LSTM(state_h_3, state_c_4, state_h_4, LSTMkernel_4, LSTMbias_4, mulmid, concact);

	/*
	printf("c:\n");
	for (int i = 0; i < 500; i++)
	{
		printf("%f,", state_c_4[i]);
	}
	printf("\n");
	printf("h:\n");
	for (int i = 0; i < 500; i++)
	{
		printf("%f,", state_h_4[i]);
	}
	printf("\n");
	*/

	//MLP
	//layer1
	for (int out = 0; out < input_length; out++)
	{
		mlpresult_1[out] = mlpbias_1[out];
	}
	for(int input_dim=0; input_dim < input_length; input_dim++)
	{
		for (int out = 0; out < input_length; out++)
		{
			mlpresult_1[out] += mlpweight_1[input_dim * input_length + out] * state_h_4[input_dim];
		}
		
	}
	for (int out = 0; out < input_length; out++)
	{
		if (mlpresult_1[out] < 0) //relu
		{
			mlpresult_1[out] = 0;
		}
	}
	//layer2
	for (int out = 0; out < input_length; out++)
	{
		mlpresult_2[out] = mlpbias_2[out];
	}
	for (int input_dim = 0; input_dim < input_length; input_dim++)
	{
		for (int out = 0; out < input_length; out++)
		{
			mlpresult_2[out] += mlpweight_2[input_dim * input_length + out] * mlpresult_1[input_dim];
		}

	}
	/*
	printf("result:\n");
	for (int i = 0; i < 500; i++)
	{
		printf("%f,", mlpresult_2[i]);
	}
	printf("\n");
	*/
}



