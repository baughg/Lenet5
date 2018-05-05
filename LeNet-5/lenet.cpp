#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <iostream>

std::ofstream debug_file("debug.txt");

char buffer_out[256];
#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight))){										\
(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];}	\
}

#define CONVOLUTE_VALID_TEST(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight))){										\
        sprintf(buffer_out,"(o0=%d,o1=%d,w0=%d,w1=%d) %lf * %lf",o0,o1,w0,w1, (input)[o0 + w0][o1 + w1], (weight)[w0][w1]);\
        debug_file << buffer_out << std::endl;\
(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
sprintf(buffer_out,"(o0=%d,o1=%d,w0=%d,w1=%d) (output)[o0][o1]=%lf",o0,o1,w0,w1,(output)[o0][o1]); debug_file << buffer_out << std::endl;}\
}

void convolute_full(LeNet5 *lenet, Feature *errors, int x,int y)
{
  // CONVOLUTE_FULL(input,             output,            weight)
  // CONVOLUTE_FULL(errors->layer5[y], errors->layer4[x], lenet->weight4_5[x][y])
  int s = GETLENGTH(errors->layer5[y]); // 1
  s = GETLENGTH(*(errors->layer5[y])); // 1
  s = GETLENGTH(lenet->weight4_5[x][y]); // 5
  s = GETLENGTH(*(lenet->weight4_5[x][y])); // 5

  for (int i0 = 0; i0 < GETLENGTH(errors->layer5[y]); ++i0) // 1
    for (int i1 = 0; i1 < GETLENGTH(*(errors->layer5[y])); ++i1) // 1
      for (int w0 = 0; w0 < GETLENGTH(lenet->weight4_5[x][y]); ++w0) // 5
        for (int w1 = 0; w1 < GETLENGTH(*(lenet->weight4_5[x][y])); ++w1) // 5
          (errors->layer4[x])[i0 + w0][i1 + w1] += (errors->layer5[y])[i0][i1] * (lenet->weight4_5[x][y])[w0][w1];
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}

#define CONVOLUTION_FORWARD_TEST(input,output,weight,bias,action)					\
{		sprintf(buffer_out,"GETLENGTH(weight)=%d, GETLENGTH(*weight)=%d",GETLENGTH(weight),GETLENGTH(*weight));\
    debug_file << buffer_out << std::endl;\
for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)	{						\
      sprintf(buffer_out,"x=%d,y=%d\n",x,y); debug_file << buffer_out << std::endl;\
CONVOLUTE_VALID_TEST(input[x], output[y], weight[x][y]);}					\
}

void convolution_forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
  //double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0],
    //double output[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1],
    //double weight[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL],
    //double bias[LAYER1], double(*action)(double)

  //convolution_forward(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
  const int output_size = GETLENGTH(features->layer1);
  const int output_height = GETLENGTH(*features->layer1);
  const int weight_width = GETLENGTH(lenet->weight0_1);
  const int weight_height = GETLENGTH(*lenet->weight0_1);

  for (int x = 0; x < weight_width; ++x)
    for (int y = 0; y < weight_height; ++y) {
      /*
      #define CONVOLUTE_VALID(input,output,weight)											\
      {																						\
        FOREACH(o0,GETLENGTH(output))														\
          FOREACH(o1,GETLENGTH(*(output)))												\
            FOREACH(w0,GETLENGTH(weight))												\
              FOREACH(w1,GETLENGTH(*(weight)))										\
                (output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
      }*/
      for (int o0 = 0; o0 < output_size; ++o0)
      {
        for (int o1 = 0; o1 < output_height; ++o1)
        {
          for (int w0 = 0; w0 < weight_width; ++w0)
          {
            for (int w1 = 0; w1 < weight_height; ++w1)
            {
              features->layer1[y][o0][o1] += features->input[x][o0 + w0][o1 + w1] * lenet->weight0_1[x][y][w0][w1];
            }
          }
        }
      }
      //CONVOLUTE_VALID(features->input[x], features->layer1[y], lenet->weight0_1[x][y]);
    }
  for (int j = 0; j < output_size; ++j)
    for (int i = 0; i < output_size; ++i)
      ((double *)features->layer1[j])[i] = action(((double *)features->layer1[j])[i] + lenet->bias0_1[j]);
}

void convolution_backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
  // CONVOLUTION_BACKWARD(input,            inerror,        outerror,       weight,           wd,                bd,              actiongrad)
  // CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
  int s = GETLENGTH(lenet->weight4_5);  // 16
  s = GETLENGTH(*lenet->weight4_5);     // 120
  s = GETCOUNT(errors->layer4);         // 400
  s = GETLENGTH(errors->layer5);        // 120
  s = GETCOUNT(errors->layer5[0]);      // 1

  for (int x = 0; x < GETLENGTH(lenet->weight4_5); ++x)     // 16
    for (int y = 0; y < GETLENGTH(*lenet->weight4_5); ++y)  // 120
      convolute_full(lenet, errors, x, y);
      //CONVOLUTE_FULL(errors->layer5[y], errors->layer4[x], lenet->weight4_5[x][y]);

  for (int i = 0; i < GETCOUNT(errors->layer4); ++i) // 400
    ((double *)errors->layer4)[i] *= actiongrad(((double *)features->layer4)[i]);

  for (int j = 0; j < GETLENGTH(errors->layer5); ++j)     // 120
    for (int i = 0; i < GETCOUNT(errors->layer5[j]); ++i) // 1
      deltas->bias4_5[j] += ((double *)errors->layer5[j])[i];

  for (int x = 0; x < GETLENGTH(lenet->weight4_5); ++x)     // 16
    for (int y = 0; y < GETLENGTH(*lenet->weight4_5); ++y)  // 120
      CONVOLUTE_VALID(features->layer4[x], deltas->weight4_5[x][y], errors->layer5[y]);
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}

void dot_product_backward(
  LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
  // DOT_PRODUCT_BACKWARD(input,            inerror,        outerror,       weight,           wd,                bd,              actiongrad)
  // DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
  int s = GETLENGTH(lenet->weight5_6);  // 120
  s = GETLENGTH(*lenet->weight5_6);     // 10
  s = GETCOUNT(errors->layer5);         // 120
  s = GETLENGTH(errors->output);        // 10

  for (int x = 0; x < GETLENGTH(lenet->weight5_6); ++x)     // 120
    for (int y = 0; y < GETLENGTH(*lenet->weight5_6); ++y)  // 10
      ((double *)errors->layer5)[x] += ((double *)errors->output)[y] * lenet->weight5_6[x][y];

  for (int i = 0; i < GETCOUNT(errors->layer5); ++i) // 120
    ((double *)errors->layer5)[i] *= actiongrad(((double *)features->layer5)[i]);

  for (int j = 0; j < GETLENGTH(errors->output); ++j) // 10
    deltas->bias5_6[j] += ((double *)errors->output)[j];

  for (int x = 0; x < GETLENGTH(lenet->weight5_6); ++x) // 120
    for (int y = 0; y < GETLENGTH(*lenet->weight5_6); ++y) // 10
      deltas->weight5_6[x][y] += ((double *)features->layer5)[x] * ((double *)errors->output)[y];
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}

double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
  const int output_size = GETLENGTH(features->layer1);
  CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
    
#ifdef SAVE_LAYER1
  FILE* image_file = NULL;
  image_file = fopen("layer1.bin", "wb");
  fwrite(features->layer1, sizeof(double), 6 * 28 * 28, image_file);
  fclose(image_file);
#endif
  //convolution_forward(lenet, features, action);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
#ifdef SAVE_LAYER2
  FILE* image_file = NULL;
  image_file = fopen("layer2.bin", "wb");
  fwrite(features->layer2, sizeof(double), 6 * 14 * 14, image_file);
  fclose(image_file);
#endif
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
#ifdef SAVE_LAYER3
  FILE* image_file = NULL;
  image_file = fopen("layer3.bin", "wb");
  fwrite(features->layer3, sizeof(double), 16 * 10 * 10, image_file);
  fclose(image_file);
#endif
  debug_file.close();
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
#ifdef SAVE_LAYER4
  FILE* image_file = NULL;
  image_file = fopen("layer4.bin", "wb");
  fwrite(features->layer4, sizeof(double), 16 * 5 * 5, image_file);
  fclose(image_file);
#endif
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
#ifdef SAVE_LAYER5
  FILE* image_file = NULL;
  image_file = fopen("layer5.bin", "wb");
  fwrite(features->layer5, sizeof(double), 120, image_file);
  fclose(image_file);
#endif
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
#ifdef SAVE_OUTPUT
  FILE* image_file = NULL;
  image_file = fopen("output.bin", "wb");
  fwrite(features->output, sizeof(double), 10, image_file);
  fclose(image_file);
#endif
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
  //dot_product_backward(lenet, deltas, errors, features, actiongrad);
  convolution_backward(lenet, deltas, errors, features, actiongrad);
	//CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

static inline void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
	double inner = 0;
	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}


void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward(lenet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(lenet, &deltas, &errors, &features, relugrad);
		#pragma omp critical
		{
			FOREACH(j, GETCOUNT(LeNet5))
				buffer[j] += ((double *)&deltas)[j];
		}
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += k * buffer[i];
}

void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	load_target(&features, &errors, label);
	backward(lenet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}

uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	return get_result(&features, count);
}

void Initial(LeNet5 *lenet)
{
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}
