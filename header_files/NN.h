#ifndef NN_H
#define NN_H

#include "matrix.h"

typedef struct Layer{
	int size; // no of neurons in the layer
	Matrix * weighted_sum;
	Matrix * output;

	Matrix * weight; // weights for the layer
	Matrix * bias; //biases for the layer
	struct Layer * next; // pointer to next layer (if any)
	struct Layer * prev; // pointer to previous layer (if any)	

	Matrix * weight_gradient;
	Matrix * bias_gradient;

	Matrix * delta;
	char * activation;
	
}Layer;

typedef struct{
	int input_size;
	char *loss_function;
	Layer * layers; // head pointer to the linked list of layers
	Layer * tail_layer; // pointer to the last layer in the network
}NeuralNet;

NeuralNet* buildNetwork(int input_size, char * loss_function);
void addLayer(NeuralNet *net, int size, char * activation);
void trainNetwork(NeuralNet *net, Matrix *input, Matrix *output, double lr);

#endif