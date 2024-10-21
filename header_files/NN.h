#ifndef NN_H
#define NN_H

#include "utilities.h"

typedef struct Layer{
	int size; // no of neurons in the layer
	Matrix * neuron;
	Matrix * pre_activation_neuron;
	Matrix * activation_gradient_neuron;
	Matrix * weight_gradient;
	Matrix * bias_gradient;
	Matrix * weight; // weights for the layer
	Matrix * bias; //biases for the layer
	Matrix * error;
	char * activation;
	struct Layer * next; // pointer to next layer (if any)
	struct Layer * prev; // pointer to previous layer (if any)
}Layer;

typedef struct{
	int input_size;
	char *task;
	Layer * layers; // head pointer to the linked list of layers
	Layer * tail_layer; // pointer to the last layer in the network
}NeuralNet;

NeuralNet* buildNetwork(int input_size);
void addLayer(NeuralNet *net, int size, char * activation);
void popLayer(NeuralNet *net);
void freeLayer(Layer *layer);
void activateLayer(Layer *layer);
void activationGradient(Layer *layer);

Matrix* applyReLU(Matrix *input);
Matrix* applySigmoid(Matrix *input);
Matrix* gradientReLU(Matrix *input);
Matrix* gradientSigmoid(Matrix *input);

void trainNetwork(NeuralNet *net, Matrix *input, Matrix *output, double lr);
void predict(NeuralNet *net, Matrix *input);
void saveNetwork(NeuralNet *net, const char * net_name);
void loadNetwork(NeuralNet *net, const char *net_name);
void freeNetwork(NeuralNet *net);

#endif