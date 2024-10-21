#include "utilities.h"
#include "NN.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

NeuralNet* buildNetwork(int input_size){
    if(input_size<=0){
        printf("ERROR : invalid input size\n");
        return NULL;
    }
	NeuralNet* n = (NeuralNet*)malloc(sizeof(NeuralNet));
	n->input_size = input_size;
	n->layers = NULL;
	n->tail_layer = NULL;
	return n;
}

void addLayer(NeuralNet *net, int size, char *activation) {
    Layer *layer = (Layer*)malloc(sizeof(Layer));
    if (!layer) {
        printf("ERROR: Failed to allocate layer\n");
        return;
    }
    
    // Initialize the new layer
    layer->size = size;
    layer->activation = activation;
    layer->bias = createMatrix(layer->size, 1);
    layer->neuron = createMatrix(layer->size, 1);
    layer->next = NULL;
    layer->prev = NULL; 
    layer->activation_gradient_neuron = NULL;
    layer->pre_activation_neuron = NULL;
    layer->weight_gradient = NULL;
    layer->bias_gradient = NULL;
    layer->error = NULL;

    if (!net->layers) {
        layer->weight = createMatrix(layer->size, net->input_size);
        net->layers = layer;
    } else {
        layer->prev = net->tail_layer;  // Set previous layer
        layer->weight = createMatrix(layer->size, net->tail_layer->size);
        net->tail_layer->next = layer;  // Set next pointer of previous layer
    }
    
    net->tail_layer = layer;  // Update tail pointer
    
    initializeMatrix(layer->weight);
    fillMatrix(layer->bias, 0);
}


void popLayer(NeuralNet *net){
	if(!net->layers){
		printf("ERROR : No layers to pop\n");
		return;
	}
    if(net->layers == net->tail_layer){
        Layer *temp = net->layers;
        net->layers = net->tail_layer = NULL;
        //freeLayer(temp);
        //free(temp);
        return;
    }
    else{
        Layer *temp = net->tail_layer;
        net->tail_layer = temp->prev;       
        net->tail_layer->next = NULL;          
        //freeLayer(temp);
        //free(temp);
    }
}

void freeLayer(Layer *layer){
	freeMatrix(layer->weight);
	freeMatrix(layer->bias);
	freeMatrix(layer->neuron);
    freeMatrix(layer->pre_activation_neuron);
    freeMatrix(layer->weight_gradient);
	free(layer->activation);
	free(layer->next);
	free(layer->prev);
}

void trainNetwork(NeuralNet *net, Matrix *input, Matrix *output, double lr) {
    if(!net || !net->layers){
        printf("ERROR in training the network\n");
        return;
    }

    Layer *current_layer;
    //-----------------------------forward propagate--------------------------
    for(current_layer = net->layers; current_layer; current_layer=current_layer->next){
        if(current_layer==net->layers){
            current_layer->neuron = dot(current_layer->weight, input);
        }
        else{
            current_layer->neuron = dot(current_layer->weight, current_layer->prev->neuron);
        }
        current_layer->neuron = add(current_layer->neuron, current_layer->bias);
        current_layer->pre_activation_neuron = copyMatrix(current_layer->neuron);
        current_layer->activation_gradient_neuron = copyMatrix(current_layer->neuron);

        activationGradient(current_layer);
        activateLayer(current_layer); 
    }

    //-----------------------------backward propagate-------------------------
    for(current_layer=net->tail_layer; current_layer; current_layer=current_layer->prev){
        Matrix *previous_layer_transpose = (current_layer->prev)? transpose(current_layer->prev->neuron): transpose(input);

        //if currently at last layer
        if(current_layer == net->tail_layer){
            current_layer->error = subtract(output, current_layer->neuron);
            current_layer->weight_gradient = dot(current_layer->error, previous_layer_transpose);
            current_layer->weight_gradient = scale(current_layer->weight_gradient, lr);
            current_layer->bias_gradient = scale(current_layer->error, lr);

            current_layer->weight = subtract(current_layer->weight, current_layer->weight_gradient);
            current_layer->bias = subtract(current_layer->bias, current_layer->bias_gradient);
        }

        else{
            current_layer->error = dot(transpose(current_layer->next->weight), current_layer->next->error);
            current_layer->error = multiply(current_layer->error, current_layer->activation_gradient_neuron);
            current_layer->weight_gradient = dot(current_layer->error, previous_layer_transpose);
            current_layer->weight_gradient = scale(current_layer->weight_gradient, lr);
            current_layer->bias_gradient = scale(current_layer->error, lr);

            current_layer->weight = subtract(current_layer->weight, current_layer->weight_gradient);
            current_layer->bias = subtract(current_layer->bias, current_layer->bias_gradient);
        }
    }

}


void predict(NeuralNet *net, Matrix *input){
	if(!net || !net->layers){
        printf("ERROR in training the network\n");
        return;
    }
    for(Layer *current_layer = net->layers; current_layer; current_layer=current_layer->next){
        if(current_layer==net->layers){
            current_layer->neuron = dot(current_layer->weight, input);
        }
        else{
            current_layer->neuron = dot(current_layer->weight, current_layer->prev->neuron);
        }
        current_layer->neuron = add(current_layer->neuron, current_layer->bias);
        current_layer->pre_activation_neuron = copyMatrix(current_layer->neuron);
        current_layer->activation_gradient_neuron = copyMatrix(current_layer->neuron);

        activationGradient(current_layer);
        activateLayer(current_layer); 
    }
	printMatrix(net->tail_layer->neuron);
}

void activateLayer(Layer *layer){
    if(strcmp(layer->activation, "sigmoid")==0){
        layer->neuron = applySigmoid(layer->neuron);
        return;
    }
    else if(strcmp(layer->activation, "ReLU")==0){
        layer->neuron = applyReLU(layer->neuron);
        return;
    }
}

Matrix* applyReLU(Matrix *input) {
    Matrix *output = copyMatrix(input);
    for (int i = 0; i < output->row; i++) {
        for (int j = 0; j < output->col; j++) {
            output->val[i][j] = fmax(0, output->val[i][j]);
        }
    }
    return output;
}

Matrix* applySigmoid(Matrix *input) {   
    Matrix *output = copyMatrix(input);
    for (int i = 0; i < output->row; i++) {
        for (int j = 0; j < output->col; j++) {
            double x = output->val[i][j];
            output->val[i][j] = 1.0f / (1.0f + exp(-x));
        }
    }
    return output;
}

void activationGradient(Layer *layer){
    if(strcmp(layer->activation, "sigmoid")==0){
        layer->activation_gradient_neuron = gradientSigmoid(layer->neuron);
        return;
    }
    else if(strcmp(layer->activation, "ReLU")==0){
        layer->activation_gradient_neuron = gradientReLU(layer->neuron);
        return;
    }
}


Matrix* gradientReLU(Matrix *input) {
    Matrix *gradient = createMatrix(input->row, input->col);
    for (int i = 0; i < input->row; i++) {
        for (int j = 0; j < input->col; j++) {
            gradient->val[i][j] = input->val[i][j] > 0 ? 1 : 0; 
        }
    }
    return gradient;
}

Matrix* gradientSigmoid(Matrix *input) {
    Matrix *gradient = createMatrix(input->row, input->col);
    for (int i = 0; i < input->row; i++) {
        for (int j = 0; j < input->col; j++) {
            double sig = 1 / (1 + exp(-input->val[i][j]));
            gradient->val[i][j] = sig * (1 - sig); 
        }
    }
    return gradient;
}


void saveNetwork(NeuralNet *net, const char * net_name){
	//to be defined later
}
void loadNetwork(NeuralNet *net, const char *net_name){
	//to be defined later
}
void freeNetwork(NeuralNet *net){
	Layer *temp = net->tail_layer->prev;
	while(temp!=net->layers){
		freeLayer(temp->next);
		temp=temp->prev;
	}
	freeLayer(net->layers);
	free(net);
}