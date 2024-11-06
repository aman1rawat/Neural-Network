#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#include "utilities.h"
#include "NN.h"
#include "activation.h"
#include "loss.h"


Neuralnetwork* buildnetworkwork(int input_size, char * loss_function){
    if(input_size<=0){
        printf("ERROR : invalid input size\n");
        return NULL;
    }
	Neuralnetwork* networkwork = (Neuralnetwork*)malloc(sizeof(Neuralnetwork));
	networkwork->input_size = input_size;
	networkwork->layers = NULL;
	networkwork->tail_layer = NULL;
    networkwork->loss_function = loss_function;
	return networkwork;
}

void addLayer(Neuralnetwork *network, int size, char *activation) {
    Layer *layer = (Layer*)malloc(sizeof(Layer));
    if(!layer){
        printf("ERROR: Failed to allocate layer\n");
        return;
    }
    
    layer->size = size;
    layer->weighted_sum = NULL;
    layer->output = NULL;

    if(!network->layers){
        network->layers = network->tail_layer = layer;
        layer->prev = layer->next = NULL;
        layer->weight = createMatrix(layer->size, network->input_size);
    } 
    else{
        layer->prev = network->tail_layer; 
        network->tail_layer->next = layer;   
        layer->next = NULL;
        layer->weight = createMatrix(layer->size, layer->prev->size);
    }

    layer->weight_gradient = NULL;
    layer->bias_gradient = NULL;


    strcmp(layer->activation, activation);
    layer->delta = NULL;
    
    initializeMatrix(layer->weight);
    fillMatrix(layer->bias, 0);
}

void trainNetwork(Neuralnetwork *network, Matrix *input, Matrix *output, double lr) {
    if(!network || !network->layers){
        printf("ERROR in training the networkwork\n");
        return;
    }

    //-----------------------------------forward propagate----------------------------------------
    for(Layer *current_layer = network->layers; current_layer; current_layer=current_layer->next){
        if(current_layer==network->layers){
            current_layer->weighted_sum = dot(current_layer->weight, input);
        }
        else{
            current_layer->weighted_sum = dot(current_layer->weight, current_layer->prev->weighted_sum);
        }
        current_layer->weighted_sum = add(current_layer->weighted_sum, current_layer->bias);
        current_layer->output = activate(current_layer);
    }

    //------------------------------------backward propagate--------------------------------------
   
    //----------------------------------first loop for delta terms---------------------------------
    for(Layer *current_layer=network->tail_layer; current_layer; current_layer=current_layer->prev){
        if(current_layer == network->tail_layer){
            Matrix *loss_gradient = get_loss_gradient(network, output);
            Matrix *local_gradient = activationGradient(current_layer);
            current_layer->delta = multiply(loss_gradient, local_gradient);
            freeMatrix(loss_gradient);
            freeMatrix(local_gradient);
        }
        else{
            current_layer->delta = dot(transpose(current_layer->next->weight), current_layer->next->delta);
            Matrix *local_gradient = activationGradient(current_layer);
            current_layer->delta = multiply(current_layer->delta, local_gradient);
            freeMatrix(local_gradient);
        }
    }
    
    //----------------------------second loop for weight and bias gradient----------------------------
    for(Layer *current_layer=network->tail_layer; current_layer; current_layer=current_layer->prev){
        Matrix *prev_layer_transpose = (current_layer->prev)?transpose(current_layer->prev->output):transpose(input);
        current_layer->weight_gradient = dot(current_layer->delta, prev_layer_transpose);
        current_layer->weight_gradient = scale(current_layer->weight_gradient, lr);
        current_layer->bias_gradient = copyMatrix(current_layer->delta);
        current_layer->bias_gradient = scale(current_layer->bias_gradient, lr);
    }

    //----------------------------third loop to update weights and biases----------------------------
    for(Layer *current_layer=network->tail_layer; current_layer; current_layer=current_layer->prev){
        current_layer->weight = subtract(current_layer->weight, current_layer->weight_gradient);
        current_layer->bias = subtract(current_layer->bias, current_layer->bias_gradient);
    }
}