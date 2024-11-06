#include<stdio.h>
#include<stdlib.h>

#include "matrix.h"
#include "NN.h"
#include "activation.h"
#include "loss.h"

Matrix* get_loss_gradient(NeuralNet *network, Matrix *output){
	if(strcmp(network->loss_function,"MSE")==0){
		Matrix *loss_gradient = subtract(output, network->tail_layer->output);
		loss_gradient = scale(loss_gradient, -2);
		return loss_gradient;
	}
}