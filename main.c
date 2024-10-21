#include <stdio.h>
#include <stdlib.h>

#include "NN.h"
#include "utilities.h"

#define lr 0.01

void LayerInfo(Layer *t){
    while(t){
        printf("Layer weights :\n");
        printMatrix(t->weight);
        printf("Layer biases : \n");
        printMatrix(t->bias);
        printf("Layer pre activation : \n");
        printMatrix(t->pre_activation_neuron);
        printf("Layer post activation : \n");
        printMatrix(t->neuron);
        printf("\n-----------------------------------\n");
        t=t->next;
    }
}

int main() {
    Matrix **input = (Matrix**)malloc(4*sizeof(Matrix*));
    for(int i=0;i<4;i++){
        input[i] = createMatrix(1,1);
    }
    input[0]->val[0][0] = 1;
    input[1]->val[0][0] = 2;
    input[2]->val[0][0] = 3;
    input[3]->val[0][0] = 4;

    Matrix **output = (Matrix**)malloc(4*sizeof(Matrix*));
    for(int i=0;i<4;i++){
        output[i] = createMatrix(1,1);
    }
    output[0]->val[0][0] = 2;
    output[1]->val[0][0] = 4;
    output[2]->val[0][0] = 6;
    output[3]->val[0][0] = 8;


    NeuralNet *net = buildNetwork(1);
    addLayer(net, 3, "sigmoid");
    addLayer(net, 1, "sigmoid");
    printf("Layers created\n");
    for(int i=0;i<4;i++){
        printf("\n-----------------iteration %d------------------\n", i);
        trainNetwork(net, input[i%4], output[i%4], lr);
        Layer *t = net->layers;
        LayerInfo(t);
    }
    predict(net, input[0]);
    predict(net, input[1]);
    predict(net, input[2]);
    predict(net, input[3]);

}

    
    
