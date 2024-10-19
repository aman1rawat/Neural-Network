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
    output[0]->val[0][0] = 1;
    output[1]->val[0][0] = 4;
    output[2]->val[0][0] = 9;
    output[3]->val[0][0] = 16;


    NeuralNet *net = buildNetwork(1, "regression");
    addLayer(net, 2, "sigmoid");
    addLayer(net, 1, "sigmoid");
    // if(!net->layers){
    //     printf("no layers\n");
    // }
    // else{
    //     Layer *t = net->layers;
    //     while(t){
    //         printf("Layer info :\n");
    //         printf("Layer size : %d\n", t->size);
    //         printf("Layer activation : %s\n", t->activation);
    //         printf("Layer weights :\n");
    //         printMatrix(t->weight);
    //         printMatrix(t->bias);
    //         printf("\n----------------------------------\n");
    //         t=t->next;
    //     }
    // }
    printf("Layers created\n");
    for(int i=0;i<1;i++){
        printf("\n-----------------iteration %d------------------\n", i);
        trainNetwork(net, input[i], output[i], lr);
        Layer *t = net->layers;
        LayerInfo(t);
    }

}

    
    
