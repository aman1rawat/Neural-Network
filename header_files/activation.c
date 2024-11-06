#include<stdio.h>
#include<stdlib.h>

#include "matrix.h"
#include "NN.h"
#include "activation.h"
#include "loss.h"

Matrix* activate(Layer *layer){
    if(strcmp(layer->activation, "sigmoid")==0){
        return applySigmoid(layer->weighted_sum);
    }
    else if(strcmp(layer->activation, "ReLU")==0){
        return applyReLU(layer->weighted_sum);
    }
}

Matrix* applySigmoid(Matrix *weighted_sum) {   
    Matrix *output = copyMatrix(weighted_sum);
    for (int i = 0; i < output->row; i++) {
        for (int j = 0; j < output->col; j++) {
            double x = output->val[i][j];
            output->val[i][j] = 1.0f / (1.0f + exp(-x));
        }
    }
    return output;
}

Matrix* applyReLU(Matrix *weighted_sum) {
    Matrix *output = copyMatrix(weighted_sum);
    for (int i = 0; i < output->row; i++) {
        for (int j = 0; j < output->col; j++) {
            output->val[i][j] = fmax(0, output->val[i][j]);
        }
    }
    return output;
}


Matrix* activationGradient(Layer *layer){
    if(strcmp(layer->activation, "sigmoid")==0){
        return gradientSigmoid(layer->weighted_sum);
    }
    else if(strcmp(layer->activation, "ReLU")==0){
        return gradientReLU(layer->weighted_sum);
    }
}

Matrix* gradientReLU(Matrix *weighted_sum) {
    Matrix *gradient = createMatrix(weighted_sum->row, weighted_sum->col);
    for (int i = 0; i < weighted_sum->row; i++) {
        for (int j = 0; j < weighted_sum->col; j++) {
            gradient->val[i][j] = weighted_sum->val[i][j] > 0 ? 1 : 0; 
        }
    }
    return gradient;
}

Matrix* gradientSigmoid(Matrix *weighted_sum) {
    Matrix *gradient = createMatrix(weighted_sum->row, weighted_sum->col);
    for (int i = 0; i < weighted_sum->row; i++) {
        for (int j = 0; j < weighted_sum->col; j++) {
            double sig = 1 / (1 + exp(-weighted_sum->val[i][j]));
            gradient->val[i][j] = sig * (1 - sig); 
        }
    }
    return gradient;
}