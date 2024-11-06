#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "matrix.h"
#include "NN.h"

Matrix* activate(Layer *layer);
Matrix* applySigmoid(Matrix *input);
Matrix* applyReLU(Matrix *input);

Matrix* activationGradient(Layer *layer);
Matrix* gradientSigmoid(Matrix *input);
Matrix* gradientReLU(Matrix *input);

#endif