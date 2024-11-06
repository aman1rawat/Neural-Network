#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"
#include "NN.h"

Matrix* get_loss_gradient(NeuralNet *network, Matrix *output);

#endif