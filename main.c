#include <stdio.h>
#include <stdlib.h>

#include "header_files/NN.h"
#include "header_files/matrix.h"

#define lr 0.01




int main() {
	Matrix** input = (Matrix**)malloc(4*sizeof(Matrix*));
	Matrix** output = (Matrix**)malloc(4*sizeof(Matrix*));
	input[0] = createMatrix(1,1); input[0]->val[0][0] = 1;
	input[1] = createMatrix(1,1); input[1]->val[0][0] = 2;
	input[2] = createMatrix(1,1); input[2]->val[0][0] = 3;
	input[3] = createMatrix(1,1); input[3]->val[0][0] = 4;
	output[0] = createMatrix(1,1); output[0]->val[0][0] = 1;
	output[1] = createMatrix(1,1); output[1]->val[0][0] = 2;
	output[2] = createMatrix(1,1); output[2]->val[0][0] = 3;
	output[3] = createMatrix(1,1); output[3]->val[0][0] = 4;

	for(int i=0;i<4;i++){
		printf("Input : ");
		printMatrix(input[0]);
		printf("Output : ");
		printMatrix(output[0]);
	}    
}

    
    
