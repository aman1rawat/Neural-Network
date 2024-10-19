#ifndef UTILITIES_H
#define UTILITIES_H

typedef struct{
	double ** val;
	int row, col;
}Matrix;

Matrix * createMatrix(int row, int col);
Matrix* copyMatrix(Matrix* m);
void initializeMatrix(Matrix *m);
void fillMatrix(Matrix *m, int n);
void freeMatrix(Matrix *m);
void printMatrix(Matrix* m);
Matrix* flattenMatrix(Matrix* m);
Matrix* multiply(Matrix *m1, Matrix *m2);
Matrix* add(Matrix *m1, Matrix *m2);
Matrix* subtract(Matrix *m1, Matrix *m2);
Matrix* dot(Matrix *m1, Matrix *m2);
Matrix* addScalar(Matrix* m, double n);
Matrix* transpose(Matrix *input);
Matrix* scale(Matrix* m, double scaleAmount);

#endif