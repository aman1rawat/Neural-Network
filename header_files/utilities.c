#include "utilities.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

Matrix * createMatrix(int row, int col) {
    Matrix *matrix = (Matrix*)malloc(sizeof(Matrix));  // Allocate for one Matrix struct
    if (!matrix) return NULL;  // Check allocation

    matrix->row = row;
    matrix->col = col;
    
    matrix->val = (double**)malloc(row * sizeof(double*));
    if (!matrix->val) {
        free(matrix);
        return NULL;
    }
    
    for (int i = 0; i < row; i++) {
        matrix->val[i] = (double*)malloc(col * sizeof(double));
        if (!matrix->val[i]) {
            // Clean up previously allocated memory
            for (int j = 0; j < i; j++) {
                free(matrix->val[j]);
            }
            free(matrix->val);
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

Matrix* copyMatrix(Matrix* m) {
	Matrix* matrix = createMatrix(m->row, m->col);
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			matrix->val[i][j] = m->val[i][j];
		}
	}
	return matrix;
}

void initializeMatrix(Matrix *m) {
    float limit = sqrt(6.0 / (m->row + m->col));
    srand(time(0));
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->col; j++) {
            float rand_val = (float)rand() / RAND_MAX; 
            m->val[i][j] = (rand_val * 2 * limit) - limit; 
        }
    }
}

void fillMatrix(Matrix *m, int n){
	for(int i=0;i<m->row;i++){
		for(int j=0;j<m->col;j++){
			m->val[i][j] = (double)n;
		}
	}
}

void freeMatrix(Matrix *m){
	for(int i=0;i<m->row;i++){
		free(m->val[i]);
	}
	free(m->val);
	free(m);
}

void printMatrix(Matrix* m) {
    if (!m) {
        printf("NULL matrix\n");
        return;
    }
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->col; j++) {
            printf("%.6f ", m->val[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

Matrix* flattenMatrix(Matrix* m) {
    Matrix* matrix = createMatrix(m->row * m->col, 1);
    
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->col; j++) {
            // Assign values to the new column matrix
            matrix->val[i * m->col + j][0] = m->val[i][j];
        }
    }
    
    return matrix; 
}

Matrix* multiply(Matrix *m1, Matrix *m2){
	if((m1->row==m2->row) && (m1->col==m2->col)) {
		Matrix *m = createMatrix(m1->row, m1->col);
		for (int i = 0; i < m1->row; i++) {
			for (int j = 0; j < m2->col; j++) {
				m->val[i][j] = m1->val[i][j] * m2->val[i][j];
			}
		}
		return m;
	} 
	else {
		printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
		exit(1);
	}
}

Matrix* add(Matrix *m1, Matrix *m2){
	if((m1->row==m2->row) && (m1->col==m2->col)){
		Matrix *m = createMatrix(m1->row, m1->col);
		for (int i = 0; i < m1->row; i++) {
			for (int j = 0; j < m2->col; j++) {
				m->val[i][j] = m1->val[i][j] + m2->val[i][j];
			}
		}
		return m;
	} 
	else{
		printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
		exit(1);
	}
}

Matrix* subtract(Matrix *m1, Matrix *m2){
	if((m1->row==m2->row) && (m1->col==m2->col)){
		Matrix *m = createMatrix(m1->row, m1->col);
		for (int i = 0; i < m1->row; i++) {
			for (int j = 0; j < m2->col; j++) {
				m->val[i][j] = m1->val[i][j] - m2->val[i][j];
			}
		}
		return m;
	} 
	else{
		printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
		exit(1);
	}
}

Matrix* dot(Matrix *m1, Matrix *m2){
	if (m1->col == m2->row) {
		Matrix *m = createMatrix(m1->row, m2->col);
		for (int i = 0; i < m1->row; i++) {
			for (int j = 0; j < m2->col; j++) {
				double sum = 0;
				for (int k = 0; k < m2->row; k++) {
					sum += m1->val[i][k] * m2->val[k][j];
				}
				m->val[i][j] = sum;
			}
		}
		return m;
	}
	else{
		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
		exit(1);
	}
}

Matrix* addScalar(Matrix* m, double n) {
	Matrix* matrix = copyMatrix(m);
	for (int i = 0; i < matrix->row; i++) {
		for (int j = 0; j < matrix->col; j++) {
			matrix->val[i][j] += n;
		}
	}
	return matrix;
}

Matrix* transpose(Matrix *input) {
    Matrix *transposed = createMatrix(input->col, input->row);
    
    for (int i = 0; i < input->row; i++) {
        for (int j = 0; j < input->col; j++) {
            transposed->val[j][i] = input->val[i][j];
        }
    }
    return transposed;
}

Matrix* scale(Matrix* m, double scaleAmount){
	Matrix *mat = copyMatrix(m);
	for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->col; j++) {
            mat->val[i][j] *= scaleAmount;
        }
	}
	return mat;
}