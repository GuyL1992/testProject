
void kmeansAlgorithm(double** observations, double** centroids, int n, int d,int k);
void acumulate_sum(double* a, double* b, int d);
int checkDiff(double** centroids, double** centroids_in_progress, int k, int d);
void findmatch(int* closest_index, double** centroids, double* vector,int k, int d);
double distance(double* a, double* b, int d);
int eps_distance(double* vec1, double* vec2, int d);



void printMatrix(double** matrix, int n, int d);
double** allocationMatrix(int n, int d);
void freeMatrix(double** matrix,int n);
void getDimentions(char arr[],int* n, int* d);
double** getObservationsFromInput(char* input, int n, int d);
void print_row_vector(double* vector, int n);
int checkNegativeZero(double value);
void printMatrix(double** matrix, int a, int b);
void multiplyMatrix(double** result, double** aMatrix, double** bMatrix, int n);
void copyMatrix(double** target, double** source,int n);
double* allocationVector(int n);

double calcWeight(double* a, double* b, int d);
void formWeightedMatrix(double** weightedMatrix,double** vectorsMatrix, int n, int d);

void formDegreeMatrix (double** degreeMatrix, double** weightedMatrix, int n, int isRegularMode);

void formLnormMatrix (double** lNormMatrix, double** weightedMatrix,double** degreeSqrtMatrix, int n);

void formRotaionAndAtagMatrix(double** P ,double** A, int n);

void formIdentityMatrix(double** V, int n);
double ** getTransposeMatrix(double** matrix, int n);
void getSimilarMatrix(double** A, double** P,double** temp, int n);
double getOff(double ** A, int n);
void jaccobiAlgorithm (double** V,double** A, int n);

int cmpfuncDouble (const void * a, const void * b);
int TheEigengapHeuristic(double* eigenValues, int lenOfArr);

void normelize_matrix(double** matrix, int n, int d);
int cmpfuncEignvalues (const void * a, const void * b);
void sortEigenVectors(double**T, double** eigenVectors, double* eingeValues, int n,int k);





double** getDataPoints(double** observations, int n, int d, int* k);
void wamProcess(double** observations, int n, int d);
void lnormProcess (double** observations, int n, int d);
void ddgProcess (double** observations, int n, int d);
void jacobiProcess(double** A, int n);
