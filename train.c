#include "genann.h"

int main(int argc, char *argv[])
{

    /* Loading your training and test data. */
    double **training_data_input, **training_data_output, **test_data_input;

    

    /* New network with 2 inputs,
    * 2 hidden layers of 3 neurons each,
    * and 2 outputs. */
    /* 
        The number of hidden neurons should be between the size of the input layer and the size of the output layer.
        The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
        The number of hidden neurons should be less than twice the size of the input layer.
    */
    genann *ann = genann_init(2, 1, 3, 2);

    /* Learn on the training set. */
    for (i = 0; i < 300; ++i) {
        for (j = 0; j < 100; ++j)
            genann_train(ann, training_data_input[j], training_data_output[j], 0.1);
    }

    /* Run the network and see what it predicts. */
    double const *prediction = genann_run(ann, test_data_input[0]);
    printf("Output for the first test data point is: %f, %f\n", prediction[0], prediction[1]);

    genann_free(ann);
}