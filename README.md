# Machine-Learning-Cpp

Machine-Learning-Cpp is an object-oriented neural-network written in C++.

The program leverages Microsoft C++ AMP to perform operations in a heterogenous environment.

The start-point for the program is found in source.cpp --> main(). Here is where the program will read the required input data and format it such that the nnet object can process it.

Once the data has been correctly read and input, it is stored in a struct which is then passed to a nnet obect (nnect.c). It is here where a vector of base classes are created which represent the neural net graph. Each object represents one layer in the graph. The vector is comprised of the base object Neuron, which must be overridden by one of its child classes, logistic, tanh, relu, etc.

Once the graph is ready, processing begins running the four main steps in nnet; Forward Propagation, Backward Propagation, Error Accumulation, and Weight Update. The entire process is run according to the number of epochs defined in source.cpp.

Each layer object relies on a small number of basic matrix operations to complete. These operations are all maintained in the static class nnet_math. nnet_math performs parallel processing on the GPU using Microsoft C++ AMP.
