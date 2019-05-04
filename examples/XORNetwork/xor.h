// Automatically generated NN header using keras2enn.py
//
#if !defined(K2ENN_XOR_H)
#define K2ENN_XOR_H

#include <NeuralNetwork.h>

namespace XOR {

using namespace EasyNeuralNetworks;

// Neural network weights definition
TYPE arr_dense_1[] PROGMEM = {
-1.3915728330612183, 1.3914809226989746, -0.00014474787167273462, 
0.10878174751996994, 0.8022587299346924, -0.10883966088294983, 
-1.4388160705566406, 1.438902497291565, 3.120443579973653e-05, 
-1.3655648231506348, 0.806423008441925, 1.365424633026123, 
0.6220335364341736, 0.8619970679283142, -0.6222371459007263, 
};
tensor<TYPE> w_dense_1(ProgmemHelper<TYPE>(arr_dense_1), /* width= */ 3, /* height= */ 5, /* depth= */ 1);

TYPE arr_dense_2[] PROGMEM = {
2.0431265830993652, -0.32554858922958374, 1.8586311340332031, -2.1148834228515625, -1.195865511894226, 1.3320156335830688, 
};
tensor<TYPE> w_dense_2(ProgmemHelper<TYPE>(arr_dense_2), /* width= */ 6, /* height= */ 1, /* depth= */ 1);



// Neural network layers definition
InputLayer<TYPE> input(/* width */ 2);
auto act_dense_1 = ReLUActivation<TYPE>();
DenseLayer<TYPE> dense_1(input, /* out_width= */ 5, /* weights= */ w_dense_1, /* activation= */ act_dense_1);
auto act_dense_2 = TanhActivation<TYPE>();
DenseLayer<TYPE> dense_2(dense_1, /* out_width= */ 1, /* weights= */ w_dense_2, /* activation= */ act_dense_2);
// Neural network
NeuralNetwork<TYPE> nn(3, &input, &dense_1, &dense_2);

};

#endif
