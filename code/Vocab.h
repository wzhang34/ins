#ifndef HVOCAB_H_INCLUDED
#define HVOCAB_H_INCLUDED

#include <cstring>

#include "IO.h"
#include "entry.h"

using std::string;

class HE;

/**
// structure of storage (flat): k = 3, l = 2
// format: layer_id,number
//                  root (gabage, not used)
//     1,0           1,1            1,2
//  2,0 2,1 2,2  2,3 2,4 2,5    2,6 2,7 2,8
*/

class Vocab
{

public:

    float* vec; // number of float : k^0*d + k^1*d + k^2*d + ... + k^l*d
    int k; // branking factor
    int l; // number of layers
    int d; // dimension of feature
    int num_leaf; // number of leaf nodes
    int *sp; // starting position of each layer

private:
    int total_len; // total number of

public:

    Vocab(int k_assign, int l_assign, int dim);

    ~Vocab();



    /**
    // quantize v (size: (n+m)xd) using this vocabulary. quantization index is kept in out
    // v: (n+m) x d
    // out: n x l. keeps the hierarchical quantization of v against codebook of vec
    // int d for dim is not passed cuz size(v) = nxd should be consistent with voc->d
    // m: skip the first d columns of data in v, which is often [x y ori scale]
    */
    void quantize2hie(float* v, int* out, int n, int m);


    /**
    // quantize a set of (n) vector v into the leaf layer
    // v: n x d
    // out: n x 1
    // m: skip the first m column of v.
    */
    void quantize2leaf(float* v, unsigned int* out, int n, int m);

    void quantize2leaf(float* v, unsigned int* out, int n, int m, int ma);

    /**
    // be ware of the memory it allocated
    // from feature file to entry list
    */
    Entry* quantizeFile(string file, HE* he, int& len, int nt, int ma);

    /**
    // load/write the vocabulary 'vec' from/to disk
    */
    void loadFromDisk(std::string dir);
    void write2Disk(std::string file);

private:

    /**
    // quantize single vector 'v' (size: 1 x d) to 'idx'
    // v: 1 x d. vector
    // idx: 1 x l. quantization result of each layer
    */
    void quantize_once(float* v, int* idx);
};

#endif // HVOCAB_H_INCLUDED
