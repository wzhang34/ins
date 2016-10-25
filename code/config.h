#ifndef CONFIG_H_INCLUDED
#define CONFIG_H_INCLUDED

/** format of feature file: *.feat
//  n m d
//  row col scale ori d1 d2 d3 d4 ...
//  row col scale ori d1 d2 d3 d4 ...
//  ...  */

#include <string>


using std::string;

struct Config
{
    int             mode;               // which task are you in?
    int             verbose;            // level of verbosity

    string          dataId;             // specific id for the dataset working on
    int             num_ret;            // # of images returned by this retrieval system

    string          extn;               // extension of feature file
    int             dim;                // dimension of feature

    string          train_desc;         // directory of feature files for training vocab
    string          index_desc;         // directory of feature files of reference image
    string          query_desc;         // directory of feature files of query image

    // clustering
    float           num_per_file;       // sample factor of percentage of points sampled per feature file
    int             nt;                 // number of threads used
    int             bf;                 // branching factor
    int             num_layer;          // # of layers
    int             iter;               // # of iterations needed for kmeans
    int             T;                  // limit maximal number of sampling for kmeans
    int             attempts;           // # of attempts of clustering

    int             ma;                 // # of multiple assignment

    int             he_len;             // length of hamming code
    int             ht;                 // hamming distance threshold
    int             im_sz;              // max image size: im_sz x im_sz
    string          p_mat;              // location of projection matrix file

    int             search_mode;        // searching mode

    Config()
    {
        mode = 0;
        verbose = 0;
        dataId = "tmp_id";
        num_ret = 10;

        extn = ".feat";
        dim = 128;

        train_desc = "";
        index_desc = "";
        query_desc = "";

        ma = 4;

        nt = 1;
        attempts = 3;
        iter = 20;
        T = 10000;
        bf = 100;
        num_layer = 2;
        num_per_file = 0.01;

        ht = 12;
        p_mat = "pmat_32.mat";
    }
};



#endif // CONFIG_H_INCLUDED
