#include <cmath>
#include <vector>
#include <cstring>
#include <cassert>

#include "IO.h"
#include "Vocab.h"
#include "HE.h"
#include "MultiThd.h"


struct distance
{
    float val;
    int idx;
};

bool sort_by_val( const distance & lhs, const distance & rhs )
{
   return lhs.val < rhs.val;
}

class HE;

struct q_file_arg // quantization file args
{
    // in
    float* feat;
    Vocab* voc;
    HE* he;
    int d;
    int m;

    int ma;

    // out
    Entry* entrylist;
};

Vocab::Vocab(int k_assign, int l_assign, int dim)
{
    k = k_assign;
    l = l_assign;
    d = dim;
    num_leaf = pow(k, l);

    total_len = d * ((pow(k, l+1) -1) / (k-1)); // sum_i ( k^i*d )
    vec = new float[total_len];
    memset (vec, 0, sizeof(float)*total_len);

    // init the starting position of each layer
    sp = new int[l+1];
    sp[0] = 0;
    for(int i = 1; i < l+1; i++)
        sp[i] = sp[i-1] + pow(k, i-1)*d;
}

Vocab::~Vocab()
{
    delete[] vec;
    delete[] sp;
}


void Vocab::quantize2hie(float* v, int* out, int n, int m)
{
    for(int i = 0; i < n; i++)
        quantize_once(v + i*(d+m) + m, out + i*l);
}


void Vocab::quantize2leaf(float* v, unsigned int* out, int n, int m)
{
    memset(out, 0, sizeof(unsigned int)*n); // init out to 0
    int* out_hie = new int[l];
    for(int i = 0; i < n; i++)
    {
        quantize_once(v + i*(d+m) + m, out_hie);
        // turn hierarchical quantization result to flat quantization result
        for(int j = 0; j < l; j++)
            out[i] = out[i]*k + out_hie[j];
    }

    delete[] out_hie;
}


static void quanti_task(void* args, int tid, int i, pthread_mutex_t& mutex)
{
    q_file_arg* t = (q_file_arg*) args;

    unsigned int* out = new unsigned int[t->ma];
    int pos = i* (t->d + t->m);

    //t->voc->quantize2leaf(t->feat+pos, out, 1, t->m);
    t->voc->quantize2leaf(t->feat+pos, out, 1, t->m, t->ma);

    for(int m = 0; m < t->ma; m++)
    {
        int idx = i*t->ma + m;
        t->entrylist[idx].set(t->feat[pos], t->feat[pos+1], out[m],
                              t->feat[pos + 2], t->feat[pos + 3],
                              t->he->genCode(t->feat + pos, out[m], t->m));
    }

    delete[] out;

}

Entry* Vocab::quantizeFile(string file, HE* he, int& len, int nt, int ma)
{
    int n, m, d;
    float* feat = IO::readFeatFile(file, n, m, d);
    len = n*ma;

    Entry* entrylist = new Entry[n*ma];

    q_file_arg args = {feat, this, he, d, m, ma, entrylist};
    MultiThd::compute_tasks(n, nt, &quanti_task, &args);

    delete[] feat;
    return entrylist;
}




void Vocab::loadFromDisk(string dir)
{
    printf("Loading vocabulary: '%s'...\n", dir.c_str());
    int row, col;
    for(int i = 0; i < l; i++)
    {
        float* tmpmat = IO::loadFMat(dir + "vocab.l" + Util::num2str(i+1), row, col, -1);
        assert(row == pow(k, i+1) && col == d);
        memcpy(vec + sp[i+1], tmpmat, row*col*sizeof(float));
        delete[] tmpmat;
    }
}

void Vocab::write2Disk(string file)
{
    // note the first layer(layer-0) is not used, so any number is meaningless.
    // the reason why it's there: for consistent formulation
    for(int i = 1; i < l+1; i++) // when flatten to disk, l layers are included [1, l]
        IO::writeMat(vec + sp[i], (int)pow(k,i), d, file + "vocab.l" + Util::num2str(i));
}


// v: the vector to be quantized. size: d x 1
// idx: quantization result of each layer. size: l x 1
void Vocab::quantize_once(float* v, int* idx)
{
    // global_index keeps the quantized overall index starting from vec[0]. possible range: [0, d*((int)(pow(k, l+1)-1))/(k-1))
    // idx keeps the local quantized index. possible range: [0, k)
    //                  0
    //        1         2         3
    //     4  5  6   7  8  9   10 11 12
    // when global_index = 2, global_index * k + 1 gives the index of the child node
    int global_index = 0;
    for(int i = 0; i < l; i++) // each layer
    {
        global_index = global_index*k + 1; // goto the starting position for quantization in next layer
        float dist_best = 1e100;
        for(int j = 0; j < k; j++)
        {
            float dist = Util::dist_l2_sq(v, vec + (global_index + j)*d, d);
            if(dist < dist_best)
            {
                dist_best = dist;
                idx[i] = j;
            }
        }
        global_index += idx[i];
    }
}

void Vocab::quantize2leaf(float* v, unsigned int* out, int n, int m, int ma)
{
    memset(out, 0, sizeof(unsigned int)*n*ma); // init out to 0

    int* out_hie = new int[l-1];

    for(int p = 0; p < n; p++) // each point
    {
        int global_index = 0;
        for(int i = 0; i < l-1; i++) // each layer except the last layer
        {
            global_index = global_index*k + 1; // goto the starting position for quantization in next layer
            float dist_best = 1e100;
            for(int j = 0; j < k; j++)
            {
                float dist = Util::dist_l2_sq(v + p*(d+m) + m, vec + (global_index + j) * d, d);
                if(dist < dist_best)
                {
                    dist_best = dist;
                    out_hie[i] = j;
                }
            }
            global_index += out_hie[i];
        }
        global_index = global_index*k + 1;

        int sec_last_idx = 0;
        for(int i = 0; i < l-1; i++) // only quantized to l-1 layer
            sec_last_idx = sec_last_idx*k + out_hie[i];

        vector<distance> dists(k);
        for(int i = 0; i < k; i++)
        {
            dists[i].val = Util::dist_l2_sq(v + p*(d+m) + m, vec + (global_index + i)*d, d);
            dists[i].idx = i;
        }

        std::sort(dists.begin(), dists.end(), &sort_by_val);

        for(int i = 0; i < ma; i++)
        {
            out[p*ma + i] = sec_last_idx*k+ dists[i].idx;
        }
    }

    delete[] out_hie;
}
