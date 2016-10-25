#ifndef CLUSTERING_H_INCLUDED
#define CLUSTERING_H_INCLUDED

#include <cstring>
#include <pthread.h>
#include <ctime>
#include <cstdio>

#include "MultiThd.h"
#include "util.h"

struct nn_par
{
    // input
    const float* centers;
    const float* data;
    int k;
    int d;
    // output
    int *assignment;   // assignment of each points to center id
    float* cost;
};


struct kmeans_par
{
    // input
    int     verbose;        // verbosity level
    float   *data;          // data to be clustered. there are 'nxd' floating points in data
    int     n;              // number of data
    int     d;              // dimension of data
    int     k;              // number of centroids
    int     iter;           // number of iterations
    int     attempts;       // number of try. run with min cost will be returned
    int     nt;             // number of threads to use

    // output
    float   *centers;       // pointer to centers to be generated. size (k*d) must be pre-allocated
};

class Clustering
{
public:

    /**
    // k-means clustering routine
    // para: k-menas parameter
    // return: the cost of clustering. (average squared distance)
    */
    static float kmeans(kmeans_par* para)
    {
        if (para->verbose)
            printf("K-MEANS -- data: %u x %u  k: %u  iter: %u  attempts: %u.\n", para->n, para->d, para->k, para->iter, para->attempts);

        float cost = 1e100;
        float* centers_tmp = new float[para->k * para->d];
        for(int i = 0; i < para->attempts; i++)
        {
            float cost_tmp;

            kmeans_once(para->verbose, para->data, centers_tmp, para->n, para->d, para->k, para->iter, cost_tmp, para->nt);
            if (para->verbose)
                printf("Attempts: %u, cost: %e\n", i, cost_tmp);

            if( cost_tmp < cost )
            {
                cost = cost_tmp;
                memcpy(para->centers, centers_tmp, sizeof(float) * para->k * para->d);
            }
        }

        delete[] centers_tmp;
        return cost;
    }

private:
    /**
    // one time clustering with kmeans
    // data     : data to be clustered. there should be 'n*d' floating points in data
    // centers  : pointer to centers to be generated. size (k*d) should be pre-allocated
    // n        : number of items
    // d        : dimension of item
    // k        : number of centroids
    // iter     : number of iterations
    // cost     : cost of this run
    //
    */
    static void kmeans_once(int verbose, const float* data, float* centers, int n, int d, int k, int iter, float& cost, int nt)
    {
        //int* initial_center_idx = init_rand(n, k);
        int* initial_center_idx = init_kpp(n, k, d, data);

        // init centers with random generated index
        for(int i = 0; i < k; i++)
            for(int j = 0; j<d; j++)
                centers[i*d + j] = data[initial_center_idx[i]*d + j];

        delete[] initial_center_idx;


        int* ownership = new int[n];
        float* cost_tmp = new float[n];

        // begin iteration
        for(int iteration = 0; iteration < iter; iteration ++)
        {
            cost = 0;
            // re-assignment of center_id to each data
            /** assign points to clusters with multi-threading */
            nn_par ti = {centers, data, k, d, ownership, cost_tmp};
            MultiThd::compute_tasks(n, nt, &nn_task, &ti);
            for(int j = 0; j < n; j ++)
                cost += cost_tmp[j];


            if (verbose > 1)
                printf("Iter: %d   Cost: %e\n", iteration, cost);

            // re-calc centers
            memset(centers, 0, sizeof(float) * d * k);
            for(int i = 0; i<k; i++) // each center
            {
                int cnt = 0;
                for(int j = 0; j<n; j++) // each point
                {
                    if(ownership[j] == i) // assigned to center-i
                    {
                        cnt ++;
                        for(int p = 0; p<d; p++)
                        {
                            centers[i*d + p] += data[j*d + p];
                        }
                    }
                }

                for(int j = 0; j<d; j++)
                {
                    centers[i*d + j] /= cnt;
                }
            }

        }// end of iteration

        delete[] cost_tmp;
        delete[] ownership;
    }

    /**
    // assignment nearest center id to i-th point.
    // single computation for multi-threading computation.
    // arg      : type of nn_par, contains the input and output needed for computation
    // tid      : thread id
    // i        : the index of this computation task against all tasks
    // mutex    : mutex for updating shared data
    */
    static void nn_task(void* arg, int tid, int i, pthread_mutex_t& mutex)
    {
        nn_par* t = (nn_par*) arg;
        //printf("tid: %d, i: %d\n", tid, i);

        float dist_best = 1e100;
        for(int m = 0; m < t->k; m++) // loop for k centers, decides the nearest one
        {
            float dist = Util::dist_l2_sq(t->centers + m*t->d, t->data + i*t->d, t->d);
            if(dist < dist_best)
            {
                dist_best = dist;
                t->assignment[i] = m;
            }
        }
        t->cost[i] = dist_best;
    }

    /**
    // return k random number from [0, n-1]
    */
    static int* init_rand(int n, int k)
    {
        int* perm = Util::rand_perm(n, time(NULL));
        int* ret = new int[k];

        // return first k numbers after permutation
        memcpy(ret, perm, sizeof(int)*k);

        delete[] perm;
        return ret;
    }

    /**
    // return k seeds using k-means ++
    */
    static int* init_kpp (int n, int k, int d, const float * data)
    {
        int* sel_id = new int[k];

        float * dist_nn = new float[n]; // keeping shortest distance from each point to selected centers
        float * dist_tmp = new float[n];
        for(int i = 0; i<n; i++)
        {
            dist_nn[i] = 1e100;
        }

        // init the first center in uniformly random
        unsigned int seed = time(NULL);
        sel_id[0] = rand_r(&seed) % n;

        // init rest of the centers propotional to distance to seleced centers
        for (int i = 1 ; i < k ; i++)
        {
            int j;
            for (j = 0 ; j < n ; j++)
            {
                dist_tmp[j] = Util::dist_l2_sq(&data[j*d], &data[sel_id[i-1]*d], d); // get the dist from data[j] and last assigned center
                dist_nn[j] = std::min(dist_tmp[j], dist_nn[j]);
            }

            // normalize with l1_norm so as to act like proportional probability
            memcpy (dist_tmp, dist_nn, n * sizeof(float));
            float l1_norm = 0.0f;
            for(j = 0; j<n; j++)
                l1_norm += dist_tmp[j];

            for(j = 0; j<n; j++)
                dist_tmp[j] /= l1_norm;

            seed = time(NULL);
            double rd = rand_r(&seed)/(RAND_MAX + 1.0);

            for (j = 0 ; j < n - 1 ; j++)
            {
                rd -= dist_tmp[j];
                if (rd < 0)
                    break;
            }

            sel_id[i] = j;
        }

        delete[] dist_nn;
        delete[] dist_tmp;

        return sel_id;
    }

};

#endif // CLUSTERING_H_INCLUDED
