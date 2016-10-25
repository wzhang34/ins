#ifndef MUTITHD_MUTEX_H_INCLUDED
#define MUTITHD_MUTEX_H_INCLUDED

#include <cassert>
#include <pthread.h>

struct context_t
{
    pthread_mutex_t mutex;
    int i, n, tid;
    void (*task_fun) (void *arg, int tid, int i, pthread_mutex_t& m);
    void *task_arg;
};


class MultiThd
{
public:

    static void compute_tasks (int n, int nthread, void (*task_fun) (void *arg, int tid, int i, pthread_mutex_t& m), void *task_arg)
    {
        // init context
        context_t context;
        context.i = 0; // i-th task
        context.n = n; // number of task
        context.tid = 0; // boosted thread number
        context.task_fun = task_fun;
        context.task_arg = task_arg;
        pthread_mutex_init (&context.mutex, NULL);

        if(nthread==1)
            start_routine(&context);
        else
        {
            pthread_t *threads = new pthread_t[nthread];

            for (int i = 0; i < nthread; i++)
                pthread_create (&threads[i], NULL, &start_routine, &context);

            /* all computing */

            for (int i = 0; i < nthread; i++)
                pthread_join (threads[i], NULL);

            delete[] threads;
        }
    }

private:

    static void *start_routine (void *cp)
    {
        // get context
        context_t * context = (struct context_t*)cp;
        int tid;
        pthread_mutex_lock (&context->mutex);
        tid = context->tid++;
        pthread_mutex_unlock (&context->mutex);

        while(1) // keep computing till no more item in list
        {
            int item;
            pthread_mutex_lock (&context->mutex);
            item = context->i++;
            pthread_mutex_unlock (&context->mutex);
            if (item >= context->n) // no more item to do
                break;
            else
                context->task_fun (context->task_arg, tid, item, context->mutex);
        }

        return NULL;
    }

};



#endif // MUTITHD_MUTEX_H_INCLUDED
