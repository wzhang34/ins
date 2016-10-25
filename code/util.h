#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

//This file should be a project independent one, including commonly used functions in C++;

#include <vector>
#include <sstream>
#include <cmath>
#include <sched.h>
#include <cstdlib>
#include <sys/sysinfo.h>

#include <map>
#include <cstdlib>
#include <cmath>
#include <dirent.h>
#include <algorithm>
#include <cstring>
#include <cassert>


class Util
{
public:

//------------------------------------string---------------------------------------

    template <class T>
    static std::string num2str(T number)
    {
        std::stringstream ss;
        ss << number;
        return ss.str();
    }

    static int lastIndexOf(std::string str, char ch)
    {
        for(int i = str.length()-1; i>0; i--)
        {
            if(str[i] == ch)
                return i;
        }
        return 1;
    }

    static bool endWith(std::string str, std::string postfix)
    {
        assert(postfix.length() <= str.length() );
        return postfix == str.substr(str.length() - postfix.length(), postfix.length());
    }

    static bool startWith(std::string str, std::string prefix)
    {
        assert(prefix.length() <= str.length());
        return str.substr(0, prefix.length()) == prefix;
    }

    static std::string parseFileName(std::string fullpath)
    {
        int pos1 = Util::lastIndexOf(fullpath, '/');
        int pos2 = Util::lastIndexOf(fullpath, '.');
        return fullpath.substr(pos1+1, pos2-pos1-1);
    }

    static std::string strtok(std::string& str, std::string deli)
    {
        int i = str.find_first_of(deli, 0);

        std::string ret = str.substr(0, i);
        str = str.substr(i+1, str.size()-i-1);
        return ret;
    }

    static std::string trim(std::string str)
    {
        std::string trimStr = " ";
        std::string::size_type pos = str.find_first_not_of( trimStr );
        str.erase( 0, pos );

        pos = str.find_last_not_of( trimStr );
        str.erase( pos + 1 );

        return str;
    }

//------------------------------------math---------------------------------------

    template <class T>
    static float l1_norm(T* vec, int size)
    {
        float n = 0.0;
        for(int i = 0; i<size; i++)
            n += fabs(vec[i]);
        return n;
    }

    template <class T>
    static float l2_norm(T* vec, int size)
    {
        float n = 0.0;
        for(int i =0; i<size; i++)
            n += (vec[i]*vec[i]);

        return sqrt(n);
    }

    template <class T>
    static void normalize(T* vec, int size)
    {
        float norm = Util::l2_norm(vec, size);
        for(int i = 0; i<size; i++)
            vec[i] /= norm;
    }


//------------------------------------Misc----------------------------------------

    static void project(const float* P, int row, int col, const float* x, float* out)
    {
        // out = Px
        memset(out, 0, sizeof(float)*row);
        for(int i = 0; i < row; i++)
            for(int j = 0; j < col; j++)
                out[i] += P[i*col+j]*x[j];
    }

    static void exec(std::string cmd)
    {
        int ret = system(cmd.c_str());
        if (ret != 0)
        {
            printf("Error when excuting the following command: \n%s\n", cmd.c_str());
        }
    }

    // return a random perm of [0 1 2... n-1]
    static int* rand_perm(int n, unsigned int seed)
    {
        int *perm = new int[n];
        for (int i = 0; i < n; i++)
            perm[i] = i;

        int tmp;
        for (int i = 0; i < n-1 ; i++)
        {
            int j = i +  rand_r(&seed) % (n - i);
            //swap
            tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }

        return perm;
    }


    // squared l2 distance
    static float dist_l2_sq(const float* a, const float* b, int d)
    {
        float dist = 0.0f;
        for(int i = 0; i<d; i++)
            dist += (a[i] - b[i])*(a[i] - b[i]);


        return dist;///(Util::l2_norm(a, d)*Util::l2_norm(b, d));
    }

    static float dist_l2_sq1(const float* a, const float* b, int d)
    {
        float norm1 = Util::l1_norm(a, d);
        float norm2 = Util::l1_norm(b, d);

        float dist = 0.0f;
        for(int i = 0; i<d; i++)
        {
            dist += sqrt(a[i]*b[i]);
        }

        return 2- 2*dist/sqrt(norm1*norm2);
    }


    // get number of cpu cores
    static int count_cpu ()
    {
        cpu_set_t set;
        sched_getaffinity (0, sizeof (cpu_set_t), &set);
        int count = 0;
        for (int i = 0; i < CPU_SETSIZE; i++)
        if (CPU_ISSET (i, &set))
          count++;
        return count;
    }


    // get memory size in MB
    static int getTotalSystemMemory()
    {
        struct sysinfo myinfo;
        sysinfo(&myinfo);

        return (int)(myinfo.mem_unit * myinfo.totalram / 1024 / 1024);
    }

    // gives C(n, r) in double, in case return value is too large to fit into a long
    static double combination (int n, int r)
    {
        double ret = 1;
        for(int i = r+1; i<n+1; i++)
            ret *= (double)i;

        for(int i = 1; i<n-r+1; i++)
            ret /= (double)i;

        return ret;
    }
};

#endif // UTIL_H_INCLUDED









