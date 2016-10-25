#ifndef IO_H_INCLUDED
#define IO_H_INCLUDED

#define byte unsigned char

#include <cstdio>
#include <dirent.h>
#include <sys/stat.h>
#include <vector>
#include <cstring>

#include "MultiThd.h"
#include "util.h"
#include "config.h"

using std::string;
using std::vector;



extern Config con;


struct div_par
{
    FILE* fin; // file handler for input base matrix
    const float* centers; // centers defines the quantization rule. size: k x d
    int k;
    int d;

    int layer;
    int number;
    string mtrx;

    // aux fields for hanlding the race when writing output
    int* writting;

    // out
    int* cnt; // number of points quantized to each center. size: k x 1
};

class IO
{
public:

    // append a line at the end of a file
    static void appendLine(string filename, string info)
    {
        FILE* fout = fopen(filename.c_str(), "a");
        assert(fout != NULL);

        fprintf(fout, "%s\n", info.c_str());

        fclose(fout);
    }

    /**!!! be ware of the memory this returns*/
    /// read feat_file
    /// return feature_file: n x (m+d)
    static float* readFeatFile(string feat_file, int& n, int& m, int& d)
    {
        FILE* fin = fopen(feat_file.c_str(), "rb");
        chkFileErr(fin, feat_file);

        assert( 1 == fread(&n, sizeof(int), 1, fin) );
        assert( 1 == fread(&m, sizeof(int), 1, fin) );
        assert( 1 == fread(&d, sizeof(int), 1, fin) );

        byte* tmp = new byte[d];
        float* feature = new float[n*(m+d)];
        for(int i = 0; i < n; i++)
        {
            assert( m == (int)fread(feature + i*(m+d), sizeof(float), m, fin));

            assert( d == (int)fread(tmp, sizeof(byte), d, fin));

            // convert to float array
            int pos = i * (m+d) + m;
            for(int j = 0; j < d; j++)
                feature[pos + j] = (float)(tmp[j]);
        }
        //assert( n*(m+d) == (int)fread(feature, sizeof(float), n*(m+d), fin) );

        delete[] tmp;

        fclose(fin);

        return feature;
    }

    // obsolete
    // read text-based matrix file
    /*static float* readMatrix(string mtrx, int& row, int& col)
    {
        FILE* fin = fopen(mtrx.c_str(), "r");
        chkFileErr(fin, mtrx);


        fscanf(fin, "%d %d", &row, &col);
        float* mat = new float[row*col];

        for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                fscanf(fin, "%f", mat + i*col + j);
        }

        fclose(fin);
        return mat;
    }*/

    /**
    // memory-unsafe
    // load matrix from disk to memory
    // mtrx: the matrix file. size: row*col
    // row & col: reference to row and col of the matrix
    // mat [out]: pointer to matrix to be initialized
    // when limit_points >  0, only limit_points entry will be read
    // when limit_points <= 0, all data in 'mtrx' will be read
    */
    // following load*Mat load different types of matrix file to memory.
    // format of matrix:
    // row (int) col (int)
    // row * col (data type)
    static float* loadFMat(string mtrx, int& row, int& col, int limit_points)
    {
        FILE* fin = fopen(mtrx.c_str(), "rb");
        chkFileErr(fin, mtrx);

        assert( 1 ==  fread(&row, sizeof(int), 1, fin) );
        assert( 1 ==  fread(&col, sizeof(int), 1, fin) );


        if (limit_points > 0)
            row = std::min((int)limit_points, row);

        float* mat = new float[col*row];

        assert( row*col == (int)fread(mat, sizeof(float), row*col, fin) );
        fclose(fin);

        return mat;
    }


    static int* loadIMat(string mtrx, int& row, int& col, int limit_points)
    {
        FILE* fin = fopen(mtrx.c_str(), "rb");
        chkFileErr(fin, mtrx);

        assert( 1 == fread(&row, sizeof(int), 1, fin) );
        assert( 1 == fread(&col, sizeof(int), 1, fin) );


        if (limit_points > 0)
            row = std::min(limit_points, row);

        int* mat = new int[row*col];

        assert( row*col == (int)fread(mat, sizeof(int), row*col, fin) );
        fclose(fin);

        return mat;
    }


    static void writeMat(float* mat, int n, int d, std::string file)
    {
        FILE* fout = fopen(file.c_str(), "wb");
        chkFileErr(fout, file);

        fwrite(&n, sizeof(int), 1, fout);
        fwrite(&d, sizeof(int), 1, fout);

        fwrite(mat, sizeof(float), n*d, fout);

        fclose(fout);
    }

    static void writeMat(int* mat, int n, int d, std::string file)
    {
        FILE* fout = fopen(file.c_str(), "wb");
        chkFileErr(fout, file);

        fwrite(&n, sizeof(int), 1, fout);
        fwrite(&d, sizeof(int), 1, fout);

        fwrite(mat, sizeof(int), n*d, fout);

        fclose(fout);
    }


    // get all directories (except . and ..)
    static vector<string> getFolders(string dir)
    {
        if (!Util::endWith(dir, "/"))
        {
            printf("Error: %s is not a directory.\n", dir.c_str());
            exit(1);
        }

        vector<string> folders;

        DIR* dp = opendir(dir.c_str());
        struct dirent* dirp;
        if(!dp)
        {
            printf("error when opening directory: %s\n", dir.c_str());
            exit(1);
        }
        std::string filename;
        while((dirp = readdir(dp)) != NULL)
        {
            filename = dirp->d_name;
            if ( filename == "." || filename == "..")
                continue;

            struct stat s;
            stat((dir + filename).c_str(), &s);
            if (S_ISDIR(s.st_mode)) //dirp is a directory: DT_DIR
                folders.push_back(dir + filename);

        }
        closedir(dp);

        std::sort(folders.begin(), folders.end());

        return folders;
    }

    /**
    // return files in a directory
    // search_mode: 0: topdir only, 1: recursively
    // name_mode: 0: str = startwithstr, 1: str = endwithstr
    // dir should end up with '/'
    */
    static vector<string> getFileList(string dir, string substr, int search_mode, int name_mode)
    {
        if(!Util::endWith(dir, "/"))
        {
            printf("Error: %s is not a directory.\n", dir.c_str());
            exit(1);
        }

        vector<string> filelist;
        vector<string> subFilelist;
        filelist.clear();
        subFilelist.clear();

        DIR* dp = opendir(dir.c_str());
        struct dirent* dirp;
        if(!dp)
        {
            printf("error when opening directory: %s\n", dir.c_str());
            exit(1);
        }

        std::string filename;
        while((dirp = readdir(dp)) != NULL)
        {
            filename = dirp->d_name;
            if ( filename == "." || filename == "..")
                continue;

            struct stat s;
            stat((dir + filename).c_str(), &s);
            if (S_ISDIR(s.st_mode) && search_mode == 1)
            {//dirp is a directory: DT_DIR
                subFilelist = IO::getFileList(dir + filename + "/", substr, 1, name_mode);
            }
            else
            {// it's a regular file
                if(name_mode == 0 && Util::startWith(filename, substr)) // filename should match in startwithstr manner
                {
                    filelist.push_back(dir+filename);
                }
                else if(name_mode == 1 && Util::endWith(filename, substr))// filename end with something in common endwithstr
                {
                    filelist.push_back(dir+filename);
                }
            }


            filelist.insert(filelist.end(), subFilelist.begin(), subFilelist.end());
            subFilelist.clear();

        }
        closedir(dp);

        std::sort(filelist.begin(), filelist.end());

        return filelist;
    }


    // check if file is ok
    static void chkFileErr(FILE* f, string fn)
    {
        if (!f)
        {
            printf("FILE IO ERROR: %s\n", fn.c_str());
            exit(1);
        }
    }

    static void mkdir(string dir)
    {
        if (!f_exists(dir))
            Util::exec("mkdir " + dir);
    }

    // remove file
    static void rm(string file)
    {
        if(f_exists(file)) // if file exists
            Util::exec("rm " + file);
    }


    static bool f_exists(string file)
    {
        struct stat sb;
        return stat(file.c_str(), &sb) == 0;
    }


    /**
    // note this can not be like IO::loadMatrix, since the limitation of memory size.
    // so this function read & write data sequencially (slow)
    // divide 'mtrx' into n sub matrix, according to the quantization rule defined by 'centers'
    // mtrx     : base matrix file. the complete name for spilting is 'mtrx_layer_number'
    // layer    : layer id
    // number   : number id of center in layer 'layer'
    // centers  : defines quantization using centers. size n x d
    // n        : size for centers
    // d        : dimension for centers
    */
    static void divMatByCenters_MT(string mtrx, int layer, int number, const float* centers, int k, int d, int nt)
    {
        // write reserved bytes to each divided matrix file
        int fakeHead[2] = {0, d};
        for(int i = 0; i < k; i++)
        {
            FILE* fout = fopen((mtrx + ".l" + Util::num2str(layer+1) + ".n" + Util::num2str(number*k + i)).c_str(), "wb");
            fwrite(fakeHead, sizeof(int), 2, fout);
            fclose(fout);
        }

        // start dividing
        string baseMtrx = mtrx + ".l" + Util::num2str(layer) + ".n" + Util::num2str(number);
        FILE* fin = fopen( baseMtrx.c_str(), "rb");
        IO::chkFileErr(fin, baseMtrx);

        int row, col;
        int* count = new int[k];
        int* writting = new int[k];
        memset(count, 0, sizeof(int)*k);
        memset(writting, 0, sizeof(int)*k);

        assert( 1 == fread(&row, sizeof(int), 1, fin) );
        assert( 1 == fread(&col, sizeof(int), 1, fin) );
        assert(d == col || "dimension of feature must be consistent.");

        div_par para = {fin, centers, k, d, layer, number, mtrx, writting, count};
        MultiThd::compute_tasks(row, nt, &div_task, &para);
        fclose(fin);

        for(int i = 0; i < k; i++)
        {
            FILE* fout = fopen((mtrx + ".l" + Util::num2str(layer+1) + ".n" + Util::num2str(number*k + i)).c_str(), "r+b");
            fwrite(count + i, sizeof(int), 1, fout);
            fclose(fout);
        }

        delete[] count;
        delete[] writting;
    }

    static void div_task(void* arg, int tid, int i, pthread_mutex_t& mutex)
    {
        div_par* t = (div_par*) arg;

        float* vec = new float[t->d];

        // read data
        pthread_mutex_lock(&mutex);
        assert(t->d == (int)fread(vec, sizeof(float), t->d, t->fin));
        pthread_mutex_unlock(&mutex);

        float dist_best = 1e100;
        int nn_id = -1;
        for(int m = 0; m < t->k; m++) // loop for k centers, decides the nearest one
        {
            float dist = Util::dist_l2_sq(t->centers + m*t->d, vec, t->d);
            if(dist < dist_best)
            {
                dist_best = dist;
                nn_id = m;
            }
        }


        while(1)
        {
            pthread_mutex_lock(&mutex);

            // prepare to write file if no other threads writting to 'nn_id'-th file
            if(t->writting[nn_id] == 0)
            {
                t->writting[nn_id] = 1;
                t->cnt[nn_id] ++;

                pthread_mutex_unlock(&mutex);

                break; // go ahead and write file
            }
            else // wait another thread finishing writting
            {
                pthread_mutex_unlock(&mutex);
                usleep(rand() % 10);
            }
        }



        FILE* fout = fopen((t->mtrx + ".l" + Util::num2str(t->layer+1) + ".n" + Util::num2str(t->number*t->k + nn_id)).c_str(), "ab");
        fwrite(vec, sizeof(float), t->d, fout);
        fclose(fout);

        pthread_mutex_lock(&mutex);
        t->writting[nn_id] = 0;
        pthread_mutex_unlock(&mutex);

        delete[] vec;
    }






    /**
    // genMtrx(string dir, string mtrx, int num_points, bool normalize);
    // generate matrix by sampling on the feature files in 'dir'
    // dir         : src feature directory
    // mtrx        : dest mtrx file location
    // num_point   : # of points to be sampled from 'dir' to 'mtrx'
    */
    static void genMtrx(std::string dir, std::string mtrx, int num_point)
    {
        printf("Samping %d points.\n", num_point);
        vector<string> fileList = IO::getFileList(dir.c_str(), con.extn, 1, 1);
        assert( fileList.size() || !"Directory is empty for sampling.");


        FILE* fout = fopen(mtrx.c_str(), "wb"); // num_points x con.dim
        IO::chkFileErr(fout, mtrx);
        fwrite(&num_point, sizeof(int), 1, fout);
        fwrite(&con.dim, sizeof(int), 1, fout);


        int i = 0; // indicating number of points sampled.
        float tmp;
        FILE* fin = NULL;

        int* header = new int[3]; // [n m d]
        byte* vec = new byte[con.dim]; // single feature
        srand(time(NULL));
        while( i < num_point ) // sample num_points entries
        {
            string filename = fileList[rand() % fileList.size()];
            fin = fopen(filename.c_str(), "rb");
            IO::chkFileErr(fin, filename);


            assert( 3 == fread(header, sizeof(int), 3, fin) ); // read header
            assert( header[2] == con.dim);

            int sample_this_file = std::ceil(header[0] * con.num_per_file);
            if (sample_this_file == 0) // if no points to sample for this file, skip.
            {
                fclose(fin);
                continue;
            }

            // gen random lines ids to sample
            vector<int> rand_line;
            for (int j = 0; j< sample_this_file; j++)
            {
                int rand_num = rand() % header[0];

                while ( find(rand_line.begin(), rand_line.end(), rand_num) != rand_line.end() )
                    rand_num = rand() % header[0];

                rand_line.push_back(rand_num);
            }
            std::sort(rand_line.begin(), rand_line.end());


            // cp the desired features
            int lastPos = 0;
            for(int j = 0; j < sample_this_file; j++)
            {
                fseek(fin, (rand_line[j] - lastPos) * (sizeof(byte) * header[2] + sizeof(float)*header[1]) + sizeof(float) * header[1], SEEK_CUR);
                lastPos = rand_line[j] + 1;

                assert( header[2] == (int)fread(vec, sizeof(byte), header[2], fin) );
                for(int k = 0; k < header[2]; k++)
                {
                    tmp = (float)(vec[k]);
                    assert(tmp <= 255);
                    fwrite(&tmp, sizeof(float), 1, fout);
                }

            }


            fclose(fin);
            rand_line.clear();
            i += sample_this_file;
            printf("\r%.2lf%%", ((float)i/num_point)*100); fflush(stdout);
        }
        fclose(fout);
        printf("\n");

        delete[] vec;
        delete[] header;
    }
};

#endif // IO_H_INCLUDED
