#ifndef ENTRY_H_INCLUDED
#define ENTRY_H_INCLUDED



#define PI 3.1415926
#define ROUND(d) ((unsigned int)(floor(d+0.5)))

#include <cmath>
#include <cstdio>

const float BASE = log(0.71);

struct Entry
{

    // id of entry is tricky here. serves as both id and word_id
    unsigned int id:32;          // id. taking 32 bits can index up to 4 trillion images
    unsigned int row:16;            // row of point, [0, 65536)
    unsigned int col:16;            // col of point, [0, 65536)
    unsigned int ori:16;             //orientation, ori = round(raw_ori*1000)
    unsigned int scale:16;           //scale, scale = round(log2(raw_scale)/1000);
    unsigned int sig:32;            //hamming signature.

    Entry(){} // do nothing constructor

    // new entry with raw data
    /*Entry(float r, float c, unsigned int idx, float scl, float orient, unsigned int signature)
    {
        row     = ROUND(r);
        col     = ROUND(c);
        id      = idx;
        scale   = ROUND((log(scl) - BASE)*10000); // [0 ln(128)*10000] = [0 48520] < 2^16 = 65536
        ori     = ROUND((orient+PI)*10000); // [0 2*PI*10000] = [0 62800] < 2^16
        sig     = signature;
    }

    // new entry with processed data
    Entry(unsigned int r, unsigned int c, unsigned int idx, unsigned int scl, unsigned int orient, unsigned int signature)
    {
        row     = r;
        col     = c;
        id      = idx;
        scale   = scl;
        ori     = orient;
        sig     = signature;
    }*/

    void set(float r, float c, unsigned int idx, float scl, float orient, unsigned int signature)
    {
        row     = ROUND(r);
        col     = ROUND(c);
        id      = idx;
        scale   = ROUND((log(scl) - BASE)*10000); // [0 ln(128)*10000] = [0 48520] < 2^16 = 65536
        ori     = ROUND((orient+PI)*10000); // [0 2*PI*10000] = [0 62800] < 2^16
        sig     = signature;
    }

    /*
    void set(unsigned int r, unsigned int c, unsigned int idx, unsigned int scl, unsigned int orient, unsigned int signature)
    {
        row     = r;
        col     = c;
        id      = idx;
        scale   = scl;
        ori     = orient;
        sig     = signature;
    }*/

    // cpy constructor
    Entry(const Entry& p) // copy constructor
    {
        this->id        = p.id;
        this->ori       = p.ori;
        this->scale     = p.scale;
        this->sig       = p.sig;
        this->row       = p.row;
        this->col       = p.col;
    }

    ~Entry(){}

    // for debug
    void print()
    {
        printf("%u %u %u %u %u %u\n", row, col, id, scale, ori, sig);
    }

    // write an entry record to fout
    void write(FILE* fout)
    {
        unsigned int* items = new unsigned int[6];
        items[0] = row;
        items[1] = col;
        items[2] = id;
        items[3] = scale;
        items[4] = ori;
        items[5] = sig;

        fwrite(items, sizeof(unsigned int), 6, fout);

        delete[] items;
    }

    // read a single entry from fin
    void read(FILE* fin)
    {
        unsigned int *items = new unsigned int[6];
        assert(6 == fread(items, sizeof(unsigned int), 6, fin));

        row     = items[0];
        col     = items[1];
        id      = items[2];
        scale   = items[3];
        ori     = items[4];
        sig     = items[5];

        delete[] items;
    }
};

#endif // ENTRY_H_INCLUDED
