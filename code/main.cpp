#include <cstdio>
#include <cstdlib>
#include <string>

#include "ParamReader.h"
#include "config.h"
#include "Vocab_Gen.h"
#include "Index.h"
#include "HE.h"
#include "SearchEngine.h"


using std::string;



Config con; // global configuration

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: \n%s <config_file> <mode>\n", argv[0]);
        exit(1);
    }

    char* conf              = argv[1];
    con.mode                = atoi(argv[2]);

    CParamReader *params    = new CParamReader (conf);

    con.dataId              = params->GetStr ("dataId");
    string id               = con.dataId;
    con.nt                  = params->GetInt ("nt");
    con.nt                  = std::min(con.nt, Util::count_cpu());


    switch(con.mode)
    {
        case 1: // whole procedure
        {
            con.train_desc          = params->GetStr ("train_desc");
            con.index_desc          = params->GetStr ("index_desc");
            con.query_desc          = params->GetStr ("query_desc");

            con.dim                 = params->GetInt ("dim");
            con.bf                  = params->GetInt ("bf");
            con.num_layer           = params->GetInt ("num_layer");
            con.num_per_file        = params->GetFlt ("num_per_file");
            con.T                   = params->GetInt ("T");
            con.iter                = params->GetInt ("iter");
            con.attempts            = params->GetInt ("attempts");
            con.verbose             = params->GetInt ("verbose");

            con.ht                  = params->GetInt ("ht");
            con.he_len              = params->GetInt ("he_len");
            con.p_mat               = params->GetStr ("pmat");


            con.ht                  = params->GetInt ("ht");
            con.search_mode         = params->GetInt ("search_mode");
            con.num_ret             = params->GetInt ("num_ret");

            Timer* t = new Timer();
            t->start();
            Vocab_Gen::genVoc(con.train_desc, con.bf, con.num_layer);
            IO::appendLine("time.txt", Util::num2str(t->elapsed()));


            Vocab* voc = new Vocab(con.bf, con.num_layer, con.dim);
            voc->loadFromDisk(id + ".out/vk_words/");

            t->start();
            HE* he = new HE(con.he_len, con.dim, con.p_mat, voc->num_leaf, con.ht);
            he->train(voc, id + ".out/matrix/M.l0.n0", con.nt);
            Util::exec("mv " + id + ".out/matrix/M.l0.n0.median " + id + ".out/he.median");
            IO::appendLine("time.txt", Util::num2str(t->elapsed()));

            he->loadMedian(id + ".out/he.median");

            t->start();
            IO::mkdir(id + ".out/index/");
            Index::indexFiles(voc, he, con.index_desc, ".feat", id + ".out/index/" + id + "/", con.nt);
            IO::appendLine("time.txt", Util::num2str(t->elapsed()));

            SearchEngine* engine = new SearchEngine(voc, he);
            engine->loadIndexes(id + ".out/index/");
            t->start();
            engine->search_dir(con.query_desc, id + ".out/result", con.num_ret);\
            IO::appendLine("time.txt", Util::num2str(t->elapsed()));

            delete voc;
            delete he;
            break;
        }
        case 2: // gen vocab
        {
            con.verbose             = params->GetInt ("verbose");
            con.train_desc          = params->GetStr ("train_desc");
            con.dim                 = params->GetInt ("dim");
            con.bf                  = params->GetInt ("bf");
            con.num_layer           = params->GetInt ("num_layer");
            con.num_per_file        = params->GetFlt ("num_per_file");
            con.T                   = params->GetInt ("T");
            con.iter                = params->GetInt ("iter");
            con.attempts            = params->GetInt ("attempts");

            Vocab_Gen::genVoc(con.train_desc, con.bf, con.num_layer);
            break;
        }
        case 3: // hamming training
        {
            con.dim                 = params->GetInt ("dim");
            con.bf                  = params->GetInt ("bf");
            con.num_layer           = params->GetInt ("num_layer");

            con.ht                  = params->GetInt ("ht");
            con.he_len              = params->GetInt ("he_len");
            con.p_mat               = params->GetStr ("pmat");


            Vocab* voc = new Vocab(con.bf, con.num_layer, con.dim);
            voc->loadFromDisk(id + ".out/vk_words/");

            HE* he = new HE(con.he_len, con.dim, con.p_mat, voc->num_leaf, con.ht);
            he->train(voc, id + ".out/matrix/M.l0.n0", con.nt);
            Util::exec("mv " + id + ".out/matrix/M.l0.n0.median " + id + ".out/he.median");

            delete voc;
            delete he;

            break;
        }
        case 4: // index file
        {
            con.index_desc          = params->GetStr ("index_desc");
            con.dim                 = params->GetInt ("dim");

            con.bf                  = params->GetInt ("bf");
            con.num_layer           = params->GetInt ("num_layer");

            con.he_len              = params->GetInt ("he_len");
            con.p_mat               = params->GetStr ("pmat");
            con.ht                  = params->GetInt ("ht");


            Vocab* voc = new Vocab(con.bf, con.num_layer, con.dim);
            voc->loadFromDisk(id + ".out/vk_words/");

            HE* he = new HE(con.he_len, con.dim, con.p_mat, voc->num_leaf, con.ht);
            he->loadMedian(id + ".out/he.median");

            IO::mkdir(id + ".out/index/");
            Index::indexFiles(voc, he, con.index_desc, ".feat", id + ".out/index/" + id + "/", con.nt);

            delete voc;
            delete he;
            break;
        }
        case 5: // online search
        {
            con.query_desc          = params->GetStr ("query_desc");
            con.ht                  = params->GetInt ("ht");
            con.search_mode         = params->GetInt ("search_mode");
            con.dim                 = params->GetInt ("dim");
            con.bf                  = params->GetInt ("bf");
            con.num_layer           = params->GetInt ("num_layer");

            con.he_len              = params->GetInt ("he_len");
            con.p_mat               = params->GetStr ("pmat");
            con.im_sz               = params->GetInt ("im_size");

            con.num_ret             = params->GetInt ("num_ret");
            con.ma                  = params->GetInt ("ma");


            Vocab* voc = new Vocab(con.bf, con.num_layer, con.dim);
            voc->loadFromDisk(id + ".out/vk_words/");

            HE* he = new HE(con.he_len, con.dim, con.p_mat, voc->num_leaf, con.ht);
            he->loadMedian(id + ".out/he.median");

            SearchEngine* engine = new SearchEngine(voc, he);
            engine->loadIndexes(id + ".out/index/");
            engine->search_dir(con.query_desc, id + ".out/result", con.num_ret);

            delete engine;
            delete voc;
            delete he;

            break;
        }
        default:
        {
            printf("Un-defined operation! Exitting ...\n");
            return 1;
        }

    }


    delete params;
    return 0;
}
