#ifndef _CPARAMREADER_H
#define _CPARAMREADER_H

////////////////////////////////////////////////////////////////////////////////////
// Notes: the parameters are
// 1. This class is used to read a configuration file
// 2. The format for the configuration is as follow:
//    param1 = setting1
//    param2 = setting2
// 3. To use
//    CParamReader *params	= new CParamReader ("config.txt");  // Declaration
//
//    int maxNumOfStudents	= params->GetParamInt  ("max_students");
//    float simThres		= params->GetParamFlt  ("simThres");
//    string name			= params->GetParamStr  ("studentname");
//    bool bDisplay			= params->GetParamBool ("bDisplay");
//
//    params->print();   // Display to screen
//
////////////////////////////////////////////////////////////////////////////////////
// modified from HK Tan's ParamReader in TAlign. Thanks Hung Kong Tan

#include <cstdio>
#include <map>
#include <cstring>

#include "util.h"

using std::string;
using std::map;

class CParamReader {

	map<string, string> params;

public:

	~CParamReader ()
	{
        params.clear();
	}

	CParamReader (string paramFileName)
	{
		ReadParamFile (paramFileName);
	}




	void ReadParamFile (string paramFileName)
	{
		string seps = "=", var;
		FILE* fin = fopen(paramFileName.c_str(), "r");
		if (!fin)
		{
		    printf("Config file not found: %s\n", paramFileName.c_str());
		}

		const int max_len = 2000;
        char* buffer = new char[max_len];

        while ( fgets(buffer, max_len, fin) )
        {
            string line(buffer);
            if(line[0] == '\n' || line[0] == '#')
                continue;

            var = Util::trim(Util::strtok(line, seps));
            string val = Util::trim(line).substr(0, line.length() -2);

            if( params.find(var) != params.end() )
            {
                printf("redefinition of parameter: %s\n", var.c_str());
                fflush(stdout);
                exit(1);
            }
            params.insert( std::pair<string, string>(var, val) );
        }

        delete[] buffer;

        fclose(fin);
	}


	string GetStr (string param)
	{
        if ( params.find(param) != params.end() )
            return params[param];
        else
        {
            printf("error: undefined %s.\n", param.c_str());
            exit(1);
        }
	}

	int GetInt (std::string param)
	{
        if ( params.find(param) != params.end() )
            return atoi(params[param].c_str());
        else
        {
            printf("error: undefined %s.\n", param.c_str());
            exit(1);
        }

	}

	float GetFlt (std::string param)
	{
	    if ( params.find(param) != params.end() )
            return atof(params[param].c_str());
        else
        {
            printf("error: undefined %s.\n", param.c_str());
            exit(1);
        }
	}

	void print()
	{
	    printf("Parameters loaded:\n");
		for (map<string, string>::iterator it = params.begin(); it != params.end(); it++ )
			printf("%s  =  %s\n", it->first.c_str(), it->second.c_str());

        printf("\n"); fflush(stdout);
	}
};

#endif
