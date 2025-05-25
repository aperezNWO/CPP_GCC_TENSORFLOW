
#ifndef TENSORFLOWAPP_H // include guard
#define TENSORFLOWAPP_H
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <map>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <random>
#include <regex>
#include <cctype>

#ifndef TESSERACTAPP_H // include guard
#define TESSERACTAPP_H
#endif


#define DLL_EXPORT extern "C" __declspec(dllexport) __stdcall __cdecl

using namespace std;

class TensorFlowApp
{
    public :
        //
        TensorFlowApp();
        ~TensorFlowApp();
        //
        const char*  GetTensorFlowAPIVersion();
        const char*  GetTensorFlowAppVersion(); 
        //
        int          ReadConfigFile();
     public :
        //
        map<string, string> configMap;

};

