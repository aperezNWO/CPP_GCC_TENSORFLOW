
#ifndef TENSORFLOWAPP_H // include guard
#define TENSORFLOWAPP_H
#endif

#include <tensorflow/c/c_api.h>
#include "Algorithm.h"

#define DLL_EXPORT extern "C" __declspec(dllexport) __stdcall

using namespace std;

class TensorFlowApp :
	public Algorithm
{
    public :
        //
        TensorFlowApp();
        ~TensorFlowApp();
        //
        const char*  GetTensorFlowAPIVersion();
        std::string  GetTensorFlowAppVersion(); 
        //
        //int          ReadConfigFile();
     public :
        //
        //map<string, string> configMap;

};

