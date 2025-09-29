
#ifndef TENSORFLOWAPP_H // include guard
#define TENSORFLOWAPP_H
#endif

#include <tensorflow/c/c_api.h>
#include "Algorithm.h"


#define DLL_EXPORT extern "C" __declspec(dllexport) __stdcall

using namespace std;

// Forward declaration of class
class TensorFlowApp;

// C-style export types
extern "C" {
    typedef struct {
        int board[9];
        int moves[9]; // -1 if not used
        int winner;   // 1=X, -1=O, 0=draw
        int moveCount;
    } TicTacToeResult;

    bool PlayTicTacToeGame(int* boardOut, int* movesOut, int* winnerOut, int* moveCountOut);
}


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
        bool RunTicTacToeSelfPlay(TicTacToeResult& result);

};

