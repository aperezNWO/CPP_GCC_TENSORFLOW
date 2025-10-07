
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
        int finalBoard[9];
        int moves[9];
        int winner;
        int moveCount;

        // NEW: Include history
        int history[10][9];   // Up to 10 states (initial + 9 moves)
        int historyCount;     // Actual number of states
    } TicTacToeResultOnline;

    // Now expose a function that fills all fields
    bool PlayTicTacToeGameWithHistory(TicTacToeResultOnline* result);
}


// Function to pause until user presses Enter
void waitForEnter() {
    std::cout << "Press Enter to continue...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    //std::cin.get(); // Wait for Enter (handles newline from previous input)
}

//
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
     public :
        //
        //bool PlayTicTacToeGameWithTensorFlow(TicTacToeGameResult* result);
        bool RunTicTacToeSelfPlayOnline(TicTacToeResultOnline& result);    
};

