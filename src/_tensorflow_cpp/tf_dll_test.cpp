/*

1) TOOLCHAIN : C:\msys64\ucrt64\bin (CHANGE PATH)

2) g++ -std=c++20 -o tf_dll_test.exe tf_dll_test.cpp 


3) UTILIZAR PROYECDTO CPP_GCC_TENSORFLOW.DEV (Embarcadero Dev C++) PROVISIONALMENTE PARA 
   
   A) VISUALIZAR Y EDITAR ARCHIVOS.
   B) COMPILAR CON LINEA DE COMANDOS.
   C) EJECUTAR COMANDOS GIT
   
*/

#include <stdio.h>
#include <windows.h>
#include <iostream>
#include <string>
#include <cctype>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <ctime>
#include <fstream>   // For file reading/writing
#include <iomanip>   // For std::setprecision

typedef const char* (*GetTensorFlowAPIVersionFunc)();  // Define function pointer type
typedef const char* (*GetTensorFlowAPPVersionFunc)();  // Define function pointer type
typedef const char* (*GetCPPSTDVersionFunc)();         // Define function pointer type

typedef const char* (*GetStringFunc)();
typedef bool (*PlayTTTFunc)(int*, int*, int*, int*);

// Function to pause until user presses Enter
void waitForEnter() {
    std::cout << "Press Enter to continue...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    //std::cin.get(); // Wait for Enter (handles newline from previous input)
}

// Function to ask if user wants to continue
bool askToContinue() {
    std::string response;
    while (true) {
        std::cout << "\nDo you want to watch another game? (y/n): ";
        std::getline(std::cin, response);

        // Convert to lowercase for case-insensitive comparison
        std::string lowerResponse;
        std::transform(response.begin(), response.end(), std::back_inserter(lowerResponse),
                      [](unsigned char c){ return std::tolower(c); });

        if (lowerResponse == "y" || lowerResponse == "yes") {
            return true;
        } else if (lowerResponse == "n" || lowerResponse == "no") {
            return false;
        } else {
            std::cout << "Please enter 'y' or 'n'.\n";
        }
    }
}


int main() {
    // Load the DLL
    HMODULE hDLL = LoadLibrary("TensorFlowAppCPP.dll");
    if (hDLL == NULL) {
        printf("Failed to load DLL.\n");
        return 1;
    }
    
	//////////////////////////////////////////////////////
    //GetTensorFlowAPIVersion
    //////////////////////////////////////////////////////
    
    GetTensorFlowAPIVersionFunc GetTensorFlowAPIVersion = (GetTensorFlowAPIVersionFunc)GetProcAddress(hDLL, "GetTensorFlowAPIVersion");
    if (!GetTensorFlowAPIVersion) {
        printf("Could not locate the function 'GetTensorFlowAPIVersion'.\n");
        FreeLibrary(hDLL);
        return 1;
    }

    // Call the function
    const char* apiVersion = GetTensorFlowAPIVersion();
    printf("'GetTensorFlowAPIVersion' : %s\n", apiVersion);

	//////////////////////////////////////////////////////
    //GetTensorFlowAPPVersion
    //////////////////////////////////////////////////////
    
    
    GetTensorFlowAPPVersionFunc GetTensorFlowAPPVersion = (GetTensorFlowAPPVersionFunc)GetProcAddress(hDLL, "GetTensorFlowAppVersion");
    if (!GetTensorFlowAPPVersion) {
        printf("Could not locate the function 'GetTensorFlowAPPVersion'.\n");
        FreeLibrary(hDLL);
        return 1;
    }

    // Call the function
    const char* appVersion = GetTensorFlowAPPVersion();
    printf("'GetTensorFlowAPPVersion' : %s\n", appVersion);
    
    //////////////////////////////////////////////////////
    //GetCPPSTDVersion
    //////////////////////////////////////////////////////
    
    
    GetCPPSTDVersionFunc GetCPPSTDVersion = (GetCPPSTDVersionFunc)GetProcAddress(hDLL, "GetCPPSTDVersion");
    if (!GetCPPSTDVersion) {
        printf("Could not locate the function 'GetCPPSTDVersion'.\n");
        FreeLibrary(hDLL);
        return 1;
    }

    // Call the function
    const char* cppSTDVersion = GetCPPSTDVersion();
    printf("'GetCPPSTDVersion' : %s\n", cppSTDVersion);
    
    ///////////////////////////////////////////////////
    // === New Function: PlayTicTacToeGame ===
    ///////////////////////////////////////////////////

    PlayTTTFunc PlayTicTacToeGame = (PlayTTTFunc)GetProcAddress(hDLL, "PlayTicTacToeGame");
    if (!PlayTicTacToeGame) {
        printf("Could not locate 'PlayTicTacToeGame'\n");
        FreeLibrary(hDLL);
        return 1;
    }

   	do 
	{
		//
	    int board[9], moves[9], winner, moveCount;
	    
		// RETURN A MATRIX OF BOARD GAME STEPS
		//
	    if (PlayTicTacToeGame(board, moves, &winner, &moveCount)) {
	        printf("\n--- TIC-TAC-TOE GAME RESULT ---\n");
	        printf("Winner: %s\n", winner == 1 ? "X" : winner == -1 ? "O" : "Draw");
	        printf("Moves: ");
	        for (int i = 0; i < moveCount; ++i) printf("%d ", moves[i]);
	        printf("\n");
	    } else {
	        printf("Game execution failed.\n");
	    }
		
	} while (askToContinue());

    std::cout << "Thanks for watching! Goodbye!\n";

    /////////////////////////////////////////////////////////////////////
    // Clean up
    /////////////////////////////////////////////////////////////////////

    FreeLibrary(hDLL);
    return 0;
}

