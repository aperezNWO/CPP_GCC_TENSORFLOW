/*

//
TOOLCHAIN : C:\msys64\uctr64\bin (CHANGE PATH)

// INSTALLERS
https://www.tensorflow.org/install/lang_c?hl=es-419


// UNABLE TO COMPILE AS STATIC
gcc -I"include" -L"lib" -shared -static -static-libgcc -static-libstdc++ -m64 -o TensorFlowAppC.dll tf_dll_gen.c -ltensorflow -Wl,--subsystem,console 

// COMPILE OK - NON STATIC

g++ -I"include" -L"lib" -shared -m64 -o TensorFlowAppCPP.dll tensorFlowApp.cpp -ltensorflow  -Wl,--subsystem,windows 

// NON STATIC - REFERENCE - INHERITANCE

g++ -std=c++20 -I"include" -L"lib" -shared -m64 -o TensorFlowAppCPP.dll tensorFlowApp.cpp -ltensorflow -lAlgorithm -Wl,--subsystem,windows -DALGORITHM_EXPORTS

3) UTILIZAR PROYECDTO CPP_GCC_TENSORFLOW.DEV (Embarcadero Dev C++) PROVISIONALMENTE PARA 
   
   A) VISUALIZAR Y EDITAR ARCHIVOS.
   B) COMPILAR CON LINEA DE COMANDOS.
   C) EJECUTAR COMANDOS GIT
   
*/


#include "tensorFlowApp.h"
#include "chessAIAppCpp.h"

//
TensorFlowApp::TensorFlowApp(): Algorithm(false)
{
     //
     this->ReadConfigFile("tensorflow.ini");
}
//
TensorFlowApp::~TensorFlowApp()
{
    //
}


//
const char*  TensorFlowApp::GetTensorFlowAPIVersion()
{
	//
  	return TF_Version(); // Return the TensorFlow version directly;
}

//
std::string TensorFlowApp::GetTensorFlowAppVersion()
{
    auto it = this->configMap.find("DLL_VERSION");
    if (it != this->configMap.end()) {
        return it->second;
    }
    return "UNKNOWN"; 
}

bool TensorFlowApp::RunTicTacToeSelfPlay(TicTacToeResult& result) {
	//
    NeuralNetwork net(9, 18, 9);
    const std::string modelFile = "tictactoe_model.txt";

    if (!net.loadModel(modelFile)) {
        std::cout << "[Training] No model found. Training 5000 games...\n";
        for (int i = 0; i < 5000; ++i) trainStep(net);
        net.saveModel(modelFile);
        std::cout << "[Saved] Model saved to '" << modelFile << "'\n";
    }

    // Play one game
    TicTacToe game;
    int turn = 1;
    std::vector<int> moves;

    while (true) {
        std::vector<double> input = boardToInput(game.board);
        net.forward(input);
        
		//int move = selectMove(net.output, game);
        int move   = selectMoveWithSoftmax(net.output, game);
        
        game.board[move] = turn;
        moves.push_back(move);


        //CLEAR_SCREEN(); 
        system("cls");
        
        /*
		for (int i = 0; i < 9; ++i) {
            printf("%c%c", " XO."[game.board[i] + 1], (i+1) % 3 == 0 ? '\n' : ' ');
        }*/
        
        game.print();
        
        waitForEnter();

        int winner;
        if (game.isGameOver(winner)) {
            result.winner = winner;
            break;
        }
        turn = -turn;
    }
	
    // Copy results
    for (int i = 0; i < 9; ++i) {
        result.board[i] = game.board[i];
        result.moves[i] = (i < static_cast<int>(moves.size())) ? moves[i] : -1;
    }
    result.moveCount = static_cast<int>(moves.size());
	
    return true;
}


/////////////////////////////////////////////////////////////////////
// DLL ENTRY POINTS
/////////////////////////////////////////////////////////////////////

DLL_EXPORT bool PlayTicTacToeGame(int* boardOut, int* movesOut, int* winnerOut, int* moveCountOut) {
    try {
        static TensorFlowApp app;
        TicTacToeResult result{};
        if (!app.RunTicTacToeSelfPlay(result)) return false;

        for (int i = 0; i < 9; ++i) {
            boardOut[i] = result.board[i];
            movesOut[i] = result.moves[i];
        }
        *winnerOut = result.winner;
        *moveCountOut = result.moveCount;

        return true;
    } catch (...) {
        return false;
    }
}

DLL_EXPORT const char* GetTensorFlowAPIVersion() 
{
    static std::string versionCache;
    {
        std::unique_ptr<TensorFlowApp> app = std::make_unique<TensorFlowApp>();
        versionCache = app->GetTensorFlowAPIVersion(); // TF_Version()
    }
    return versionCache.c_str();
}

DLL_EXPORT const char* GetTensorFlowAppVersion() {
    static std::string version;
    if (version.empty()) {
        TensorFlowApp app;
        version = app.GetTensorFlowAppVersion();
    }
    return version.c_str();
}

DLL_EXPORT const char* GetCPPSTDVersion()
{
    static std::string version;
    if (version.empty()) {
        TensorFlowApp app;
        version = app.GetCPPSTDVersion(__cplusplus);
    }
    return version.c_str();
}
