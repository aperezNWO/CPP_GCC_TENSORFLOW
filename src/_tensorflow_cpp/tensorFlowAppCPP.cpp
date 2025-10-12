/*

//
TOOLCHAIN : C:\msys64\uctr64\bin (CHANGE PATH)

// INSTALLERS
https://www.tensorflow.org/install/lang_c?hl=es-419


// UNABLE TO COMPILE AS STATIC
gcc -I"include" -L"lib" -shared -static -static-libgcc -static-libstdc++ -m64 -o TensorFlowAppC.dll tf_dll_gen.c -ltensorflow -Wl,--subsystem,console 

// COMPILE OK - NON STATIC

g++ -I"include" -L"lib" -shared -m64 -o TensorFlowAppCPP.dll tensorFlowAppCPP.cpp -ltensorflow  -Wl,--subsystem,windows 

// NON STATIC - REFERENCE - INHERITANCE

g++ -std=c++20 -I"include" -L"lib" -shared -m64 -o TensorFlowAppCPP.dll tensorFlowAppCPP.cpp -ltensorflow -lAlgorithm -Wl,--subsystem,windows -DALGORITHM_EXPORTS

3) UTILIZAR PROYECDTO CPP_GCC_TENSORFLOW.DEV (Embarcadero Dev C++) PROVISIONALMENTE PARA 
   
   A) VISUALIZAR Y EDITAR ARCHIVOS.
   B) COMPILAR CON LINEA DE COMANDOS.
   C) EJECUTAR COMANDOS GIT
   
*/


#include "tensorFlowApp.h"
#include "ticTacToeAIAppCpp.h"
#include "ticTAcToeTF.h"

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

//
bool TensorFlowApp::RunTicTacToeSelfPlayOnline(TicTacToeResultOnline& result) {
    NeuralNetwork net(9, 18, 9);
    const std::string modelFile = "tictactoe_model.txt";

    if (!net.loadModel(modelFile)) {
        std::cout << "[Training] No model found. Training 5000 games...\n";
        for (int i = 0; i < 5000; ++i) trainStep(net);
        net.saveModel(modelFile);
        std::cout << "[Saved] Model saved to '" << modelFile << "'\n";
    }

    TicTacToe game;
    int turn = 1;
    std::vector<int> moves;
    std::vector<std::vector<int>> boardHistory;

    // Save initial state
    boardHistory.push_back(game.board);

    while (true) {
        std::vector<double> input = boardToInput(game.board);
        net.forward(input);
        int move = selectMoveWithSoftmax(net.output, game); // Non-deterministic

        game.board[move] = turn;
        moves.push_back(move);

        // Save board state after move
        boardHistory.push_back(game.board);

        int winner;
        if (game.isGameOver(winner)) {
            result.winner = winner;
            break;
        }
        turn = -turn;
    }

    // Copy final board
    for (int i = 0; i < 9; ++i) {
        result.finalBoard[i] = game.board[i];
        result.moves[i] = (i < static_cast<int>(moves.size())) ? moves[i] : -1;
    }
    result.moveCount = static_cast<int>(moves.size());

    // Copy history
    result.historyCount = static_cast<int>(boardHistory.size());
    for (int s = 0; s < result.historyCount && s < 10; ++s) {
        for (int i = 0; i < 9; ++i) {
            result.history[s][i] = boardHistory[s][i];
        }
    }

    return true;
}

//
/*
bool TensorFlowApp::PlaySelfPlayGameWithHistory(TicTacToeGameResult& result) {
    
    result.success = false;

    std::vector<int> board(9, 0); // 0=empty, 1=X, -1=O
    int turn = 1; // X starts
    std::vector<int> moveHistory;

    // Save initial state
    result.boardHistory.push_back(board);

    TensorFlowTicTacToe tf;
    if (!tf.LoadModel("tictactoe_tf_model")) {
        std::cerr << "❌ Failed to initialize TensorFlow model.\n";
        return result;
    }

    while (true) {
        // Convert to float input
        float input[9];
        for (int i = 0; i < 9; ++i) input[i] = static_cast<float>(board[i]);

        int move;
        if (!tf.PredictBestMove(input, move)) {
            std::cerr << "❌ Prediction failed!\n";
            return result;
        }

        // Make move
        board[move] = turn;
        moveHistory.push_back(move);
        result.boardHistory.push_back(board);

        // Check win/draw
        auto isGameOver = [](const std::vector<int>& b, int& winner) -> bool {
            const int wins[8][3] = {
                {0,1,2}, {3,4,5}, {6,7,8},
                {0,3,6}, {1,4,7}, {2,5,8},
                {0,4,8}, {2,4,6}
            };
            for (auto& w : wins) {
                if (b[w[0]] != 0 && b[w[0]] == b[w[1]] && b[w[1]] == b[w[2]]) {
                    winner = b[w[0]];
                    return true;
                }
            }
            if (std::find(b.begin(), b.end(), 0) == b.end()) {
                winner = 0;
                return true;
            }
            winner = 0;
            return false;
        };

        int winner;
        if (isGameOver(board, winner)) {
            result.winner = winner;
            break;
        }

        turn = -turn;
    }

    // Copy final data
    for (int i = 0; i < 9; ++i) result.finalBoard[i] = board[i];
    for (size_t i = 0; i < moveHistory.size(); ++i) {
        result.moves.push_back({ moveHistory[i], (i % 2 == 0) ? 1 : -1 });
    }
    result.success = true;
    
	//
	return result.succes;
}
*/

/////////////////////////////////////////////////////////////////////
// DLL ENTRY POINTS
/////////////////////////////////////////////////////////////////////
/*
DLL_EXPORT bool PlayTicTacToeGameWithTensorFlow(TicTacToeGameResult* result)
{
    try {
        static TensorFlowApp app;
        if (!app.PlaySelfPlayGameWithHistory(*result)) return false;
        return true;
    } catch (...) {
        return false;
    }
}
*/

DLL_EXPORT bool PlayTicTacToeGameWithHistory(TicTacToeResultOnline* result) {
    try {
        static TensorFlowApp app;
        if (!app.RunTicTacToeSelfPlayOnline(*result)) return false;
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
