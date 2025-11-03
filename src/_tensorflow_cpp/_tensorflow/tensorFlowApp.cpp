/*

//
TOOLCHAIN : C:\msys64\uctr64\bin (CHANGE PATH)

// INSTALLERS
https://www.tensorflow.org/install/lang_c?hl=es-419


// NON STATIC - REFERENCE - INHERITANCE

g++ -std=c++20 -I"include" -L"lib" -shared -m64 -o TensorFlowAppCPP.dll tensorFlowApp.cpp -ltensorflow -lAlgorithm -Wl,--subsystem,windows -DALGORITHM_EXPORTS

// COMPILE FROM ROOT

g++ -std=c++20 -I"include" -L"lib" -shared -m64 -o "__dist/TensorFlowAppCPP.dll" "_tensorflow/tensorFlowApp.cpp" -ltensorflow -lAlgorithm -Wl,--subsystem,windows -DALGORITHM_EXPORTS   

*/


#include "../include/tensorFlowApp.h"


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


bool RunTicTacToeSelfPlay(TicTacToeResultOnline& result, int aiMode, double temperature) {
    if (aiMode == TENSORFLOW) {
        TensorFlowTicTacToe tf;
        if (!tf.LoadModel("tictactoe_tf_model")) {
            std::cerr << "❌ Failed to initialize TensorFlow model.\n";
            return false;
        }

        TicTacToe game;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> starter(0, 1);
        int turn = (starter(gen) == 0) ? 1 : -1;

        std::vector<int> moves;

        for (int i = 0; i < 9; ++i) {
            result.history[0][i] = game.board[i];
        }
        result.historyCount = 1;

        while (true) {
            int move = -1;

            float input[9];
            for (int i = 0; i < 9; ++i) input[i] = static_cast<float>(game.board[i]);
            if (!tf.PredictBestMove(input, move)) {
                std::cerr << "❌ Prediction failed!\n";
                return false;
            }


            if (move < 0 || move >= 9 || game.board[move] != 0) {
                auto valid = game.getValidMoves();
                if (valid.empty()) break;
                move = valid[0];
            }

            game.board[move] = turn;
            moves.push_back(move);

            if (result.historyCount < 10) {
                for (int i = 0; i < 9; ++i) {
                    result.history[result.historyCount][i] = game.board[i];
                }
                result.historyCount++;
            }

            int winner;
            if (game.isGameOver(winner)) {
                result.winner = winner;
                break;
            }
            turn = -turn;
        }

        for (int i = 0; i < 9; ++i) {
            result.finalBoard[i] = game.board[i];
            result.moves[i] = (i < static_cast<int>(moves.size())) ? moves[i] : -1;
        }
        result.moveCount = static_cast<int>(moves.size());
        
        return true;
    }

    NeuralNetworkTicTacToe net(9, 18, 9);
    const std::string modelFile = "tictactoe_model.txt";

    if (aiMode != MINIMAX && !net.loadModel(modelFile)) {
     	//
        for (int i = 0; i < 5000; ++i) 
				trainStep(net);
        //
		net.saveModel(modelFile);
    }

    TicTacToe game;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> starter(0, 1);
    int turn = (starter(gen) == 0) ? 1 : -1;

    std::vector<int> moves;

    for (int i = 0; i < 9; ++i) {
        result.history[0][i] = game.board[i];
    }
    result.historyCount = 1;

    while (true) {
        int move = -1;

        if (aiMode == MINIMAX) {
            move = minimaxMove(game.board, turn);
        } else {
            std::vector<double> input = boardToInput(game.board);
            net.forward(input);
            move = selectMove(net.output, game, aiMode, temperature);
        }

        if (move < 0 || move >= 9 || game.board[move] != 0) {
            auto valid = game.getValidMoves();
            if (valid.empty()) break;
            move = valid[0];
        }

        game.board[move] = turn;
        moves.push_back(move);

        if (result.historyCount < 10) {
            for (int i = 0; i < 9; ++i) {
                result.history[result.historyCount][i] = game.board[i];
            }
            result.historyCount++;
        }

        int winner;
        if (game.isGameOver(winner)) {
            result.winner = winner;
            break;
        }
        turn = -turn;
    }

    for (int i = 0; i < 9; ++i) {
        result.finalBoard[i] = game.board[i];
        result.moves[i] = (i < static_cast<int>(moves.size())) ? moves[i] : -1;
    }
    result.moveCount = static_cast<int>(moves.size());

    return true;
}

/////////////////////////////////////////////////////////////////////
// DLL ENTRY POINTS
/////////////////////////////////////////////////////////////////////

DLL_EXPORT bool PlayTicTacToeGameWithHistory(TicTacToeResultOnline* result, int aiMode, double temperature) 
{
    try {
    	//
        if (!result) return false;
        //
        return RunTicTacToeSelfPlay(*result, aiMode, temperature);
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
