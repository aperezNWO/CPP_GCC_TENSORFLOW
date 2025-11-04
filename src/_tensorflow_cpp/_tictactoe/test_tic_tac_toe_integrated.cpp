/*

	execute from root :
	
	g++ -std=c++20  -I"include" -L"lib" -o "__test/test_tic_tac_toe_integrated.exe"  "_tictactoe/test_tic_tac_toe_integrated.cpp" -ltensorflow -m64 -Wl,--subsystem,console
	
*/

#include "../include/tictactoe.h"
//#include "../include/tensorflow/c/c_api.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <thread>
#include <chrono>

#ifdef _WIN32
    #include <conio.h>
    #include <windows.h>
    #define CLEAR_SCREEN() system("cls")
#else
    #include <unistd.h>
    #define CLEAR_SCREEN() system("clear")
#endif

/*
// ======================================
// TensorFlow Integration
// ======================================

class TensorFlowTicTacToe {
public:
    TF_Graph* graph = nullptr;
    TF_Session* session = nullptr;
    TF_Status* status = nullptr;

    bool LoadModel(const char* export_dir) {
        if (session) return true; // Already loaded

        status = TF_NewStatus();
        graph = TF_NewGraph();

        TF_SessionOptions* opts = TF_NewSessionOptions();
        TF_Buffer* meta_graph_def = TF_NewBuffer();

        const char* tags = "serve";
        int ntags = 1;

        std::cout << "🔍 Loading SavedModel from: " << export_dir << "\n";

        session = TF_LoadSessionFromSavedModel(opts, nullptr,
                                               export_dir,
                                               &tags, ntags,
                                               graph, meta_graph_def, status);

        TF_DeleteSessionOptions(opts);
        TF_DeleteBuffer(meta_graph_def);

        if (TF_GetCode(status) != TF_OK) {
            std::cerr << "❌ Failed to load model: " << TF_Message(status) << "\n";
            return false;
        }

        std::cout << "✅ Model loaded successfully.\n";
        return true;
    }

    bool PredictBestMove(const float input_board[9], int& best_move) {
        if (!session) {
            std::cerr << "❌ Session not initialized.\n";
            return false;
        }

        // Input tensor: [1, 9]
        int64_t input_dims[] = {1, 9};
        TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, input_dims, 2, sizeof(float) * 9);
        float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
        for (int i = 0; i < 9; ++i) {
            input_data[i] = input_board[i];
        }

        // Output tensor
        int64_t output_dims[] = {1, 9};
        TF_Tensor* output_tensor = nullptr;

        const char* input_name = "serving_default_dense_input";   // Adjust if needed
        const char* output_name = "StatefulPartitionedCall";      // Adjust if needed

        TF_Output inputs[1] = {{TF_GraphOperationByName(graph, input_name), 0}};
        TF_Output outputs[1] = {{TF_GraphOperationByName(graph, output_name), 0}};

        if (!inputs[0].oper || !outputs[0].oper) {
            std::cerr << "❌ Invalid tensor name!\n";
            std::cerr << "   Input '" << input_name << "': " << (inputs[0].oper ? "FOUND" : "NOT FOUND") << "\n";
            std::cerr << "   Output '" << output_name << "': " << (outputs[0].oper ? "FOUND" : "NOT FOUND") << "\n";
            std::cerr << "💡 Use Netron.app to inspect saved_model.pb and get correct names.\n";
            TF_DeleteTensor(input_tensor);
            return false;
        }

        TF_SessionRun(session,
                      nullptr,
                      inputs, &input_tensor, 1,
                      outputs, &output_tensor, 1,
                      nullptr, 0,
                      nullptr, status);

        if (TF_GetCode(status) != TF_OK) {
            std::cerr << "❌ Inference failed: " << TF_Message(status) << "\n";
            TF_DeleteTensor(input_tensor);
            return false;
        }

        float* probs = static_cast<float*>(TF_TensorData(output_tensor));

        // Find valid moves
        std::vector<int> validMoves;
        for (int i = 0; i < 9; ++i) {
            if (input_board[i] == 0.0f) {
                validMoves.push_back(i);
            }
        }

        if (validMoves.empty()) {
            TF_DeleteTensor(input_tensor);
            TF_DeleteTensor(output_tensor);
            return false;
        }

        // === SOFTMAX SAMPLING WITH TEMPERATURE ===
        const float temperature = 1.5f; // >1.0 = more random, <1.0 = more greedy
        std::vector<float> exp_probs;
        float sum = 0.0f;

        for (int idx : validMoves) {
            float p = std::exp(probs[idx] / temperature);
            exp_probs.push_back(p);
            sum += p;
        }

        // Normalize
        for (float& p : exp_probs) p /= sum;

        // Sample move
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(exp_probs.begin(), exp_probs.end());

        best_move = validMoves[dist(gen)];

        TF_DeleteTensor(input_tensor);
        TF_DeleteTensor(output_tensor);
        return true;
    }

    ~TensorFlowTicTacToe() {
        if (session) TF_CloseSession(session, status);
        if (session) TF_DeleteSession(session, status);
        if (graph) TF_DeleteGraph(graph);
        if (status) TF_DeleteStatus(status);
    }
};

*/

// ======================================
// Game Logic with TensorFlow Integration
// ======================================

bool __RunTicTacToeSelfPlay(TicTacToeResultOnline& result, int aiMode, double temperature) {
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

            if (aiMode == TENSORFLOW) {
                float input[9];
                for (int i = 0; i < 9; ++i) input[i] = static_cast<float>(game.board[i]);
                if (!tf.PredictBestMove(input, move)) {
                    std::cerr << "❌ Prediction failed!\n";
                    return false;
                }
            } else {
                std::vector<double> input = boardToInput(game.board);
                NeuralNetworkTicTacToe net(9, 18, 9);
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

    NeuralNetworkTicTacToe net(9, 18, 9);
    const std::string modelFile = "tictactoe_model.txt";

    if (aiMode != MINIMAX && !net.loadModel(modelFile)) {
        std::cout << "[Training] No model found. Training 5000 games...\n";
        for (int i = 0; i < 5000; ++i) trainStep(net);
        net.saveModel(modelFile);
        std::cout << "[Saved] Model saved to '" << modelFile << "'\n";
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

void animateGame(const TicTacToeResultOnline& result, int aiMode) {
    const char* modeNames[] = {"Expert", "Creative", "Minimax", "Random", "TensorFlow"};
    CLEAR_SCREEN();
    std::cout << "=== TIC-TAC-TOE ANIMATED GAME ===\n";
    std::cout << "AI Mode: " << modeNames[aiMode] << "\n";

    if (result.moveCount > 0) {
        int firstMove = result.moves[0];
        int starter = (result.history[1][firstMove] == 1) ? 1 : -1;
        std::cout << "Starting Player: " << (starter == 1 ? "X" : "O") << "\n";
    }
    std::cout << "\n";

    for (int step = 0; step < result.historyCount; ++step) {
        CLEAR_SCREEN();
        std::cout << "=== TIC-TAC-TOE ANIMATED GAME ===\n";
        std::cout << "AI Mode: " << modeNames[aiMode] << "\n";

        if (step == 0 && result.moveCount > 0) {
            int firstMove = result.moves[0];
            int starter = (result.history[1][firstMove] == 1) ? 1 : -1;
            std::cout << "Starting Player: " << (starter == 1 ? "X" : "O") << "\n";
        }
        std::cout << "\n";

        TicTacToe temp;
        for (int i = 0; i < 9; ++i) temp.board[i] = result.history[step][i];
        temp.print();

        if (step > 0) {
            int movePos = result.moves[step - 1];
            int player = (step % 2 == 1) ? 1 : -1;
            if (step == 1) player = (result.history[1][movePos] == 1) ? 1 : -1;
            std::cout << "Player " << (player == 1 ? "X" : "O")
                      << " plays at position " << movePos << "\n\n";
        }

#ifdef _WIN32
        Sleep(600);
#else
        usleep(600000);
#endif
    }

    std::cout << "Game Over! Winner: ";
    if (result.winner == 1) std::cout << "X\n";
    else if (result.winner == -1) std::cout << "O\n";
    else std::cout << "Draw\n";

    std::cout << "\nMove sequence: ";
    for (int i = 0; i < result.moveCount; ++i) {
        std::cout << result.moves[i] << " ";
    }
    std::cout << "\n";
}

int getUserChoice() {
    std::cout << "\n🎮 TIC-TAC-TOE AI DEMO 🎮\n";
    std::cout << "Choose an option:\n";
    std::cout << "0 - Expert AI (Neural Net)\n";
    std::cout << "1 - Creative AI (Probabilistic)\n";
    std::cout << "2 - Deterministic - Minimax (Perfect Play)\n";
    std::cout << "3 - Random Player\n";
    std::cout << "4 - TensorFlow Model (Under Construction)\n";
    std::cout << "5 - Exit Program\n";
    std::cout << "Enter choice (0-5): ";

    int choice;
    std::cin >> choice;
    return choice;
}

// ======================================
// Main Functionality
// ======================================

void runMultipleGames(int aiMode) {
    const char* modeNames[] = {"Expert", "Creative", "Minimax", "Random", "TensorFlow"};

    if (aiMode == TENSORFLOW) {
        CLEAR_SCREEN();
        std::cout << "🧠 TensorFlow Model Integration\n";
        std::cout << "================================\n\n";
        std::cout << "Starting TensorFlow-powered self-play...\n";
    }

    double temperature = (aiMode == 1) ? 1.5 : 0.1;

    std::cout << "\n🎮 Starting '" << modeNames[aiMode] << "' mode...\n";
    std::cout << "Games will continue until you choose to stop.\n";

    int winsX = 0, winsO = 0, draws = 0;
    int gameCount = 0;

    char choice;
    do {
        CLEAR_SCREEN();
        std::cout << "🎮 Playing game " << (gameCount + 1) << " as '" << modeNames[aiMode] << "'...\n\n";

        TicTacToeResultOnline result{};
        if (__RunTicTacToeSelfPlay(result, aiMode, temperature)) {
            animateGame(result, aiMode);

            if (result.winner == 1) ++winsX;
            else if (result.winner == -1) ++winsO;
            else ++draws;
            ++gameCount;
        } else {
            std::cout << "❌ Game failed to run.\n";
        }

        std::cout << "\n🔁 Play another game with the same AI style? (y/n): ";
        std::cin >> choice;
        choice = std::tolower(static_cast<unsigned char>(choice));
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    } while (choice == 'y');

    if (gameCount > 0) {
        std::cout << "\n📊 Session Summary (" << gameCount << " games):\n";
        std::cout << "X Wins: " << winsX << " (" << (100.0 * winsX / gameCount) << "%)\n";
        std::cout << "O Wins: " << winsO << " (" << (100.0 * winsO / gameCount) << "%)\n";
        std::cout << "Draws:  " << draws  << " (" << (100.0 * draws  / gameCount) << "%)\n";
    } else {
        std::cout << "\n🚫 No games were completed.\n";
    }

    std::cout << "Press Enter to return to main menu...";
    std::cin.get();
}

// ====================
// Main Entry Point
// ====================

void executeConsole() {
    std::cout << "🧠 Welcome to Tic-Tac-Toe AI Demo!\n";

    while (true) {
        int choice = getUserChoice();

        if (choice == 5) {
            std::cout << "👋 Thank you for playing! Goodbye!\n";
            break;
        }

        if (choice >= 0 && choice <= 4) {
            runMultipleGames(choice);
        } else {
            std::cout << "❌ Invalid choice. Please enter 0–5.\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
}

int main() {
    executeConsole();
    return 0;
}
