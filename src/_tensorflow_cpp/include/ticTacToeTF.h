#ifndef TICTACTOETF_H // include guard
#define TICTACTOETF_H
#endif

#include <tensorflow/c/c_api.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <thread>
#include <chrono>

// ======================================
// Data Structure: Full Game Result
// ======================================

struct TicTacToeMove {
    int position;
    int player; // 1 = X, -1 = O
};

struct TicTacToeGameResult {
    int finalBoard[9];
    std::vector<TicTacToeMove> moves;
    int winner;
    std::vector<std::vector<int>> boardHistory;
    bool success;
};

// ======================================
// TensorFlow Model Loader & Predictor
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

        std::cout << "?? Loading SavedModel from: " << export_dir << "\n";

        session = TF_LoadSessionFromSavedModel(opts, nullptr,
                                               export_dir,
                                               &tags, ntags,
                                               graph, meta_graph_def, status);

        TF_DeleteSessionOptions(opts);
        TF_DeleteBuffer(meta_graph_def);

        if (TF_GetCode(status) != TF_OK) {
            std::cerr << "? Failed to load model: " << TF_Message(status) << "\n";
            return false;
        }

        std::cout << "? Model loaded successfully.\n";
        return true;
    }

    bool PredictBestMove(const float input_board[9], int& best_move) {
        if (!session) {
            std::cerr << "? Session not initialized.\n";
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

        // ?? Use Netron.app to verify these names!
        const char* input_name = "serving_default_dense_input";   // ? Adjust if needed
        const char* output_name = "StatefulPartitionedCall";      // ? Adjust if needed

        TF_Output inputs[1] = {{TF_GraphOperationByName(graph, input_name), 0}};
        TF_Output outputs[1] = {{TF_GraphOperationByName(graph, output_name), 0}};

        if (!inputs[0].oper || !outputs[0].oper) {
            std::cerr << "? Invalid tensor name!\n";
            std::cerr << "   Input '" << input_name << "': " << (inputs[0].oper ? "FOUND" : "NOT FOUND") << "\n";
            std::cerr << "   Output '" << output_name << "': " << (outputs[0].oper ? "FOUND" : "NOT FOUND") << "\n";
            std::cerr << "?? Use Netron.app to inspect saved_model.pb and get correct names.\n";
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
            std::cerr << "? Inference failed: " << TF_Message(status) << "\n";
            TF_DeleteTensor(input_tensor);
            return false;
        }

        float* probs = static_cast<float*>(TF_TensorData(output_tensor));

        // ? Debug: Print raw output
        std::cout << "?? Model output: ";
        for (int i = 0; i < 9; ++i) {
            std::cout << probs[i] << " ";
        }
        std::cout << "\n";

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


// ======================================
// Animation: Paced Board Display
// ======================================

void PrintBoard(const int* board) {
    for (int i = 0; i < 9; ++i) {
        char c = '.';
        if (board[i] == 1) c = 'X';
        else if (board[i] == -1) c = 'O';
        std::cout << c;
        if ((i+1) % 3 == 0) std::cout << '\n';
    }
    std::cout << "---\n";
}

void AnimateGame(const TicTacToeGameResult& result) {
    std::cout << "\n=== REPLAYING GAME: X vs O ===\n";

    for (size_t step = 0; step < result.boardHistory.size(); ++step) {
#ifdef _WIN32
        system("cls");
#else
        system("clear");
#endif

        std::cout << "\n=== MOVE " << step << " ===\n";
        PrintBoard(result.boardHistory[step].data());

        if (step > 0) {
            const auto& move = result.moves[step - 1];
            std::cout << "Player " << (move.player == 1 ? 'X' : 'O')
                      << " plays at position " << move.position << "\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(800));
    }

    if (result.winner == 1)      std::cout << "?? X wins!\n";
    else if (result.winner == -1) std::cout << "?? O wins!\n";
    else                          std::cout << "?? It's a draw!\n";
}

