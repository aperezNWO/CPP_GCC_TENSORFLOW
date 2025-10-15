// tictactoe.h
#ifndef TICTACTOE_H // include guard
#define TICTACTOE_H
#endif
#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <limits>
#include <cmath>
#include <string>
#include <iostream>

#include "tensorflow/c/c_api.h"

// ----------------------------
// C-Style Export Types
// ----------------------------

extern "C" {
    typedef struct {
        int finalBoard[9];
        int moves[9];              // -1 if not used
        int winner;
        int moveCount;

        // NEW: Game history (for animation)
        int history[10][9];        // Up to 10 states
        int historyCount;
    } TicTacToeResultOnline;
}

// ----------------------------
// Game Logic: TicTacToe Board
// ----------------------------

struct TicTacToe {
    int board[9] = {0}; // 0=empty, 1=X, -1=O

    void reset() {
        for (int i = 0; i < 9; ++i) board[i] = 0;
    }

    void print() const {
        for (int i = 0; i < 9; ++i) {
            std::cout << " ";
            if (board[i] == 1)      std::cout << "X";
            else if (board[i] == -1) std::cout << "O";
            else                    std::cout << " ";

            if (i % 3 != 2) std::cout << " |";
            if (i % 3 == 2 && i < 8) std::cout << "\n-----------\n";
        }
        std::cout << "\n\n";
    }

    bool isGameOver(int& winner) const {
        const int wins[8][3] = {
            {0,1,2}, {3,4,5}, {6,7,8},
            {0,3,6}, {1,4,7}, {2,5,8},
            {0,4,8}, {2,4,6}
        };

        for (const auto& w : wins) {
            if (board[w[0]] != 0 &&
                board[w[0]] == board[w[1]] &&
                board[w[1]] == board[w[2]]) {
                winner = board[w[0]];
                return true;
            }
        }

        for (int i = 0; i < 9; ++i) {
            if (board[i] == 0) return false;
        }
        winner = 0;
        return true;
    }

    std::vector<int> getValidMoves() const {
        std::vector<int> moves;
        for (int i = 0; i < 9; ++i) {
            if (board[i] == 0) moves.push_back(i);
        }
        return moves;
    }
};

// ----------------------------
// Neural Network Stub
// ----------------------------

class NeuralNetworkTicTacToe {
public:
    std::vector<double> output;
    NeuralNetworkTicTacToe(int, int, int) : output(9, 0.0) {}
    void forward(const std::vector<double>& x) { output = x; }
    bool loadModel(const std::string&) { return true; }
    bool saveModel(const std::string&) { return true; }
};

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

//-----------------------------
// utilities
//-----------------------------

std::vector<double> boardToInput(const int board[9]) {
    std::vector<double> input(9);
    for (int i = 0; i < 9; ++i) {
        input[i] = static_cast<double>(board[i]);
    }
    return input;
}

// ----------------------------
// Move Selection Strategies
// ----------------------------

enum AIMode {
    EXPERT = 0,
    CREATIVE = 1,
    MINIMAX = 2,
    RANDOM = 3,
    TENSORFLOW = 4  // ← Placeholder
};

std::vector<double> softmax(const std::vector<double>& logits, double temp) {
    std::vector<double> probs(9);
    double maxVal = *std::max_element(logits.begin(), logits.end());

    double sum = 0.0;
    for (int i = 0; i < 9; ++i) {
        double expVal = std::exp((logits[i] - maxVal) / temp);
        probs[i] = expVal;
        sum += expVal;
    }
    for (double& p : probs) p /= sum;
    return probs;
}

int selectGreedy(const std::vector<double>& scores, const TicTacToe& game) {
    std::vector<std::pair<double, int>> candidates;
    for (int i = 0; i < 9; ++i) {
        if (game.board[i] == 0) {
            candidates.emplace_back(scores[i], i);
        }
    }
    if (candidates.empty()) return -1;
    return std::max_element(candidates.begin(), candidates.end())->second;
}

int selectSampled(const std::vector<double>& probs, const TicTacToe& game) {
    std::vector<int> valid;
    std::vector<double> weights;
    for (int i = 0; i < 9; ++i) {
        if (game.board[i] == 0) {
            valid.push_back(i);
            weights.push_back(probs[i]);
        }
    }
    if (valid.empty()) return -1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(weights.begin(), weights.end());
    return valid[dist(gen)];
}

int selectRandomMove(const TicTacToe& game) {
    auto valid = game.getValidMoves();
    if (valid.empty()) return -1;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, static_cast<int>(valid.size()) - 1);
    return valid[dist(gen)];
}

// ----------------------------
// Deterministic Minimax AI
// ----------------------------

int minimax(const int board[9], int depth, bool isMaximizing, int alpha = -1000, int beta = 1000) {
    const int wins[8][3] = {
        {0,1,2}, {3,4,5}, {6,7,8},
        {0,3,6}, {1,4,7}, {2,5,8},
        {0,4,8}, {2,4,6}
    };

    for (const auto& w : wins) {
        if (board[w[0]] != 0 &&
            board[w[0]] == board[w[1]] &&
            board[w[1]] == board[w[2]]) {
            return board[w[0]] == 1 ? 10 - depth : depth - 10;
        }
    }

    bool isFull = true;
    for (int i = 0; i < 9; ++i) {
        if (board[i] == 0) {
            isFull = false;
            break;
        }
    }
    if (isFull) return 0;

    if (isMaximizing) {
        int best = -1000;
        for (int i = 0; i < 9; ++i) {
            if (board[i] == 0) {
                int temp[9];
                for (int j = 0; j < 9; ++j) temp[j] = board[j];
                temp[i] = 1;
                int score = minimax(temp, depth + 1, false, alpha, beta);
                best = std::max(best, score);
                alpha = std::max(alpha, score);
                if (beta <= alpha) break;
            }
        }
        return best;
    } else {
        int best = 1000;
        for (int i = 0; i < 9; ++i) {
            if (board[i] == 0) {
                int temp[9];
                for (int j = 0; j < 9; ++j) temp[j] = board[j];
                temp[i] = -1;
                int score = minimax(temp, depth + 1, true, alpha, beta);
                best = std::min(best, score);
                beta = std::min(beta, score);
                if (beta <= alpha) break;
            }
        }
        return best;
    }
}

int minimaxMove(const int board[9], int player) {
    int bestMove = -1;
    int bestValue = (player == 1) ? -1000 : 1000;

    for (int i = 0; i < 9; ++i) {
        if (board[i] == 0) {
            int temp[9];
            for (int j = 0; j < 9; ++j) temp[j] = board[j];
            temp[i] = player;

            int moveValue = minimax(temp, 0, player == -1);

            if (player == 1 && moveValue > bestValue) {
                bestValue = moveValue;
                bestMove = i;
            }
            if (player == -1 && moveValue < bestValue) {
                bestValue = moveValue;
                bestMove = i;
            }
        }
    }

    if (bestMove == -1) {
        for (int i = 0; i < 9; ++i) {
            if (board[i] == 0) return i;
        }
    }
    return bestMove;
}

// ----------------------------
// Main Move Selector
// ----------------------------

int selectMove(const std::vector<double>& output, const TicTacToe& game, int aiMode, double temperature) {
    if (aiMode == RANDOM) {
        return selectRandomMove(game);
    }

    if (aiMode == MINIMAX || aiMode == TENSORFLOW) {
        return -1;
    }

    auto probs = softmax(output, temperature);

    switch (aiMode) {
        case EXPERT:
            return selectGreedy(probs, game);
        case CREATIVE:
            return selectSampled(probs, game);
        default:
            return selectGreedy(probs, game);
    }
}

void trainStep(NeuralNetworkTicTacToe& net) { /* Simulate */ }

