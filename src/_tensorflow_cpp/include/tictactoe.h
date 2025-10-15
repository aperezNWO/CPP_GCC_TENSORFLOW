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

