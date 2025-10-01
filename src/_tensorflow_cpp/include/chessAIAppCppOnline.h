#ifndef CHESSAIAPPCPPONLINE_H // include guard
#define CHESSAIAPPCPPONLINE_H
#endif



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


// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);
std::uniform_int_distribution<> moveDis(0, 8);

// Sigmoid function
double sigmoid(double x) {
    x = std::max(-700.0, std::min(700.0, x));
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of sigmoid
double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// Forward declare internal classes
class NeuralNetwork;
class TicTacToe;

// ======================================
// Internal: NeuralNetwork Class
// ======================================

class NeuralNetwork {
public:
    std::vector<double> input, hidden, output;
    std::vector<std::vector<double>> weights_ih, weights_ho;
    std::vector<double> bias_h, bias_o;
    double learningRate = 0.1;

    NeuralNetwork(int inputSize = 9, int hiddenSize = 18, int outputSize = 9)
        : input(inputSize), hidden(hiddenSize), output(outputSize),
          weights_ih(hiddenSize, std::vector<double>(inputSize)),
          weights_ho(outputSize, std::vector<double>(hiddenSize)),
          bias_h(hiddenSize), bias_o(outputSize) {
        randomize();
    }

    void randomize() {
        for (auto& row : weights_ih)
            for (double& w : row)
                w = dis(gen) * 2.0 - 1.0;

        for (auto& row : weights_ho)
            for (double& w : row)
                w = dis(gen) * 2.0 - 1.0;

        for (double& b : bias_h)
            b = dis(gen) * 2.0 - 1.0;

        for (double& b : bias_o)
            b = dis(gen) * 2.0 - 1.0;
    }

    void forward(const std::vector<double>& x) {
        input = x;
        for (int i = 0; i < hidden.size(); ++i) {
            double activation = bias_h[i];
            for (int j = 0; j < input.size(); ++j)
                activation += weights_ih[i][j] * input[j];
            hidden[i] = sigmoid(activation);
        }

        for (int i = 0; i < output.size(); ++i) {
            double activation = bias_o[i];
            for (int j = 0; j < hidden.size(); ++j)
                activation += weights_ho[i][j] * hidden[j];
            output[i] = sigmoid(activation);
        }
    }

    void backprop(const std::vector<double>& target) {
        std::vector<double> grad_output(output.size());
        for (int i = 0; i < output.size(); ++i) {
            double error = target[i] - output[i];
            grad_output[i] = error * sigmoidDerivative(output[i]);
        }

        std::vector<double> grad_hidden(hidden.size(), 0);
        for (int i = 0; i < hidden.size(); ++i) {
            for (int j = 0; j < output.size(); ++j)
                grad_hidden[i] += grad_output[j] * weights_ho[j][i];
            grad_hidden[i] *= sigmoidDerivative(hidden[i]);
        }

        for (int i = 0; i < output.size(); ++i) {
            bias_o[i] += learningRate * grad_output[i];
            for (int j = 0; j < hidden.size(); ++j)
                weights_ho[i][j] += learningRate * grad_output[i] * hidden[j];
        }

        for (int i = 0; i < hidden.size(); ++i) {
            bias_h[i] += learningRate * grad_hidden[i];
            for (int j = 0; j < input.size(); ++j)
                weights_ih[i][j] += learningRate * grad_hidden[i] * input[j];
        }
    }

    bool saveModel(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        auto writeMatrix = [&](const std::vector<std::vector<double>>& mat) {
            file << mat.size() << " " << mat[0].size() << "\n";
            for (const auto& row : mat) {
                for (double w : row) file << std::setprecision(10) << w << " ";
                file << "\n";
            }
        };

        auto writeVector = [&](const std::vector<double>& vec) {
            file << vec.size() << "\n";
            for (double v : vec) file << std::setprecision(10) << v << " ";
            file << "\n";
        };

        writeMatrix(weights_ih);
        writeMatrix(weights_ho);
        writeVector(bias_h);
        writeVector(bias_o);
        file.close();
        return true;
    }

    	bool loadModel(const std::string& filename) {
	    std::ifstream file(filename);
	    if (!file.is_open()) {
	        return false; // File not found or can't open
	    }
	
	    auto readMatrix = [&](std::vector<std::vector<double>>& mat) -> bool {
	        int rows, cols;
	        if (!(file >> rows >> cols)) return false;
	        mat.resize(rows, std::vector<double>(cols));
	        for (auto& row : mat) {
	            for (double& w : row) {
	                if (!(file >> w)) return false;
	            }
	        }
	        return true;
	    };
	
	    auto readVector = [&](std::vector<double>& vec) -> bool {
	        int size;
	        if (!(file >> size)) return false;
	        vec.resize(size);
	        for (double& v : vec) {
	            if (!(file >> v)) return false;
	        }
	        return true;
	    };
	
	    // Read in the same order as saveModel()
	    if (!readMatrix(weights_ih)) goto fail;
	    if (!readMatrix(weights_ho)) goto fail;
	    if (!readVector(bias_h))     goto fail;
	    if (!readVector(bias_o))     goto fail;
	
	    file.close();
	    return true;
	
	fail:
	    file.close();
	    return false;
	}
};

// ======================================
// Internal: TicTacToe Game Logic
// ======================================

class TicTacToe {
public:
    std::vector<int> board{0,0,0,0,0,0,0,0,0}; // 0=empty, 1=X, -1=O

    bool isGameOver(int& winner) const {
        const int wins[8][3] = {
            {0,1,2}, {3,4,5}, {6,7,8},
            {0,3,6}, {1,4,7}, {2,5,8},
            {0,4,8}, {2,4,6}
        };

        for (const auto& w : wins) {
            if (board[w[0]] != 0 && board[w[0]] == board[w[1]] && board[w[1]] == board[w[2]]) {
                winner = board[w[0]];
                return true;
            }
        }

        if (std::find(board.begin(), board.end(), 0) == board.end()) {
            winner = 0;
            return true;
        }

        winner = 0;
        return false;
    }

    std::vector<int> getValidMoves() const {
        std::vector<int> moves;
        for (int i = 0; i < 9; ++i)
            if (board[i] == 0) moves.push_back(i);
        return moves;
    }
};

// Helper functions
std::vector<double> boardToInput(const std::vector<int>& b) {
    return std::vector<double>(b.begin(), b.end());
}

void maskOutputs(std::vector<double>& out, const TicTacToe& g) {
    for (int i = 0; i < 9; ++i)
        if (g.board[i] != 0) out[i] = -1e9;
}

int selectMoveWithSoftmax(const std::vector<double>& output, const TicTacToe& game) {
    std::vector<double> logits = output;
    maskOutputs(logits, game);

    double maxLogit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs;
    double sumExp = 0.0;

    for (double logit : logits) {
        double expVal = std::exp(logit - maxLogit);
        probs.push_back(expVal);
        sumExp += expVal;
    }

    for (double& p : probs) p /= sumExp;

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return dist(gen);
}

void trainStep(NeuralNetwork& net) {
    TicTacToe game;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> history;
    int turn = 1;

    while (true) {
        std::vector<double> input = boardToInput(game.board);
        net.forward(input);
        int move = selectMoveWithSoftmax(net.output, game);
        history.push_back({input, net.output});
        game.board[move] = turn;

        int winner;
        if (game.isGameOver(winner)) {
            for (auto& [s, p] : history) {
                std::vector<double> target(9, 0.0);
                if (turn == 1) target[move] = (winner == 1 ? 1.0 : (winner == -1 ? -1.0 : 0.5));
                else           target[move] = (winner == -1 ? 1.0 : (winner == 1 ? -1.0 : 0.5));
                net.backprop(target);
            }
            break;
        }
        turn = -turn;
    }
}
