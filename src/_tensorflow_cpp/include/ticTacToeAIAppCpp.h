#ifndef CHESSAIAPPCPP_H // include guard
#define CHESSAIAPPCPP_H
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





// C-style export types
extern "C" {
    typedef struct {
        int board[9];
        int moves[9]; // -1 if not used
        int winner;   // 1=X, -1=O, 0=draw
        int moveCount;
    } TicTacToeResult;

    bool PlayTicTacToeGame(int* boardOut, int* movesOut, int* winnerOut, int* moveCountOut);
}


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


// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);
std::uniform_int_distribution<> moveDis(0, 8);

// Sigmoid function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-std::max(-700.0, std::min(700.0, x))));
}

// Derivative of sigmoid
double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}


// Function to pause until user presses Enter
void waitForEnter() {
    std::cout << "Press Enter to continue...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    //std::cin.get(); // Wait for Enter (handles newline from previous input)
}

class NeuralNetworkTicTacToe {
public:
    std::vector<double> input, hidden, output;
    std::vector<std::vector<double>> weights_ih, weights_ho;
    std::vector<double> bias_h, bias_o;
    double learningRate = 0.1;

    NeuralNetworkTicTacToe(int inputSize, int hiddenSize, int outputSize)
        : input(inputSize), hidden(hiddenSize), output(outputSize),
          weights_ih(hiddenSize, std::vector<double>(inputSize)),
          weights_ho(outputSize, std::vector<double>(hiddenSize)),
          bias_h(hiddenSize), bias_o(outputSize) {

        // Initialize weights and biases randomly
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
            for (int j = 0; j < input.size(); ++j) {
                activation += weights_ih[i][j] * input[j];
            }
            hidden[i] = sigmoid(activation);
        }

        for (int i = 0; i < output.size(); ++i) {
            double activation = bias_o[i];
            for (int j = 0; j < hidden.size(); ++j) {
                activation += weights_ho[i][j] * hidden[j];
            }
            output[i] = sigmoid(activation);
        }
    }

    void backprop(const std::vector<double>& target) {
        std::vector<double> outputError(output.size());
        std::vector<double> hiddenError(hidden.size());

        // Output layer gradients
        std::vector<double> grad_output(output.size());
        for (int i = 0; i < output.size(); ++i) {
            double error = target[i] - output[i];
            grad_output[i] = error * sigmoidDerivative(output[i]);
        }

        // Hidden layer gradients
        std::vector<double> grad_hidden(hidden.size(), 0);
        for (int i = 0; i < hidden.size(); ++i) {
            for (int j = 0; j < output.size(); ++j) {
                grad_hidden[i] += grad_output[j] * weights_ho[j][i];
            }
            grad_hidden[i] *= sigmoidDerivative(hidden[i]);
        }

        // Update weights and biases (output layer)
        for (int i = 0; i < output.size(); ++i) {
            bias_o[i] += learningRate * grad_output[i];
            for (int j = 0; j < hidden.size(); ++j) {
                weights_ho[i][j] += learningRate * grad_output[i] * hidden[j];
            }
        }

        // Update weights and biases (hidden layer)
        for (int i = 0; i < hidden.size(); ++i) {
            bias_h[i] += learningRate * grad_hidden[i];
            for (int j = 0; j < input.size(); ++j) {
                weights_ih[i][j] += learningRate * grad_hidden[i] * input[j];
            }
        }
    }
    
    //  ADD THE saveModel FUNCTION HERE, INSIDE THE CLASS
	bool saveModel(const std::string& filename) {
	        std::ofstream file(filename);
	        if (!file.is_open()) {
	            return false;
	        }
	
	        auto writeMatrix = [&](const std::vector<std::vector<double>>& mat) {
	            file << mat.size() << " " << mat[0].size() << "\n";
	            for (const auto& row : mat) {
	                for (double w : row) {
	                    file << std::setprecision(10) << w << " ";
	                }
	                file << "\n";
	            }
	        };
	
	        auto writeVector = [&](const std::vector<double>& vec) {
	            file << vec.size() << "\n";
	            for (double v : vec) {
	                file << std::setprecision(10) << v << " ";
	            }
	            file << "\n";
	        };
	
	        writeMatrix(weights_ih);
	        writeMatrix(weights_ho);
	        writeVector(bias_h);
	        writeVector(bias_o);
	
	        file.close();
	        return true;
	    }
	    //
	    // Loads the model from a text file
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

// Tic-Tac-Toe Game Logic
class TicTacToe {
public:
    std::vector<int> board{0, 0, 0, 0, 0, 0, 0, 0, 0}; // 0=empty, 1=X, -1=O

    void print() {
        for (int i = 0; i < 9; ++i) {
            char c = '.';
            if (board[i] == 1) c = 'X';
            else if (board[i] == -1) c = 'O';
            std::cout << c;
            if ((i+1) % 3 == 0) std::cout << std::endl;
        }
        std::cout << "---\n";
    }

    bool isGameOver(int& winner) {
        const int wins[8][3] = {
            {0,1,2}, {3,4,5}, {6,7,8}, // rows
            {0,3,6}, {1,4,7}, {2,5,8}, // cols
            {0,4,8}, {2,4,6}           // diagonals
        };

        for (auto& w : wins) {
            if (board[w[0]] != 0 && board[w[0]] == board[w[1]] && board[w[1]] == board[w[2]]) {
                winner = board[w[0]];
                return true;
            }
        }

        // Draw?
        if (std::find(board.begin(), board.end(), 0) == board.end()) {
            winner = 0;
            return true;
        }

        winner = 0;
        return false;
    }

    std::vector<int> getValidMoves() {
        std::vector<int> moves;
        for (int i = 0; i < 9; ++i)
            if (board[i] == 0) moves.push_back(i);
        return moves;
    }

    void reset() {
        board.assign(9, 0);
    }
};

// Convert board to network input (-1, 0, 1) -> (double)
std::vector<double> boardToInput(const std::vector<int>& board) {
    std::vector<double> input;
    for (int cell : board) {
        input.push_back(static_cast<double>(cell));
    }
    return input;
}

// Mask invalid outputs (already taken spots)
void maskOutputs(std::vector<double>& output, const TicTacToe& game) {
    for (int i = 0; i < 9; ++i) {
        if (game.board[i] != 0) {
            output[i] = -1e9; // effectively zero after softmax-like selection
        }
    }
}

// Choose action: pick highest scoring valid move
int selectMove(const std::vector<double>& output, const TicTacToe& game) {
    std::vector<double> masked = output;
    maskOutputs(masked, game);

    return std::distance(masked.begin(), std::max_element(masked.begin(), masked.end()));
}

int selectMoveWithSoftmax(const std::vector<double>& output, const TicTacToe& game) {
    std::vector<double> logits = output;
    maskOutputs(logits, game); // Set invalid moves to very low value

    // Apply softmax to turn into probabilities
    double maxLogit = *std::max_element(logits.begin(), logits.end());
    double sumExp = 0.0;
    std::vector<double> probs;

    for (double logit : logits) {
        double expVal = std::exp(logit - maxLogit); // Stable softmax
        probs.push_back(expVal);
        sumExp += expVal;
    }

    // Normalize
    for (double& p : probs) p /= sumExp;

    // Sample from distribution
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return dist(gen); // gen = global random generator
}

// Simulate one self-play game and train the network
void trainStep(NeuralNetworkTicTacToe& net) {
    TicTacToe game;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> history; // (state, move_prob)

    int turn = 1; // 1 = X (network), -1 = O (network too)
    while (true) {
        std::vector<double> input = boardToInput(game.board);
        net.forward(input);

        auto validMoves = game.getValidMoves();
        if (validMoves.empty()) break;

        int move = selectMove(net.output, game);

        // Store state and raw output before masking
        history.push_back({input, net.output});

        game.board[move] = turn;

        int winner;
        if (game.isGameOver(winner)) {
            // Generate targets based on outcome
            for (auto& [state, probs] : history) {
                std::vector<double> target(9, 0.0);
                if (turn == 1) {
                    // Last move was winning/drawing
                    if (winner == 1) target[move] = 1.0;   // win
                    else if (winner == -1) target[move] = -1.0; // loss
                    else target[move] = 0.5;             // draw
                } else {
                    if (winner == -1) target[move] = 1.0;
                    else if (winner == 1) target[move] = -1.0;
                    else target[move] = 0.5;
                }

                // Simple TD-style update: reinforce final result
                net.backprop(target);
            }
            break;
        }

        turn = -turn; // switch player
    }
}




