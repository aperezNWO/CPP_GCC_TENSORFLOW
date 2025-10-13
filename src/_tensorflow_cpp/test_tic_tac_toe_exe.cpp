
/*

	g++ -std=c++20 -o test_tic_tac_toe_exe.exe test_tic_tac_toe_exe.cpp

*/

// tictactoe_animated.cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>   // For std::exp - this fixes the error
#ifdef _WIN32
    #include <conio.h>
    #include <windows.h>
    #define CLEAR_SCREEN() system("cls")
#else
    #include <unistd.h>
    #define CLEAR_SCREEN() system("clear")
#endif

// ----------------------------
// C-Style Export Types
// ----------------------------

extern "C" {
    typedef struct {
        int finalBoard[9];         // Final board state
        int moves[9];              // Sequence of moves (-1 = not used)
        int winner;                // 1=X, -1=O, 0=draw
        int moveCount;             // Total number of moves

        // NEW: Game history (for animation)
        int history[10][9];        // Up to 10 states (initial + 9 moves)
        int historyCount;          // Actual number of saved states
    } TicTacToeResultOnline;

    // Function exposed to .NET via DllImport
    bool PlayTicTacToeGameWithHistory(TicTacToeResultOnline* result);
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

    // Check if game is over
    bool isGameOver(int& winner) const {
        const int wins[8][3] = {
            {0,1,2}, {3,4,5}, {6,7,8}, // rows
            {0,3,6}, {1,4,7}, {2,5,8}, // cols
            {0,4,8}, {2,4,6}           // diagonals
        };

        for (const auto& w : wins) {
            if (board[w[0]] != 0 &&
                board[w[0]] == board[w[1]] &&
                board[w[1]] == board[w[2]]) {
                winner = board[w[0]];
                return true;
            }
        }

        // Draw?
        for (int i = 0; i < 9; ++i) {
            if (board[i] == 0) return false; // Not full
        }
        winner = 0; // Draw
        return true;
    }

    // Get valid moves
    std::vector<int> getValidMoves() const {
        std::vector<int> moves;
        for (int i = 0; i < 9; ++i) {
            if (board[i] == 0) moves.push_back(i);
        }
        return moves;
    }
};

// ----------------------------
// Neural Network Stub (Minimal)
// ----------------------------

// In real app: link to TensorFlow or use trained weights
std::vector<double> boardToInput(const int board[9]) {
    std::vector<double> input(9);
    for (int i = 0; i < 9; ++i) {
        input[i] = static_cast<double>(board[i]); // -1, 0, 1
    }
    return input;
}

int selectMoveWithSoftmax(const std::vector<double>& output, const TicTacToe& game) {
    // Softmax selection with temperature
    double temp = 1.0;
    std::vector<double> expVals;
    double maxVal = *std::max_element(output.begin(), output.end());

    double sum = 0.0;
    for (double v : output) {
        double expVal = std::exp((v - maxVal) / temp); // FIXED: std::exp from <cmath>
        expVals.push_back(expVal);
        sum += expVal;
    }

    std::vector<double> probs;
    for (double ev : expVals) {
        probs.push_back(ev / sum);
    }

    // Select highest probability valid move
    std::vector<std::pair<double, int>> candidates;
    for (int i = 0; i < 9; ++i) {
        if (game.board[i] == 0) {
            candidates.emplace_back(probs[i], i);
        }
    }

    if (candidates.empty()) return -1;

    auto best = std::max_element(candidates.begin(), candidates.end());
    return best->second;
}

// Dummy training step
void trainStep(class NeuralNetwork& net) {
    // Simulate one training batch
    // In real code: update weights
}

// Minimal NN class (just to compile)
class NeuralNetwork {
public:
    std::vector<double> output;
    NeuralNetwork(int, int, int) : output(9, 0.0) {}
    void forward(const std::vector<double>& x) { output = x; }
    bool loadModel(const std::string&) { return true; }
    bool saveModel(const std::string&) { return true; }
};
#include <string>

// ----------------------------
// Main Game Logic with History
// ----------------------------

bool RunTicTacToeSelfPlay(TicTacToeResultOnline& result) {
    NeuralNetwork net(9, 18, 9);
    const std::string modelFile = "tictactoe_model.txt";

    if (!net.loadModel(modelFile)) {
        std::cout << "[Training] No model found. Training 5000 games...\n";
        for (int i = 0; i < 5000; ++i) trainStep(net);
        net.saveModel(modelFile);
        std::cout << "[Saved] Model saved to '" << modelFile << "'\n";
    }

    TicTacToe game;
    int turn = 1; // X starts
    std::vector<int> moves;

    // Reset history
    result.historyCount = 0;

    // Save initial state
    for (int i = 0; i < 9; ++i) {
        result.history[0][i] = game.board[i];
    }
    result.historyCount = 1;

    while (true) {
        std::vector<double> input = boardToInput(game.board);
        net.forward(input);
        int move = selectMoveWithSoftmax(net.output, game);

        if (move < 0 || move >= 9 || game.board[move] != 0) {
            // Fallback: pick first valid move
            auto valid = game.getValidMoves();
            if (valid.empty()) break;
            move = valid[0];
        }

        game.board[move] = turn;
        moves.push_back(move);

        // Save board state after move
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

    // Copy final board
    for (int i = 0; i < 9; ++i) {
        result.finalBoard[i] = game.board[i];
        result.moves[i] = (i < static_cast<int>(moves.size())) ? moves[i] : -1;
    }
    result.moveCount = static_cast<int>(moves.size());

    return true;
}

// ----------------------------
// C-Style Export Function
// ----------------------------

extern "C" {
    bool PlayTicTacToeGameWithHistory(TicTacToeResultOnline* result) {
        try {
            if (!result) return false;
            return RunTicTacToeSelfPlay(*result);
        } catch (...) {
            return false;
        }
    }
}

// ----------------------------
// Console Animation Loop
// ----------------------------

void animateGame(const TicTacToeResultOnline& result) {
    CLEAR_SCREEN();
    std::cout << "=== TIC-TAC-TOE ANIMATED GAME ===\n";

    for (int step = 0; step < result.historyCount; ++step) {
        CLEAR_SCREEN();
        std::cout << "=== TIC-TAC-TOE ANIMATED GAME ===\n";
        std::cout << "Move " << step << ":\n\n";

        // Print current board
        TicTacToe temp;
        for (int i = 0; i < 9; ++i) temp.board[i] = result.history[step][i];
        temp.print();

        // Show who played (except initial)
        if (step > 0) {
            int movePos = result.moves[step - 1];
            int player = (step % 2 == 1) ? 1 : -1; // X starts
            std::cout << "Player " << (player == 1 ? "X" : "O")
                      << " plays at position " << movePos << "\n\n";
        }

        // Delay
#ifdef _WIN32
        Sleep(800);
#else
        usleep(800000); // 0.8 seconds
#endif
    }

    // Final result
    std::cout << "Game Over! Winner: ";
    if (result.winner == 1) std::cout << "X\n";
    else if (result.winner == -1) std::cout << "O\n";
    else std::cout << "Draw\n";

    std::cout << "\nFull move sequence: ";
    for (int i = 0; i < result.moveCount; ++i) {
        std::cout << result.moves[i] << " ";
    }
    std::cout << "\n";
}

// ----------------------------
// User Prompt Functions
// ----------------------------

void waitForEnter() {
    std::cout << "Press Enter to continue...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

bool askToContinue() {
    std::string response;
    while (true) {
        std::cout << "\nDo you want to watch another game? (y/n): ";
        std::getline(std::cin, response);

        std::string lowerResponse;
        std::transform(response.begin(), response.end(), std::back_inserter(lowerResponse),
                      [](unsigned char c){ return std::tolower(c); });

        if (lowerResponse == "y" || lowerResponse == "yes") return true;
        if (lowerResponse == "n" || lowerResponse == "no") return false;
        std::cout << "Please enter 'y' or 'n'.\n";
    }
}

// ----------------------------
// Main Function
// ----------------------------

int main() {
    do {
        TicTacToeResultOnline result{};
        if (PlayTicTacToeGameWithHistory(&result)) {
            animateGame(result);
        } else {
            std::cout << "Game execution failed.\n";
        }

        waitForEnter();

    } while (askToContinue());

    std::cout << "Thanks for watching! Goodbye!\n";
    return 0;
}