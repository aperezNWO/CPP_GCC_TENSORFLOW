/*

Compile with:
g++ -std=c++17 -I"include" -L"lib" -o test_tic_tac_toe_tf.exe test_tic_tac_toe_tf.cpp -ltensorflow
g++ -std=c++17  -I"include" -L"lib" -o test_tic_tac_toe_tf.exe test_tic_tac_toe_tf.cpp -ltensorflow -m64  -Wl,--subsystem,console

Run:
./test_tic_tac_toe_tf.exe

COMPILA BIEN PERO EJECUTA CON ERRORES POR INCOMPATIBILIDAD DE MODELO TENSORFLOW DE PYTHON.

*/

#include <tensorflow/c/c_api.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <thread>
#include <chrono>

// ======================================
// Data Structure: Return full game result
// ======================================

struct TicTacToeMove {
    int position;
    int player; // 1 = X, -1 = O
};

struct TicTacToeGameResult {
    int finalBoard[9];
    std::vector<TicTacToeMove> moves;
    int winner; // 1=X, -1=O, 0=draw
    std::vector<std::vector<int>> boardHistory; // For animation
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

        std::cout << "✅ Model loaded from: " << export_dir << "\n";
        return true;
    }

    bool PredictBestMove(const float input_board[9], int& best_move) {
        if (!session) return false;

        // Input tensor: [1, 9]
        int64_t input_dims[] = {1, 9};
        TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, input_dims, 2, sizeof(float) * 9);
        float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
        for (int i = 0; i < 9; ++i) input_data[i] = input_board[i];

        // Output tensor
        int64_t output_dims[] = {1, 9};
        TF_Tensor* output_tensor = nullptr;

        // Use Netron.app to verify these names
        const char* input_name = "serving_default_dense_input";  // Adjust!
        const char* output_name = "StatefulPartitionedCall:0";   // Adjust!

        TF_Output inputs[1] = {{TF_GraphOperationByName(graph, input_name), 0}};
        TF_Output outputs[1] = {{TF_GraphOperationByName(graph, output_name), 0}};

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

        // Find best move (will be masked later)
        best_move = std::distance(probs, std::max_element(probs, probs + 9));

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
// Game Logic: Self-Play with History
// ======================================

TicTacToeGameResult PlaySelfPlayGameWithHistory() {
    TicTacToeGameResult result{};
    result.success = false;

    // Initialize game state
    std::vector<int> board(9, 0); // 0=empty, 1=X, -1=O
    int turn = 1; // X starts
    std::vector<int> moveHistory;

    // Save initial state
    result.boardHistory.push_back(board);

    // Load model
    TensorFlowTicTacToe tf;
    if (!tf.LoadModel("tictactoe_tf_model")) {
        std::cerr << "Failed to initialize TensorFlow model.\n";
        return result;
    }

    while (true) {
        // Convert to float input
        float input[9];
        for (int i = 0; i < 9; ++i) input[i] = static_cast<float>(board[i]);

        int move;
        if (!tf.PredictBestMove(input, move)) {
            std::cerr << "Prediction failed!\n";
            return result;
        }

        // Mask invalid moves manually
        auto isValidMove = [&](int m) { return m >= 0 && m < 9 && board[m] == 0; };
        if (!isValidMove(move)) {
            // Fallback: pick first valid move
            move = -1;
            for (int i = 0; i < 9; ++i) {
                if (board[i] == 0) {
                    move = i;
                    break;
                }
            }
            if (move == -1) break; // No moves left
        }

        // Make move
        board[move] = turn;
        moveHistory.push_back(move);

        // Save state after move
        result.boardHistory.push_back(board);

        // Check game over
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

    // Copy final board
    for (int i = 0; i < 9; ++i) {
        result.finalBoard[i] = board[i];
    }

    // Copy moves
    for (size_t i = 0; i < moveHistory.size(); ++i) {
        result.moves.push_back({ moveHistory[i], (i % 2 == 0) ? 1 : -1 });
    }

    result.success = true;
    return result;
}

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

        // Pause for animation
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
    }

    if (result.winner == 1)      std::cout << "🎉 X wins!\n";
    else if (result.winner == -1) std::cout << "🎉 O wins!\n";
    else                          std::cout << "🤝 It's a draw!\n";
}

// ======================================
// Main Function
// ======================================

int main() {
    std::cout << "🎮 Starting Tic-Tac-Toe AI Self-Play...\n";

    // Generate full game with history
    TicTacToeGameResult game = PlaySelfPlayGameWithHistory();

    if (!game.success) {
        std::cerr << "❌ Game execution failed.\n";
        return 1;
    }

    // Animate step-by-step
    AnimateGame(game);

    // Optional: Print raw data (for DLL use)
    /*
    std::cout << "\n--- RAW DATA (for DLL export) ---\n";
    std::cout << "Winner: " << game.winner << "\n";
    std::cout << "Moves: ";
    for (auto m : game.moves) std::cout << m.position << " ";
    std::cout << "\n";
    */

    std::cout << "\nPress Enter to exit...";
    std::cin.get();
    return 0;
}