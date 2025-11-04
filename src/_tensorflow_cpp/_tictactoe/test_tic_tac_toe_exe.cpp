
/*

	execute from root :
	
	g++ -std=c++20  -I"include" -o "__test/test_tic_tac_toe_exe.exe"  "_tictactoe/test_tic_tac_toe_exe.cpp"

*/


#include "../include/tictactoe.h"

#ifdef _WIN32
    #include <conio.h>
    #include <windows.h>
    #define CLEAR_SCREEN() system("cls")
#else
    #include <unistd.h>
    #define CLEAR_SCREEN() system("clear")
#endif


// ----------------------------
// Console Animation & UI
// ----------------------------

bool ___RunTicTacToeSelfPlay(TicTacToeResultOnline& result, int aiMode, double temperature) {
    if (aiMode == TENSORFLOW) {
        return false;
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

bool PlayTicTacToeGameWithHistory(TicTacToeResultOnline* result, int aiMode, double temperature) {
        try {
            if (!result) return false;
            if (aiMode == TENSORFLOW) return false;
            return ___RunTicTacToeSelfPlay(*result, aiMode, temperature);
        } catch (...) {
            return false;
        }
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

void runMultipleGames(int aiMode) {
    const char* modeNames[] = {"Expert", "Creative", "Minimax", "Random", "TensorFlow"};

    if (aiMode == TENSORFLOW) {
        CLEAR_SCREEN();
        std::cout << "🧠 TensorFlow Model Integration\n";
        std::cout << "================================\n\n";
        std::cout << "🚧 This feature is currently under construction.\n";
        std::cout << "💡 Future version will load a .pb or ONNX model\n";
        std::cout << "   trained in Python and run inference here.\n\n";
        std::cout << "Press Enter to return to main menu...";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
        return;
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
        if (PlayTicTacToeGameWithHistory(&result, aiMode, temperature)) {
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


// ----------------------------
// Renamed main → executeConsole
// ----------------------------

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
