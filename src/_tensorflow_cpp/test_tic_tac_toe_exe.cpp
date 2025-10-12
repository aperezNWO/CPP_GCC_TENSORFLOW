
/*

	g++ -std=c++20 -o test_tic_tac_toe_exe.exe test_tic_tac_toe_exe.cpp

*/

#include "include/ticTacToeAIAppCpp.h"


#ifdef _WIN32
    #include <conio.h>
    #include <windows.h>
    #define CLEAR_SCREEN() system("cls")
#else
    #include <unistd.h>
    #define CLEAR_SCREEN() system("clear")
#endif

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


// Play one self-play test game
void playTestGame(NeuralNetwork& net) {
    TicTacToe testGame;
    int turn = 1; // X starts

    std::cout << "\n=== TEST GAME: Network vs Itself ===\n";

    while (true) {

		CLEAR_SCREEN(); // Clear screen at the start of each move
		         
        std::cout << "\n=== TEST GAME: Network vs Itself ===\n";
        testGame.print();

        std::vector<double> input = boardToInput(testGame.board);
        net.forward(input);
        //int move = selectMove(net.output, testGame);
        int move   = selectMoveWithSoftmax(net.output, testGame);

        std::cout << "Player " << (turn == 1 ? "X" : "O") << " plays at position " << move << "\n";

        testGame.board[move] = turn;

        int winner;
        if (testGame.isGameOver(winner)) {
            CLEAR_SCREEN();
            std::cout << "\n=== TEST GAME: Network vs Itself ===\n";
            testGame.print();
            if (winner == 1)      std::cout << " X wins!\n";
            else if (winner == -1) std::cout << " O wins!\n";
            else                  std::cout << " Draw!\n";

            break;
        }

        turn = -turn; // Switch player

        // Optional: small delay to make it watchable
        #ifdef _WIN32
            Sleep(500); // 500 ms
        #else
            usleep(500000); // 500 ms
        #endif
        

    }
}

// Function to pause until user presses Enter
void waitForEnter() {
    std::cout << "Press Enter to continue...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    //std::cin.get(); // Wait for Enter (handles newline from previous input)
}

// Function to ask if user wants to continue
bool askToContinue() {
    std::string response;
    while (true) {
        std::cout << "\nDo you want to watch another game? (y/n): ";
        std::getline(std::cin, response);

        // Convert to lowercase for case-insensitive comparison
        std::string lowerResponse;
        std::transform(response.begin(), response.end(), std::back_inserter(lowerResponse),
                      [](unsigned char c){ return std::tolower(c); });

        if (lowerResponse == "y" || lowerResponse == "yes") {
            return true;
        } else if (lowerResponse == "n" || lowerResponse == "no") {
            return false;
        } else {
            std::cout << "Please enter 'y' or 'n'.\n";
        }
    }
}

bool /*TensorFlowApp::*/RunTicTacToeSelfPlay(TicTacToeResult& result) {
	//
    NeuralNetwork net(9, 18, 9);
    const std::string modelFile = "tictactoe_model.txt";

    if (!net.loadModel(modelFile)) {
        std::cout << "[Training] No model found. Training 5000 games...\n";
        for (int i = 0; i < 5000; ++i) trainStep(net);
        net.saveModel(modelFile);
        std::cout << "[Saved] Model saved to '" << modelFile << "'\n";
    }

    // Play one game
    TicTacToe game;
    int turn = 1;
    std::vector<int> moves;

    while (true) {
        std::vector<double> input = boardToInput(game.board);
        net.forward(input);
        
		//int move = selectMove(net.output, game);
        int move   = selectMoveWithSoftmax(net.output, game);
        
        game.board[move] = turn;
        moves.push_back(move);


        CLEAR_SCREEN(); 
        
        game.print();
        
        waitForEnter();

        int winner;
        if (game.isGameOver(winner)) {
            result.winner = winner;
            break;
        }
        turn = -turn;
    }
	
    // Copy results
    for (int i = 0; i < 9; ++i) {
        result.board[i] = game.board[i];
        result.moves[i] = (i < static_cast<int>(moves.size())) ? moves[i] : -1;
    }
    result.moveCount = static_cast<int>(moves.size());
	
    return true;
}

bool PlayTicTacToeGame(int* boardOut, int* movesOut, int* winnerOut, int* moveCountOut) {
    try {
        //static TensorFlowApp app;
        
        
        TicTacToeResult result{};
        if (!/*app.*/RunTicTacToeSelfPlay(result)) return false;

        for (int i = 0; i < 9; ++i) {
            boardOut[i] = result.board[i];
            movesOut[i] = result.moves[i];
        }
        *winnerOut = result.winner;
        *moveCountOut = result.moveCount;

        
		return true;
    } catch (...) {
        return false;
    }
}

int main()
{
	
	do 
	{
	    int board[9], moves[9], winner, moveCount;
	    if (PlayTicTacToeGame(board, moves, &winner, &moveCount)) {
	        printf("\n--- TIC-TAC-TOE GAME RESULT ---\n");
	        printf("Winner: %s\n", winner == 1 ? "X" : winner == -1 ? "O" : "Draw");
	        printf("Moves: ");
	        for (int i = 0; i < moveCount; ++i) printf("%d ", moves[i]);
	        printf("\n");
	    } else {
	        printf("Game execution failed.\n");
	    }
		
	} while (askToContinue());

    std::cout << "Thanks for watching! Goodbye!\n";
    
	return 0;
}