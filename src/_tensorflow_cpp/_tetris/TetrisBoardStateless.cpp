/*

	Compile from root with:

	g++ -std=c++20 -o __test/TetrisBoardStateless.exe _tetris/TetrisBoardStateless.cpp 

*/

#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <locale>
#include <fstream>
#include <sstream>
#include "../include/TetrisEngineStateless.h"

// For Windows ANSI support
#ifdef _WIN32
#include <windows.h>
#endif


// ==================== Graphics & I/O Constants ====================
constexpr bool USE_FANCY_GRAPHICS = true;

const std::string RESET_COLOR = "\033[0m";
const std::string EMPTY_COLOR = "\033[1;30m";
const std::string I_COLOR = "\033[1;36m";
const std::string O_COLOR = "\033[1;33m";
const std::string T_COLOR = "\033[1;35m";
const std::string S_COLOR = "\033[1;32m";
const std::string Z_COLOR = "\033[1;31m";
const std::string J_COLOR = "\033[1;34m";
const std::string L_COLOR = "\033[1;91m";
const std::string WALL_COLOR = "\033[1;37m";
const std::string PREVIEW_COLOR = "\033[1;90m";

const std::string DEFAULT_WEIGHTS_FILE = "tetris_weights.txt";

// ==================== Console Setup ====================
void SetupConsole() {
    std::setlocale(LC_ALL, "en_US.UTF-8");
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
#endif
}

void ClearScreen() {
    if (!USE_FANCY_GRAPHICS) {
        std::cout << "\n\n";
        return;
    }
    std::cout << "\033[2J\033[1;1H";
}

std::string GetColor(int pieceId) {
    if (!USE_FANCY_GRAPHICS) return "";
    switch (pieceId) {
        case 0: return EMPTY_COLOR;
        case 1: return I_COLOR;
        case 2: return O_COLOR;
        case 3: return T_COLOR;
        case 4: return S_COLOR;
        case 5: return Z_COLOR;
        case 6: return J_COLOR;
        case 7: return L_COLOR;
        default: return RESET_COLOR;
    }
}

// ==================== Rendering (Stateless) ====================
void Render(const BoardGrid& grid, int score, int lines, int level, 
            const Piece* currentPiece = nullptr, int nextPieceId = 0) {
    BoardGrid tempGrid = grid;
    
    if (currentPiece && Engine::IsValid(grid, *currentPiece)) {
        const auto& shape = currentPiece->GetShape();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                if (shape[r][c] != 0) {
                    int py = currentPiece->y + r;
                    int px = currentPiece->x + c;
                    if (py >= 0 && py < BOARD_HEIGHT && px >= 0 && px < BOARD_WIDTH) {
                        tempGrid[py][px] = shape[r][c];
                    }
                }
            }
        }
    }

    if (!USE_FANCY_GRAPHICS) {
        std::cout << "+----------------------+      Next:\n";
        for (int r = 0; r < BOARD_HEIGHT; ++r) {
            std::cout << "|";
            for (int c = 0; c < BOARD_WIDTH; ++c) {
                std::cout << (tempGrid[r][c] == 0 ? " ." : " #");
            }
            std::cout << " |";
            
            if (r >= 2 && r <= 5 && nextPieceId > 0) {
                const auto& nextShape = TETROMINO_SHAPES[nextPieceId - 1][0];
                int previewRow = r - 2;
                std::cout << "      ";
                for (int c = 0; c < 4; ++c) {
                    std::cout << (nextShape[previewRow][c] != 0 ? "# " : "  ");
                }
            } else if (r == 1) {
                std::cout << "      ----";
            } else if (r == 0) {
                std::cout << "      Next";
            }
            
            if (r == 7) std::cout << "  Score: " << score;
            if (r == 9) std::cout << "  Lines: " << lines;
            if (r == 11) std::cout << "  Level: " << level;
            std::cout << "\n";
        }
        std::cout << "+----------------------+\n";
        return;
    }

    std::cout << WALL_COLOR << "╔═════════════════════╗" << PREVIEW_COLOR << "     " << RESET_COLOR << "\n";
    for (int r = 0; r < BOARD_HEIGHT; ++r) {
        std::cout << WALL_COLOR << "║" << RESET_COLOR;
        for (int c = 0; c < BOARD_WIDTH; ++c) {
            std::cout << (tempGrid[r][c] == 0 ? EMPTY_COLOR + " ." + RESET_COLOR : GetColor(tempGrid[r][c]) + " ■" + RESET_COLOR);
        }
        std::cout << WALL_COLOR << " ║" << RESET_COLOR;
        
        if (r >= 2 && r <= 5 && nextPieceId > 0) {
            const auto& nextShape = TETROMINO_SHAPES[nextPieceId - 1][0];
            int previewRow = r - 2;
            std::cout << PREVIEW_COLOR << "      " << RESET_COLOR;
            for (int c = 0; c < 4; ++c) {
                std::cout << (nextShape[previewRow][c] != 0 ? GetColor(nextShape[previewRow][c]) + "■ " + RESET_COLOR : "  ");
            }
        } else if (r == 1) {
            std::cout << PREVIEW_COLOR << "      ────" << RESET_COLOR;
        } else if (r == 0) {
            std::cout << PREVIEW_COLOR << "      Next" << RESET_COLOR;
        }
        
        if (r == 7) std::cout << "  Score: " << score;
        if (r == 9) std::cout << "  Lines: " << lines;
        if (r == 11) std::cout << "  Level: " << level;
        std::cout << "\n";
    }
    std::cout << WALL_COLOR << "╚═════════════════════╝" << RESET_COLOR << "\n";
}

// ==================== File I/O ====================
bool SaveWeights(const HeuristicWeights& w, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }
    file << std::fixed << std::setprecision(6);
    file << w.w_lines << "\n" << w.w_height << "\n" << w.w_holes << "\n" << w.w_bumpiness << "\n";
    bool success = file.good();
    file.close();
    return success;
}

bool LoadWeights(HeuristicWeights& w, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    file >> w.w_lines >> w.w_height >> w.w_holes >> w.w_bumpiness;
    bool success = !file.fail();
    file.close();
    return success;
}

// ==================== Visual Game Play (State Managed Here) ====================
void PlayVisibleGame(const HeuristicWeights& w) {
    ClearScreen();
    BoardGrid grid = {}; // Game state held here, not in an object
    int score = 0, lines = 0, level = 1;
    int nextPieceId = Engine::Random::Int(1, 7);
    bool over = false;

    std::cout << "--- AI Playing Tetris ---\n";
    
    while (!over) {
        int currentPieceId = nextPieceId;
        nextPieceId = Engine::Random::Int(1, 7);
        Piece p{currentPieceId, 0, 3, 0};
        
        if (Engine::IsGameOver(grid, p)) { over = true; break; }
        
        // Preview initial position
        ClearScreen();
        std::cout << "--- AI Playing ---\n";
        Render(grid, score, lines, level, &p, nextPieceId);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Find and execute best move
        Move m = Engine::FindBestMove(grid, currentPieceId, w);
        p.rotation = m.rotation; p.x = m.x;
        
        int y = 0;
        while (!Engine::IsValid(grid, {currentPieceId, m.rotation, m.x, y})) y--;
        int finalY = y;
        while (Engine::IsValid(grid, {currentPieceId, m.rotation, m.x, finalY + 1})) finalY++;
        
        // Animation
        for (int dropY = y; dropY <= finalY; ++dropY) {
            p.y = dropY;
            ClearScreen();
            std::cout << "--- AI Playing ---\n";
            Render(grid, score, lines, level, &p, nextPieceId);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        
        p.y = finalY;
        grid = Engine::PlacePiece(grid, p); // State updated here
        int cleared = Engine::ClearLines(grid);
        if (cleared) {
            lines += cleared;
            score += cleared * cleared * 100 * level;
            level = 1 + (lines / 10);
        }
        
        // Show final state
        ClearScreen();
        std::cout << "--- AI Playing ---\n";
        Render(grid, score, lines, level, nullptr, nextPieceId);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    ClearScreen();
    std::cout << "--- GAME OVER ---\nFinal Score: " << score << "\nFinal Lines: " << lines << "\n";
}

// ==================== Command Line Interface ====================
void PrintUsage(const char* programName) {
    std::cout << "Tetris AI Genetic Algorithm Solver\n\n"
              << "Usage: " << programName << " [options]\n\n"
              << "Options:\n"
              << "  --train          Train a new model and save to file\n"
              << "  --play           Load model from file and play (no training)\n"
              << "  --file <path>    Specify weights file (default: tetris_weights.txt)\n"
              << "  --help           Show this help message\n\n"
              << "Examples:\n"
              << "  " << programName << "              # Train if needed, then play\n"
              << "  " << programName << " --train      # Train and save only\n"
              << "  " << programName << " --play       # Load and play only\n"
              << "  " << programName << " --play --file my_weights.txt\n";
}

int main(int argc, char* argv[]) {
    SetupConsole();
    
    std::string filename = DEFAULT_WEIGHTS_FILE;
    bool trainMode = false;
    bool playMode = false;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--train") {
            trainMode = true;
        } else if (arg == "--play") {
            playMode = true;
        } else if (arg == "--file" && i + 1 < argc) {
            filename = argv[++i];
        } else if (arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            PrintUsage(argv[0]);
            return 1;
        }
    }
    
    HeuristicWeights best;
    
    // Handle modes
    if (playMode && trainMode) {
        std::cerr << "Error: Cannot use both --train and --play simultaneously.\n";
        return 1;
    }
    
    if (playMode) {
        std::cout << "Loading model from " << filename << "...\n";
        if (!LoadWeights(best, filename)) {
            std::cerr << "Error: Could not load weights from " << filename << "\n";
            std::cerr << "Run with --train first to create the file.\n";
            return 1;
        }
        std::cout << "Model loaded: ";
        best.Print();
        std::cout << "\n\nStarting visual demonstration...\n";
        PlayVisibleGame(best);
    } else if (trainMode) {
        std::cout << "Training new model...\n";
        best = Engine::RunGeneticAlgorithm();
        if (SaveWeights(best, filename)) {
            std::cout << "\nModel saved successfully to " << filename << std::endl;
        } else {
            std::cerr << "Error: Failed to save model to " << filename << std::endl;
            return 1;
        }
    } else {
        // Default: train if needed, then play
        if (LoadWeights(best, filename)) {
            std::cout << "Found existing model in " << filename << ": ";
            best.Print();
            std::cout << "\n\nStarting visual demonstration...\n";
            PlayVisibleGame(best);
        } else {
            std::cout << "No saved model found. Training new model...\n";
            best = Engine::RunGeneticAlgorithm();
            if (SaveWeights(best, filename)) {
                std::cout << "\nModel saved to " << filename << "\n";
            } else {
                std::cerr << "Warning: Failed to save model to " << filename << "\n";
            }
            std::cout << "\nStarting visual demonstration...\n";
            PlayVisibleGame(best);
        }
    }
    
    return 0;
}