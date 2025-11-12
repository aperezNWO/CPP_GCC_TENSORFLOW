/*
SELF PLAYING TETRIS WITH GENETIC ALGORITHM SOLVER

Compile from root with:

g++ -std=c++20  -o __test/TetrisBoard.exe  _tetris/TetrisBoard.cpp

*/
#include "TetrisEngine.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <locale>

#ifdef _WIN32
#include <windows.h>
#endif

// Add this helper function for printing weights
void PrintWeights(const TetrisEngine::HeuristicWeights& w) {
    std::cout << std::fixed << std::setprecision(4)
              << "Lines: " << w.w_lines
              << ", Height: " << w.w_height
              << ", Holes: " << w.w_holes
              << ", Bumpiness: " << w.w_bumpiness;
}

// --- Console Setup ---
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

// --- Graphics ---
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

std::string GetColor(int pieceId) {
    if (!USE_FANCY_GRAPHICS) return "";
    switch (pieceId) {
        case 1: return I_COLOR;
        case 2: return O_COLOR;
        case 3: return T_COLOR;
        case 4: return S_COLOR;
        case 5: return Z_COLOR;
        case 6: return J_COLOR;
        case 7: return L_COLOR;
        default: return EMPTY_COLOR;
    }
}

void ClearScreen() {
    if (!USE_FANCY_GRAPHICS) {
        std::cout << "\n\n";
        return;
    }
    std::cout << "\033[2J\033[1;1H";
}

// --- GA Constants ---
constexpr int POPULATION_SIZE = 50;
constexpr int NUM_GENERATIONS = 20;
constexpr int NUM_GAMES_PER_FITNESS_TEST = 1;
constexpr int MAX_MOVES_PER_GAME = 500;
constexpr double MUTATION_RATE = 0.1;
constexpr double MUTATION_STRENGTH = 0.5;
constexpr int TOURNAMENT_SIZE = 5;
constexpr double ELITISM_RATE = 0.1;
const std::string DEFAULT_WEIGHTS_FILE = "tetris_weights.txt";

// --- GA Data Structures ---
struct Individual {
    TetrisEngine::HeuristicWeights weights;
    double fitness = 0.0;
    bool operator>(const Individual& other) const { return fitness > other.fitness; }
};

// --- File I/O ---
bool SaveWeights(const TetrisEngine::HeuristicWeights& w, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file << std::fixed << std::setprecision(6);
    file << w.w_lines << "\n" << w.w_height << "\n" << w.w_holes << "\n" << w.w_bumpiness << "\n";
    bool success = file.good();
    file.close();
    return success;
}

bool LoadWeights(TetrisEngine::HeuristicWeights& w, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    file >> w.w_lines >> w.w_height >> w.w_holes >> w.w_bumpiness;
    bool success = !file.fail();
    file.close();
    return success;
}

// --- GA Operations ---
double SimulateGame(const TetrisEngine::HeuristicWeights& weights) {
    TetrisEngine::BoardEngine board;
    int lines = 0, moves = 0;
    int nextPiece = TetrisEngine::Random::Int(1, 7);
    
    while (moves < MAX_MOVES_PER_GAME) {
        int currentPiece = nextPiece;
        nextPiece = TetrisEngine::Random::Int(1, 7);
        TetrisEngine::Piece p{currentPiece, 0, 3, 0};
        if (board.IsGameOver(p)) break;
        
        auto m = TetrisEngine::FindBestMove(board, currentPiece, weights);
        p.rotation = m.rotation; p.x = m.x;
        
        int y = 0;
        while (!board.IsValid({currentPiece, m.rotation, m.x, y})) y--;
        p.y = y;
        while (board.IsValid({currentPiece, m.rotation, m.x, p.y + 1})) p.y++;
        
        board.PlacePiece(p);
        lines += board.ClearLines();
        moves++;
    }
    return static_cast<double>(lines);
}

Individual TournamentSelection(const std::vector<Individual>& pop) {
    Individual best{{}, std::numeric_limits<double>::lowest()};
    for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
        const auto& c = pop[TetrisEngine::Random::Int(0, pop.size()-1)];
        if (c.fitness > best.fitness) best = c;
    }
    return best;
}

TetrisEngine::HeuristicWeights Crossover(const TetrisEngine::HeuristicWeights& a, const TetrisEngine::HeuristicWeights& b) {
    return {
        (a.w_lines + b.w_lines) / 2.0,
        (a.w_height + b.w_height) / 2.0,
        (a.w_holes + b.w_holes) / 2.0,
        (a.w_bumpiness + b.w_bumpiness) / 2.0
    };
}

void Mutate(TetrisEngine::HeuristicWeights& w) {
    if (TetrisEngine::Random::Double(0,1) < MUTATION_RATE) w.w_lines += TetrisEngine::Random::Normal(0, MUTATION_STRENGTH);
    if (TetrisEngine::Random::Double(0,1) < MUTATION_RATE) w.w_height += TetrisEngine::Random::Normal(0, MUTATION_STRENGTH);
    if (TetrisEngine::Random::Double(0,1) < MUTATION_RATE) w.w_holes += TetrisEngine::Random::Normal(0, MUTATION_STRENGTH);
    if (TetrisEngine::Random::Double(0,1) < MUTATION_RATE) w.w_bumpiness += TetrisEngine::Random::Normal(0, MUTATION_STRENGTH);
}

TetrisEngine::HeuristicWeights RunGeneticAlgorithm() {
    std::vector<Individual> pop(POPULATION_SIZE);
    for (auto& ind : pop) ind.weights = TetrisEngine::HeuristicWeights::RandomWeights();

    std::cout << "Starting Genetic Algorithm training...\n";
    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        // Evaluate fitness
        for (auto& ind : pop) {
            double f = 0;
            for (int i = 0; i < NUM_GAMES_PER_FITNESS_TEST; ++i)
                f += SimulateGame(ind.weights);
            ind.fitness = f / NUM_GAMES_PER_FITNESS_TEST;
        }

        std::sort(pop.begin(), pop.end(), std::greater<Individual>());

        // Build new population
        std::vector<Individual> newPop;
        int elite = POPULATION_SIZE * ELITISM_RATE;
        for (int i = 0; i < elite; ++i) newPop.push_back(pop[i]);

        while (newPop.size() < POPULATION_SIZE) {
            auto p1 = TournamentSelection(pop);
            auto p2 = TournamentSelection(pop);
            auto child = Crossover(p1.weights, p2.weights);
            Mutate(child);
            newPop.push_back({child, 0.0});
        }
        pop = newPop;

        std::cout << "Gen " << gen+1 << ": Best=" << pop[0].fitness << " ";
        PrintWeights(pop[0].weights); std::cout << "\n";  // FIXED: Use free function
    }
    std::cout << "Training complete!\n";
    return pop[0].weights;
}

// --- Console Rendering ---
void RenderBoard(const TetrisEngine::BoardEngine& board, int score, int lines, int level,
                 const TetrisEngine::Piece* currentPiece = nullptr, int nextPieceId = 0) {
    auto tempGrid = board.GetGrid();
    
    if (currentPiece && board.IsValid(*currentPiece)) {
        const auto& shape = currentPiece->GetShape();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                if (shape[r][c] != 0) {
                    int py = currentPiece->y + r;
                    int px = currentPiece->x + c;
                    // FIXED: Add TetrisEngine:: namespace
                    if (py >= 0 && py < TetrisEngine::BOARD_HEIGHT && px >= 0 && px < TetrisEngine::BOARD_WIDTH) {
                        tempGrid[py][px] = shape[r][c];
                    }
                }
            }
        }
    }

    if (!USE_FANCY_GRAPHICS) {
        std::cout << "+----------------------+      Next:\n";
        // FIXED: Add TetrisEngine:: namespace
        for (int r = 0; r < TetrisEngine::BOARD_HEIGHT; ++r) {
            std::cout << "|";
            // FIXED: Add TetrisEngine:: namespace
            for (int c = 0; c < TetrisEngine::BOARD_WIDTH; ++c) {
                std::cout << (tempGrid[r][c] == 0 ? " ." : " #");
            }
            std::cout << " |";
            if (r == 0) std::cout << "      Next";
            else if (r == 1) std::cout << "      ----";
            else if (r >= 2 && r <= 5 && nextPieceId > 0) {
                const auto& nextShape = TetrisEngine::TETROMINO_SHAPES[nextPieceId - 1][0];
                int previewRow = r - 2;
                std::cout << "      ";
                for (int c = 0; c < 4; ++c) {
                    std::cout << (nextShape[previewRow][c] != 0 ? "# " : "  ");
                }
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
    // FIXED: Add TetrisEngine:: namespace
    for (int r = 0; r < TetrisEngine::BOARD_HEIGHT; ++r) {
        std::cout << WALL_COLOR << "║" << RESET_COLOR;
        // FIXED: Add TetrisEngine:: namespace
        for (int c = 0; c < TetrisEngine::BOARD_WIDTH; ++c) {
            std::cout << (tempGrid[r][c] == 0 ? 
                std::string(EMPTY_COLOR + " ." + RESET_COLOR) : 
                std::string(GetColor(tempGrid[r][c]) + " ■" + RESET_COLOR));
        }
        std::cout << WALL_COLOR << " ║" << RESET_COLOR;
        if (r == 0) std::cout << PREVIEW_COLOR << "      Next" << RESET_COLOR;
        else if (r == 1) std::cout << PREVIEW_COLOR << "      ────" << RESET_COLOR;
        else if (r >= 2 && r <= 5 && nextPieceId > 0) {
            const auto& nextShape = TetrisEngine::TETROMINO_SHAPES[nextPieceId - 1][0];
            int previewRow = r - 2;
            std::cout << PREVIEW_COLOR << "      " << RESET_COLOR;
            for (int c = 0; c < 4; ++c) {
                std::cout << (nextShape[previewRow][c] != 0 ? 
                    std::string(GetColor(nextShape[previewRow][c]) + "■ " + RESET_COLOR) : "  ");
            }
        }
        if (r == 7) std::cout << "  Score: " << score;
        if (r == 9) std::cout << "  Lines: " << lines;
        if (r == 11) std::cout << "  Level: " << level;
        std::cout << "\n";
    }
    std::cout << WALL_COLOR << "╚═════════════════════╝" << RESET_COLOR << "\n";
}

// --- Visual Game Loop ---
void PlayVisibleGame(const TetrisEngine::HeuristicWeights& w) {
    ClearScreen();
    TetrisEngine::BoardEngine board;
    int score = 0, lines = 0, level = 1;
    int nextPieceId = TetrisEngine::Random::Int(1, 7);
    bool over = false;

    std::cout << "--- AI Playing Tetris ---\n";
    
    while (!over) {
        int currentPieceId = nextPieceId;
        nextPieceId = TetrisEngine::Random::Int(1, 7);
        TetrisEngine::Piece p{currentPieceId, 0, 3, 0};
        
        if (board.IsGameOver(p)) { over = true; break; }
        
        ClearScreen();
        std::cout << "--- AI Playing ---\n";
        RenderBoard(board, score, lines, level, &p, nextPieceId);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        auto m = TetrisEngine::FindBestMove(board, currentPieceId, w);
        p.rotation = m.rotation; p.x = m.x;
        
        int y = 0;
        while (!board.IsValid({currentPieceId, m.rotation, m.x, y})) y--;
        int finalY = y;
        while (board.IsValid({currentPieceId, m.rotation, m.x, finalY + 1})) finalY++;
        
        // Animation
        for (int dropY = y; dropY <= finalY; ++dropY) {
            p.y = dropY;
            ClearScreen();
            std::cout << "--- AI Playing ---\n";
            RenderBoard(board, score, lines, level, &p, nextPieceId);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        
        p.y = finalY;
        board.PlacePiece(p);
        int cleared = board.ClearLines();
        if (cleared) {
            lines += cleared;
            score += cleared * cleared * 100 * level;
            level = 1 + (lines / 10);
        }
        
        ClearScreen();
        std::cout << "--- AI Playing ---\n";
        RenderBoard(board, score, lines, level, nullptr, nextPieceId);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    ClearScreen();
    std::cout << "--- GAME OVER ---\nFinal Score: " << score << "\nFinal Lines: " << lines << "\n";
}

// --- CLI ---
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
              << "  " << programName << " --play       # Load and play only\n";
}

// --- Main ---
int main(int argc, char* argv[]) {
    SetupConsole();
    
    std::string filename = DEFAULT_WEIGHTS_FILE;
    bool trainMode = false;
    bool playMode = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--train") trainMode = true;
        else if (arg == "--play") playMode = true;
        else if (arg == "--file" && i + 1 < argc) filename = argv[++i];
        else if (arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            PrintUsage(argv[0]);
            return 1;
        }
    }
    
    if (playMode && trainMode) {
        std::cerr << "Error: Cannot use both --train and --play.\n";
        return 1;
    }
    
    TetrisEngine::HeuristicWeights best;
    
    if (playMode) {
        std::cout << "Loading model from " << filename << "...\n";
        if (!LoadWeights(best, filename)) {
            std::cerr << "Error: Could not load weights. Run with --train first.\n";
            return 1;
        }
        std::cout << "Model loaded. Starting visual demonstration...\n";
        PlayVisibleGame(best);
    } else if (trainMode) {
        std::cout << "Training new model...\n";
        best = RunGeneticAlgorithm();
        if (SaveWeights(best, filename)) {
            std::cout << "\nModel saved successfully to " << filename << std::endl;
        }
    } else {
        if (LoadWeights(best, filename)) {
            std::cout << "Found existing model. Starting visual demonstration...\n";
            PlayVisibleGame(best);
        } else {
            std::cout << "No saved model found. Training new model...\n";
            best = RunGeneticAlgorithm();
            SaveWeights(best, filename);
            std::cout << "\nStarting visual demonstration...\n";
            PlayVisibleGame(best);
        }
    }
    
    return 0;
}
