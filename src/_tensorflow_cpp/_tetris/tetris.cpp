/*
SELF PLAYING TETRIS WITH GENETIC ALGORITHM SOLVER

Compile with:
g++ -std=c++20 -O3 -o tetris_ga_solver tetris_ga_solver.cpp

[ask to fix to : kimi.com]

-- [_] 


*/

#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <iomanip>
#include <locale>
#include <fstream>
#include <sstream>

// For Windows ANSI support
#ifdef _WIN32
#include <windows.h>
#endif

// --- Global Constants ---
constexpr int BOARD_WIDTH = 10;
constexpr int BOARD_HEIGHT = 20;

constexpr int POPULATION_SIZE = 50;
constexpr int NUM_GENERATIONS = 20;
constexpr int NUM_GAMES_PER_FITNESS_TEST = 1;
constexpr int MAX_MOVES_PER_GAME = 500;
constexpr double MUTATION_RATE = 0.1;
constexpr double MUTATION_STRENGTH = 0.5;
constexpr int TOURNAMENT_SIZE = 5;
constexpr double ELITISM_RATE = 0.1;

// Set to false for ASCII-only mode (maximum compatibility)
constexpr bool USE_FANCY_GRAPHICS = true;

// Default file to save/load weights
const std::string DEFAULT_WEIGHTS_FILE = "tetris_weights.txt";

// --- Console Setup ---
void SetupConsole() {
    // Enable UTF-8 locale for proper character encoding
    std::setlocale(LC_ALL, "en_US.UTF-8");
    
#ifdef _WIN32
    // Enable ANSI escape sequences on Windows 10+
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

// --- Random Number Generation ---
namespace Random {
    std::mt19937 generator(std::random_device{}());

    int Int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(generator);
    }

    double Double(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(generator);
    }

    double Normal(double mean, double stddev) {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(generator);
    }
}

// --- Graphics Constants ---
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

void ClearScreen() {
    if (!USE_FANCY_GRAPHICS) {
        std::cout << "\n\n";
        return;
    }
    std::cout << "\033[2J\033[1;1H";
}

// Define the Shape type
using Shape = std::array<std::array<int, 4>, 4>;

// --- Move struct ---
struct Move {
    int rotation = 0;
    int x = 0;
    double score = 0.0;
};

// --- Tetromino Shapes ---
const std::array<std::array<Shape, 4>, 7> TETROMINO_SHAPES = {{
    // I Piece
    std::array<Shape, 4>{{
        Shape{{{0,0,0,0}, {1,1,1,1}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,1,0,0}, {0,1,0,0}, {0,1,0,0}, {0,1,0,0}}},
        Shape{{{0,0,0,0}, {0,0,0,0}, {1,1,1,1}, {0,0,0,0}}},
        Shape{{{0,0,1,0}, {0,0,1,0}, {0,0,1,0}, {0,0,1,0}}}
    }},
    // O Piece
    std::array<Shape, 4>{{
        Shape{{{0,2,2,0}, {0,2,2,0}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,2,2,0}, {0,2,2,0}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,2,2,0}, {0,2,2,0}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,2,2,0}, {0,2,2,0}, {0,0,0,0}, {0,0,0,0}}}
    }},
    // T Piece
    std::array<Shape, 4>{{
        Shape{{{0,0,0,0}, {3,3,3,0}, {0,3,0,0}, {0,0,0,0}}},
        Shape{{{0,3,0,0}, {3,3,0,0}, {0,3,0,0}, {0,0,0,0}}},
        Shape{{{0,3,0,0}, {3,3,3,0}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,3,0,0}, {0,3,3,0}, {0,3,0,0}, {0,0,0,0}}}
    }},
    // S Piece
    std::array<Shape, 4>{{
        Shape{{{0,0,0,0}, {0,4,4,0}, {4,4,0,0}, {0,0,0,0}}},
        Shape{{{0,4,0,0}, {0,4,4,0}, {0,0,4,0}, {0,0,0,0}}},
        Shape{{{0,0,0,0}, {0,4,4,0}, {4,4,0,0}, {0,0,0,0}}},
        Shape{{{0,4,0,0}, {0,4,4,0}, {0,0,4,0}, {0,0,0,0}}}
    }},
    // Z Piece
    std::array<Shape, 4>{{
        Shape{{{0,0,0,0}, {5,5,0,0}, {0,5,5,0}, {0,0,0,0}}},
        Shape{{{0,0,5,0}, {0,5,5,0}, {0,5,0,0}, {0,0,0,0}}},
        Shape{{{0,0,0,0}, {5,5,0,0}, {0,5,5,0}, {0,0,0,0}}},
        Shape{{{0,0,5,0}, {0,5,5,0}, {0,5,0,0}, {0,0,0,0}}}
    }},
    // J Piece
    std::array<Shape, 4>{{
        Shape{{{0,0,0,0}, {6,6,6,0}, {0,0,6,0}, {0,0,0,0}}},
        Shape{{{0,6,0,0}, {0,6,0,0}, {6,6,0,0}, {0,0,0,0}}},
        Shape{{{6,0,0,0}, {6,6,6,0}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,6,6,0}, {0,6,0,0}, {0,6,0,0}, {0,0,0,0}}}
    }},
    // L Piece
    std::array<Shape, 4>{{
        Shape{{{0,0,0,0}, {7,7,7,0}, {7,0,0,0}, {0,0,0,0}}},
        Shape{{{7,7,0,0}, {0,7,0,0}, {0,7,0,0}, {0,0,0,0}}},
        Shape{{{0,0,7,0}, {7,7,7,0}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,7,0,0}, {0,7,0,0}, {0,7,7,0}, {0,0,0,0}}}
    }}
}};

// --- Game Piece ---
struct Piece {
    int typeId;
    int rotation;
    int x;
    int y;

    const Shape& GetShape() const {
        return TETROMINO_SHAPES[typeId - 1][rotation];
    }
};

// --- Board Class ---
class Board {
public:
    Board() {
        grid.fill({});
    }

    bool IsValid(const Piece& piece) const {
        const auto& shape = piece.GetShape();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                if (shape[r][c] != 0) {
                    int boardX = piece.x + c;
                    int boardY = piece.y + r;
                    if (boardX < 0 || boardX >= BOARD_WIDTH || boardY < 0 || boardY >= BOARD_HEIGHT)
                        return false;
                    if (boardY >= 0 && grid[boardY][boardX] != 0)
                        return false;
                }
            }
        }
        return true;
    }

    void PlacePiece(const Piece& piece) {
        const auto& shape = piece.GetShape();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                if (shape[r][c] != 0) {
                    int py = piece.y + r;
                    int px = piece.x + c;
                    if (py >= 0 && py < BOARD_HEIGHT && px >= 0 && px < BOARD_WIDTH) {
                        grid[py][px] = piece.typeId;
                    }
                }
            }
        }
    }

    int ClearLines() {
        int lines = 0;
        for (int r = BOARD_HEIGHT - 1; r >= 0; ) {
            bool full = true;
            for (int c = 0; c < BOARD_WIDTH; ++c) {
                if (grid[r][c] == 0) {
                    full = false;
                    break;
                }
            }
            if (full) {
                lines++;
                for (int rr = r; rr > 0; --rr) {
                    grid[rr] = grid[rr - 1];
                }
                grid[0].fill(0);
            } else {
                --r;
            }
        }
        return lines;
    }

    bool IsGameOver(const Piece& piece) const {
        return !IsValid(piece);
    }

    void Render(int score, int lines, int level) const {
        if (!USE_FANCY_GRAPHICS) {
            // ASCII-only rendering
            std::cout << "+----------------------+\n";
            for (int r = 0; r < BOARD_HEIGHT; ++r) {
                std::cout << "|";
                for (int c = 0; c < BOARD_WIDTH; ++c) {
                    if (grid[r][c] == 0) {
                        std::cout << " .";
                    } else {
                        std::cout << " #";
                    }
                }
                std::cout << " |";
                if (r == 2) std::cout << "  Score: " << score;
                if (r == 4) std::cout << "  Lines: " << lines;
                if (r == 6) std::cout << "  Level: " << level;
                std::cout << "\n";
            }
            std::cout << "+----------------------+\n";
            return;
        }

        // Fancy Unicode rendering
        std::cout << WALL_COLOR << "╔══════════════════════╗" << RESET_COLOR << "\n";
        for (int r = 0; r < BOARD_HEIGHT; ++r) {
            std::cout << WALL_COLOR << "║" << RESET_COLOR;
            for (int c = 0; c < BOARD_WIDTH; ++c) {
                if (grid[r][c] == 0) {
                    std::cout << EMPTY_COLOR << " ." << RESET_COLOR;
                } else {
                    std::cout << GetColor(grid[r][c]) << " ■" << RESET_COLOR;
                }
            }
            std::cout << WALL_COLOR << " ║" << RESET_COLOR;
            if (r == 2) std::cout << "  Score: " << score;
            if (r == 4) std::cout << "  Lines: " << lines;
            if (r == 6) std::cout << "  Level: " << level;
            std::cout << "\n";
        }
        std::cout << WALL_COLOR << "╚══════════════════════╝" << RESET_COLOR << "\n";
    }

    int GetAggregateHeight() const {
        int total = 0;
        for (int c = 0; c < BOARD_WIDTH; ++c) {
            for (int r = 0; r < BOARD_HEIGHT; ++r) {
                if (grid[r][c] != 0) {
                    total += BOARD_HEIGHT - r;
                    break;
                }
            }
        }
        return total;
    }

    int GetHoles() const {
        int holes = 0;
        for (int c = 0; c < BOARD_WIDTH; ++c) {
            bool block = false;
            for (int r = 0; r < BOARD_HEIGHT; ++r) {
                if (grid[r][c] != 0) block = true;
                else if (block) holes++;
            }
        }
        return holes;
    }

    int GetBumpiness() const {
        int heights[BOARD_WIDTH] = {};
        for (int c = 0; c < BOARD_WIDTH; ++c) {
            for (int r = 0; r < BOARD_HEIGHT; ++r) {
                if (grid[r][c] != 0) {
                    heights[c] = BOARD_HEIGHT - r;
                    break;
                }
            }
        }
        int bump = 0;
        for (int i = 0; i < BOARD_WIDTH - 1; ++i) {
            bump += std::abs(heights[i] - heights[i+1]);
        }
        return bump;
    }

private:
    std::array<std::array<int, BOARD_WIDTH>, BOARD_HEIGHT> grid;
};

// --- Heuristic Weights & Individual ---
struct HeuristicWeights {
    double w_lines = 0.0;
    double w_height = 0.0;
    double w_holes = 0.0;
    double w_bumpiness = 0.0;

    static HeuristicWeights RandomWeights() {
        return {
            Random::Double(0.1, 1.0),
            Random::Double(-1.0, -0.1),
            Random::Double(-1.0, -0.1),
            Random::Double(-1.0, -0.1)
        };
    }

    void Print() const {
        std::cout << std::fixed << std::setprecision(4)
                  << "Lines: " << w_lines
                  << ", Height: " << w_height
                  << ", Holes: " << w_holes
                  << ", Bumpiness: " << w_bumpiness;
    }
};

struct Individual {
    HeuristicWeights weights;
    double fitness = 0.0;
    bool operator>(const Individual& other) const { return fitness > other.fitness; }
};

// --- File I/O Functions ---
bool SaveWeights(const HeuristicWeights& w, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }
    file << std::fixed << std::setprecision(6);
    file << w.w_lines << "\n";
    file << w.w_height << "\n";
    file << w.w_holes << "\n";
    file << w.w_bumpiness << "\n";
    bool success = file.good();
    file.close();
    return success;
}

bool LoadWeights(HeuristicWeights& w, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false; // File doesn't exist, not necessarily an error
    }
    file >> w.w_lines;
    file >> w.w_height;
    file >> w.w_holes;
    file >> w.w_bumpiness;
    bool success = !file.fail();
    file.close();
    return success;
}

// --- AI & GA Functions ---
Move FindBestMove(const Board& board, int pieceId, const HeuristicWeights& weights) {
    Move best = {0, 0, std::numeric_limits<double>::lowest()};
    for (int r = 0; r < 4; ++r) {
        for (int x = -3; x < BOARD_WIDTH + 3; ++x) {
            int y = 0;
            while (!board.IsValid({pieceId, r, x, y}) && y > -BOARD_HEIGHT) y--;
            
            if (y <= -BOARD_HEIGHT) continue;
            
            while (board.IsValid({pieceId, r, x, y + 1})) y++;
            
            Piece piece{pieceId, r, x, y};
            if (!board.IsValid(piece)) continue;

            Board next = board;
            next.PlacePiece(piece);
            int lines = next.ClearLines();
            int height = next.GetAggregateHeight();
            int holes = next.GetHoles();
            int bump = next.GetBumpiness();

            double score = lines * lines * weights.w_lines +
                          height * weights.w_height +
                          holes * weights.w_holes +
                          bump * weights.w_bumpiness;

            if (score > best.score) {
                best = {r, x, score};
            }
        }
    }
    return best;
}

double SimulateGame(const HeuristicWeights& weights) {
    Board board;
    int lines = 0, moves = 0;
    while (moves < MAX_MOVES_PER_GAME) {
        int id = Random::Int(1, 7);
        Piece p{id, 0, 3, 0};
        if (board.IsGameOver(p)) break;
        Move m = FindBestMove(board, id, weights);
        p.rotation = m.rotation; p.x = m.x;
        int y = 0;
        while (!board.IsValid({id, m.rotation, m.x, y})) y--;
        p.y = y;
        while (board.IsValid({id, m.rotation, m.x, p.y + 1})) p.y++;
        board.PlacePiece(p);
        lines += board.ClearLines();
        moves++;
    }
    return static_cast<double>(lines);
}

Individual TournamentSelection(const std::vector<Individual>& pop) {
    Individual best{{}, std::numeric_limits<double>::lowest()};
    for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
        const auto& c = pop[Random::Int(0, pop.size()-1)];
        if (c.fitness > best.fitness) best = c;
    }
    return best;
}

HeuristicWeights Crossover(const HeuristicWeights& a, const HeuristicWeights& b) {
    return {
        (a.w_lines + b.w_lines) / 2.0,
        (a.w_height + b.w_height) / 2.0,
        (a.w_holes + b.w_holes) / 2.0,
        (a.w_bumpiness + b.w_bumpiness) / 2.0
    };
}

void Mutate(HeuristicWeights& w) {
    if (Random::Double(0,1) < MUTATION_RATE) w.w_lines += Random::Normal(0, MUTATION_STRENGTH);
    if (Random::Double(0,1) < MUTATION_RATE) w.w_height += Random::Normal(0, MUTATION_STRENGTH);
    if (Random::Double(0,1) < MUTATION_RATE) w.w_holes += Random::Normal(0, MUTATION_STRENGTH);
    if (Random::Double(0,1) < MUTATION_RATE) w.w_bumpiness += Random::Normal(0, MUTATION_STRENGTH);
}

HeuristicWeights RunGeneticAlgorithm() {
    std::vector<Individual> pop(POPULATION_SIZE);
    for (auto& ind : pop) ind.weights = HeuristicWeights::RandomWeights();

    std::cout << "Starting Genetic Algorithm training...\n";
    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        for (auto& ind : pop) {
            double f = 0;
            for (int i = 0; i < NUM_GAMES_PER_FITNESS_TEST; ++i)
                f += SimulateGame(ind.weights);
            ind.fitness = f / NUM_GAMES_PER_FITNESS_TEST;
        }

        std::sort(pop.begin(), pop.end(), std::greater<Individual>());

        std::vector<Individual> newPop;
        int elite = POPULATION_SIZE * ELITISM_RATE;
        for (int i = 0; i < elite; ++i) newPop.push_back(pop[i]);

        while (newPop.size() < POPULATION_SIZE) {
            auto p1 = TournamentSelection(pop);
            auto p2 = TournamentSelection(pop);
            HeuristicWeights child = Crossover(p1.weights, p2.weights);
            Mutate(child);
            newPop.push_back({child, 0.0});
        }
        pop = newPop;

        std::cout << "Gen " << gen+1 << ": Best=" << pop[0].fitness << " ";
        pop[0].weights.Print(); std::cout << "\n";
    }
    std::cout << "Training complete!\n";
    return pop[0].weights;
}

void PlayVisibleGame(const HeuristicWeights& w) {
    ClearScreen();
    Board board;
    int score = 0, lines = 0, level = 1;
    bool over = false;

    std::cout << "--- AI Playing Tetris ---\n";
    while (!over) {
        int id = Random::Int(1, 7);
        Piece p{id, 0, 3, 0};
        if (board.IsGameOver(p)) { over = true; break; }
        Move m = FindBestMove(board, id, w);
        p.rotation = m.rotation; p.x = m.x;
        int y = 0;
        while (!board.IsValid({id, m.rotation, m.x, y})) y--;
        p.y = y;
        while (board.IsValid({id, m.rotation, m.x, p.y+1})) p.y++;
        board.PlacePiece(p);
        int cleared = board.ClearLines();
        if (cleared) {
            lines += cleared;
            score += cleared * cleared * 100 * level;
            level = 1 + (lines / 10);
        }
        ClearScreen();
        std::cout << "--- AI Playing ---\n";
        board.Render(score, lines, level);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ClearScreen();
    std::cout << "--- GAME OVER ---\nFinal Score: " << score << "\nFinal Lines: " << lines << "\n";
}

// --- Command Line Interface ---
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
    
    // Handle different modes
    if (playMode && trainMode) {
        std::cerr << "Error: Cannot use both --train and --play simultaneously.\n";
        return 1;
    }
    
    if (playMode) {
        // Load and play only
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
        // Train and save only
        std::cout << "Training new model...\n";
        best = RunGeneticAlgorithm();
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
            best = RunGeneticAlgorithm();
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