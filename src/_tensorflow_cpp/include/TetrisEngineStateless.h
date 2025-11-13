#pragma once
#include <array>
#include <vector>
#include <random>
#include <string>
#include <limits>

// ==================== Constants ====================
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

// ==================== Type Definitions ====================
using BoardGrid = std::array<std::array<int, BOARD_WIDTH>, BOARD_HEIGHT>;
using Shape = std::array<std::array<int, 4>, 4>;

struct Piece {
    int typeId;
    int rotation;
    int x;
    int y;
    const Shape& GetShape() const;
};

struct Move {
    int rotation = 0;
    int x = 0;
    double score = 0.0;
};

struct HeuristicWeights {
    double w_lines = 0.0;
    double w_height = 0.0;
    double w_holes = 0.0;
    double w_bumpiness = 0.0;
    
    static HeuristicWeights RandomWeights();
    void Print() const;
};

struct Individual {
    HeuristicWeights weights;
    double fitness = 0.0;
    bool operator>(const Individual& other) const { return fitness > other.fitness; }
};

// Tetromino shapes (defined in .cpp)
extern const std::array<std::array<Shape, 4>, 7> TETROMINO_SHAPES;

// ==================== Engine Functions (Pure Logic) ====================
namespace Engine {
    // Random utilities
    namespace Random {
        extern std::mt19937 generator;
        int Int(int min, int max);
        double Double(double min, double max);
        double Normal(double mean, double stddev);
    }

    // Core game logic - all state passed via parameters
    bool IsValid(const BoardGrid& grid, const Piece& piece);
    BoardGrid PlacePiece(const BoardGrid& grid, const Piece& piece);
    int ClearLines(BoardGrid& grid); // Modifies grid in-place for efficiency
    bool IsGameOver(const BoardGrid& grid, const Piece& piece);
    int GetAggregateHeight(const BoardGrid& grid);
    int GetHoles(const BoardGrid& grid);
    int GetBumpiness(const BoardGrid& grid);
    Move FindBestMove(const BoardGrid& grid, int pieceId, const HeuristicWeights& weights);
    double SimulateGame(const HeuristicWeights& weights);
    
    // Genetic Algorithm operations
    Individual TournamentSelection(const std::vector<Individual>& pop);
    HeuristicWeights Crossover(const HeuristicWeights& a, const HeuristicWeights& b);
    void Mutate(HeuristicWeights& w);
    HeuristicWeights RunGeneticAlgorithm();
}
