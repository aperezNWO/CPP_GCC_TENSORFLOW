#include "TetrisEngineStateless.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <iomanip>

// ==================== Random Implementation ====================
namespace Engine::Random {
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

// ==================== Tetromino Shapes Definition ====================
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

const Shape& Piece::GetShape() const {
    return TETROMINO_SHAPES[typeId - 1][rotation];
}

// ==================== HeuristicWeights Implementation ====================
HeuristicWeights HeuristicWeights::RandomWeights() {
    return {
        Engine::Random::Double(0.1, 1.0),
        Engine::Random::Double(-1.0, -0.1),
        Engine::Random::Double(-1.0, -0.1),
        Engine::Random::Double(-1.0, -0.1)
    };
}

void HeuristicWeights::Print() const {
    std::cout << std::fixed << std::setprecision(4)
              << "Lines: " << w_lines
              << ", Height: " << w_height
              << ", Holes: " << w_holes
              << ", Bumpiness: " << w_bumpiness;
}

// ==================== Stateless Game Logic ====================
bool Engine::IsValid(const BoardGrid& grid, const Piece& piece) {
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

BoardGrid Engine::PlacePiece(const BoardGrid& grid, const Piece& piece) {
    BoardGrid newGrid = grid;
    const auto& shape = piece.GetShape();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (shape[r][c] != 0) {
                int py = piece.y + r;
                int px = piece.x + c;
                if (py >= 0 && py < BOARD_HEIGHT && px >= 0 && px < BOARD_WIDTH) {
                    newGrid[py][px] = piece.typeId;
                }
            }
        }
    }
    return newGrid;
}

int Engine::ClearLines(BoardGrid& grid) {
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

bool Engine::IsGameOver(const BoardGrid& grid, const Piece& piece) {
    return !Engine::IsValid(grid, piece);
}

int Engine::GetAggregateHeight(const BoardGrid& grid) {
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

int Engine::GetHoles(const BoardGrid& grid) {
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

int Engine::GetBumpiness(const BoardGrid& grid) {
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

Move Engine::FindBestMove(const BoardGrid& grid, int pieceId, const HeuristicWeights& weights) {
    Move best = {0, 0, std::numeric_limits<double>::lowest()};
    for (int r = 0; r < 4; ++r) {
        for (int x = -3; x < BOARD_WIDTH + 3; ++x) {
            int y = 0;
            Piece testPiece{pieceId, r, x, y};
            while (!Engine::IsValid(grid, testPiece) && y > -BOARD_HEIGHT) {
                y--;
                testPiece.y = y;
            }
            
            if (y <= -BOARD_HEIGHT) continue;
            
            while (Engine::IsValid(grid, {pieceId, r, x, y + 1})) y++;
            
            Piece finalPiece{pieceId, r, x, y};
            if (!Engine::IsValid(grid, finalPiece)) continue;

            BoardGrid nextGrid = Engine::PlacePiece(grid, finalPiece);
            int lines = Engine::ClearLines(nextGrid);
            int height = Engine::GetAggregateHeight(nextGrid);
            int holes = Engine::GetHoles(nextGrid);
            int bump = Engine::GetBumpiness(nextGrid);

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

double Engine::SimulateGame(const HeuristicWeights& weights) {
    BoardGrid grid = {}; // Initialize empty grid
    int lines = 0, moves = 0;
    int nextPiece = Engine::Random::Int(1, 7);
    
    while (moves < MAX_MOVES_PER_GAME) {
        int currentPiece = nextPiece;
        nextPiece = Engine::Random::Int(1, 7);
        
        Piece p{currentPiece, 0, 3, 0};
        if (Engine::IsGameOver(grid, p)) break;
        
        Move m = Engine::FindBestMove(grid, currentPiece, weights);
        p.rotation = m.rotation; p.x = m.x;
        
        int y = 0;
        while (!Engine::IsValid(grid, {currentPiece, m.rotation, m.x, y})) y--;
        p.y = y;
        while (Engine::IsValid(grid, {currentPiece, m.rotation, m.x, p.y + 1})) p.y++;
        
        grid = Engine::PlacePiece(grid, p);
        lines += Engine::ClearLines(grid);
        moves++;
    }
    return static_cast<double>(lines);
}

// ==================== Genetic Algorithm Implementation ====================
Individual Engine::TournamentSelection(const std::vector<Individual>& pop) {
    Individual best{{}, std::numeric_limits<double>::lowest()};
    for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
        const auto& c = pop[Engine::Random::Int(0, pop.size()-1)];
        if (c.fitness > best.fitness) best = c;
    }
    return best;
}

HeuristicWeights Engine::Crossover(const HeuristicWeights& a, const HeuristicWeights& b) {
    return {
        (a.w_lines + b.w_lines) / 2.0,
        (a.w_height + b.w_height) / 2.0,
        (a.w_holes + b.w_holes) / 2.0,
        (a.w_bumpiness + b.w_bumpiness) / 2.0
    };
}

void Engine::Mutate(HeuristicWeights& w) {
    if (Engine::Random::Double(0,1) < MUTATION_RATE) w.w_lines += Engine::Random::Normal(0, MUTATION_STRENGTH);
    if (Engine::Random::Double(0,1) < MUTATION_RATE) w.w_height += Engine::Random::Normal(0, MUTATION_STRENGTH);
    if (Engine::Random::Double(0,1) < MUTATION_RATE) w.w_holes += Engine::Random::Normal(0, MUTATION_STRENGTH);
    if (Engine::Random::Double(0,1) < MUTATION_RATE) w.w_bumpiness += Engine::Random::Normal(0, MUTATION_STRENGTH);
}

HeuristicWeights Engine::RunGeneticAlgorithm() {
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