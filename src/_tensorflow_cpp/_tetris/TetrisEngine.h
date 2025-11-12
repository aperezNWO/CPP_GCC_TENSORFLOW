#ifndef TETRIS_ENGINE_H
#define TETRIS_ENGINE_H

#include <array>
#include <random>
#include <limits>
#include <algorithm>

namespace TetrisEngine {

// --- Constants ---
constexpr int BOARD_WIDTH = 10;
constexpr int BOARD_HEIGHT = 20;

// --- Type Definitions ---
using Shape = std::array<std::array<int, 4>, 4>;

// --- Random Number Generation ---
namespace Random {
    inline std::mt19937& Generator() {
        static std::mt19937 gen(std::random_device{}());
        return gen;
    }

    inline int Int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(Generator());
    }

    inline double Double(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(Generator());
    }

    inline double Normal(double mean, double stddev) {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(Generator());
    }
}

// --- Tetromino Definitions ---
const std::array<std::array<Shape, 4>, 7> TETROMINO_SHAPES = {{
    // I Piece (ID: 1)
    std::array<Shape, 4>{{
        Shape{{{0,0,0,0}, {1,1,1,1}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,1,0,0}, {0,1,0,0}, {0,1,0,0}, {0,1,0,0}}},
        Shape{{{0,0,0,0}, {0,0,0,0}, {1,1,1,1}, {0,0,0,0}}},
        Shape{{{0,0,1,0}, {0,0,1,0}, {0,0,1,0}, {0,0,1,0}}}
    }},
    // O Piece (ID: 2)
    std::array<Shape, 4>{{
        Shape{{{0,2,2,0}, {0,2,2,0}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,2,2,0}, {0,2,2,0}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,2,2,0}, {0,2,2,0}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,2,2,0}, {0,2,2,0}, {0,0,0,0}, {0,0,0,0}}}
    }},
    // T Piece (ID: 3)
    std::array<Shape, 4>{{
        Shape{{{0,0,0,0}, {3,3,3,0}, {0,3,0,0}, {0,0,0,0}}},
        Shape{{{0,3,0,0}, {3,3,0,0}, {0,3,0,0}, {0,0,0,0}}},
        Shape{{{0,3,0,0}, {3,3,3,0}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,3,0,0}, {0,3,3,0}, {0,3,0,0}, {0,0,0,0}}}
    }},
    // S Piece (ID: 4)
    std::array<Shape, 4>{{
        Shape{{{0,0,0,0}, {0,4,4,0}, {4,4,0,0}, {0,0,0,0}}},
        Shape{{{0,4,0,0}, {0,4,4,0}, {0,0,4,0}, {0,0,0,0}}},
        Shape{{{0,0,0,0}, {0,4,4,0}, {4,4,0,0}, {0,0,0,0}}},
        Shape{{{0,4,0,0}, {0,4,4,0}, {0,0,4,0}, {0,0,0,0}}}
    }},
    // Z Piece (ID: 5)
    std::array<Shape, 4>{{
        Shape{{{0,0,0,0}, {5,5,0,0}, {0,5,5,0}, {0,0,0,0}}},
        Shape{{{0,0,5,0}, {0,5,5,0}, {0,5,0,0}, {0,0,0,0}}},
        Shape{{{0,0,0,0}, {5,5,0,0}, {0,5,5,0}, {0,0,0,0}}},
        Shape{{{0,0,5,0}, {0,5,5,0}, {0,5,0,0}, {0,0,0,0}}}
    }},
    // J Piece (ID: 6)
    std::array<Shape, 4>{{
        Shape{{{0,0,0,0}, {6,6,6,0}, {0,0,6,0}, {0,0,0,0}}},
        Shape{{{0,6,0,0}, {0,6,0,0}, {6,6,0,0}, {0,0,0,0}}},
        Shape{{{6,0,0,0}, {6,6,6,0}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,6,6,0}, {0,6,0,0}, {0,6,0,0}, {0,0,0,0}}}
    }},
    // L Piece (ID: 7)
    std::array<Shape, 4>{{
        Shape{{{0,0,0,0}, {7,7,7,0}, {7,0,0,0}, {0,0,0,0}}},
        Shape{{{7,7,0,0}, {0,7,0,0}, {0,7,0,0}, {0,0,0,0}}},
        Shape{{{0,0,7,0}, {7,7,7,0}, {0,0,0,0}, {0,0,0,0}}},
        Shape{{{0,7,0,0}, {0,7,0,0}, {0,7,7,0}, {0,0,0,0}}}
    }}
}};

// --- Data Structures ---
struct Piece {
    int typeId = 0;
    int rotation = 0;
    int x = 0;
    int y = 0;

    const Shape& GetShape() const {
        return TETROMINO_SHAPES[typeId - 1][rotation];
    }
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

    static HeuristicWeights RandomWeights() {
        return {
            Random::Double(0.1, 1.0),
            Random::Double(-1.0, -0.1),
            Random::Double(-1.0, -0.1),
            Random::Double(-1.0, -0.1)
        };
    }
};

// --- Board Engine (Pure Logic) ---
class BoardEngine {
private:
    std::array<std::array<int, BOARD_WIDTH>, BOARD_HEIGHT> grid;

public:
    BoardEngine() { Reset(); }

    void Reset() {
        grid.fill({});
    }

    const auto& GetGrid() const { return grid; }

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

    std::vector<int> Serialize() const {
        std::vector<int> state;
        state.reserve(BOARD_WIDTH * BOARD_HEIGHT);
        for (int r = 0; r < BOARD_HEIGHT; ++r) {
            for (int c = 0; c < BOARD_WIDTH; ++c) {
                state.push_back(grid[r][c]);
            }
        }
        return state;
    }
};

// --- AI Evaluation ---
inline Move FindBestMove(const BoardEngine& board, int pieceId, const HeuristicWeights& weights) {
    Move best = {0, 0, std::numeric_limits<double>::lowest()};
    for (int r = 0; r < 4; ++r) {
        for (int x = -3; x < BOARD_WIDTH + 3; ++x) {
            int y = 0;
            Piece testPiece{pieceId, r, x, y};
            while (!board.IsValid(testPiece) && testPiece.y > -BOARD_HEIGHT) {
                testPiece.y--;
            }
            if (testPiece.y <= -BOARD_HEIGHT) continue;
            
            while (board.IsValid({pieceId, r, x, testPiece.y + 1})) {
                testPiece.y++;
            }

            if (!board.IsValid(testPiece)) continue;

            BoardEngine next = board;
            next.PlacePiece(testPiece);
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

// --- C-Style API for WebAssembly ---
extern "C" {
    inline BoardEngine* Board_Create() { return new BoardEngine(); }
    inline void Board_Destroy(BoardEngine* board) { delete board; }
    inline void Board_Reset(BoardEngine* board) { board->Reset(); }
    inline int Board_IsValid(BoardEngine* board, int type, int rot, int x, int y) {
        return board->IsValid({type, rot, x, y}) ? 1 : 0;
    }
    inline void Board_PlacePiece(BoardEngine* board, int type, int rot, int x, int y) {
        board->PlacePiece({type, rot, x, y});
    }
    inline int Board_ClearLines(BoardEngine* board) { return board->ClearLines(); }
    inline int Board_IsGameOver(BoardEngine* board, int type, int rot, int x, int y) {
        return board->IsGameOver({type, rot, x, y}) ? 1 : 0;
    }
    inline int Board_GetAggregateHeight(BoardEngine* board) { return board->GetAggregateHeight(); }
    inline int Board_GetHoles(BoardEngine* board) { return board->GetHoles(); }
    inline int Board_GetBumpiness(BoardEngine* board) { return board->GetBumpiness(); }
    inline void Board_GetGrid(BoardEngine* board, int* outArray) {
        const auto& grid = board->GetGrid();
        for (int r = 0; r < BOARD_HEIGHT; ++r) {
            for (int c = 0; c < BOARD_WIDTH; ++c) {
                outArray[r * BOARD_WIDTH + c] = grid[r][c];
            }
        }
    }
    inline Move Board_FindBestMove(BoardEngine* board, int pieceId, const HeuristicWeights* weights) {
        return FindBestMove(*board, pieceId, *weights);
    }
}

} // namespace TetrisEngine

#endif // TETRIS_ENGINE_H