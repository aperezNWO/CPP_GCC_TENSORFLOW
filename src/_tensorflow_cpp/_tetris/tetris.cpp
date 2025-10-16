/*

SELF PLAYING TIC TAC TOE.

g++ -std=c++17 -o tetris.exe tetris.cpp -mconsole

*/
#include <iostream>
#include <vector>
#include <windows.h>
#include <ctime>
#include <algorithm>
#include <cmath>

using namespace std;

// === Constantes ===
const int BOARD_WIDTH = 10;
const int BOARD_HEIGHT = 20;
const char EMPTY_CELL = ' ';

// === Formas de Tetrominós ===
vector<vector<vector<vector<int>>>> TETROMINOES = {
    // I
    {{{0,0,0,0},
      {1,1,1,1},
      {0,0,0,0},
      {0,0,0,0}}},

    // O
    {{{1,1},
      {1,1}}},

    // T
    {{{0,1,0},
      {1,1,1},
      {0,0,0}}},

    // S
    {{{0,1,1},
      {1,1,0},
      {0,0,0}}},

    // Z
    {{{1,1,0},
      {0,1,1},
      {0,0,0}}},

    // J
    {{{1,0,0},
      {1,1,1},
      {0,0,0}}},

    // L
    {{{0,0,1},
      {1,1,1},
      {0,0,0}}}
};

// === Colores de consola para cada pieza ===
const int COLORS[] = {9, 14, 13, 10, 12, 1, 6}; // Azul, Amarillo, Magenta, Verde, Rojo, Azul oscuro, Cyan

// === Tablero del juego ===
vector<vector<char>> board(BOARD_HEIGHT, vector<char>(BOARD_WIDTH, EMPTY_CELL));
vector<vector<int>> colorBoard(BOARD_HEIGHT, vector<int>(BOARD_WIDTH, 7)); // 7 = blanco

// === Pieza actual ===
struct Piece {
    int x, y;           // posición
    int type;           // tipo de pieza (0-6)
    int rotation;       // rotación actual
};

Piece currentPiece = {-1, -1, -1, -1};
bool gameOver = false;
int score = 0;

// === Dibujar el tablero con colores ===
void drawBoard() {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD coord = {0, 0};
    SetConsoleCursorPosition(hConsole, coord);

    // Borde superior
    cout << string(BOARD_WIDTH * 2 + 2, '-') << '\n';

    for (int y = 0; y < BOARD_HEIGHT; ++y) {
        cout << '|';
        for (int x = 0; x < BOARD_WIDTH; ++x) {
            SetConsoleTextAttribute(hConsole, colorBoard[y][x]);
            cout << board[y][x] << ' ';
        }
        SetConsoleTextAttribute(hConsole, 7); // Restaurar blanco
        cout << '|' << '\n';
    }

    cout << string(BOARD_WIDTH * 2 + 2, '-') << '\n';
    cout << "Puntuación: " << score << "\n";
    if (gameOver) {
        cout << "¡FIN DEL JUEGO!\n";
    }
}

// === Verificar colisión ===
bool checkCollision(int px, int py, const vector<vector<int>>& shape) {
    for (size_t y = 0; y < shape.size(); ++y) {
        for (size_t x = 0; x < shape[y].size(); ++x) {
            if (shape[y][x]) {
                int nx = px + static_cast<int>(x);
                int ny = py + static_cast<int>(y);

                if (nx < 0 || nx >= BOARD_WIDTH || ny >= BOARD_HEIGHT) {
                    return true;
                }
                if (ny >= 0 && board[ny][nx] != EMPTY_CELL) {
                    return true;
                }
            }
        }
    }
    return false;
}

// === Fijar la pieza en el tablero ===
void lockPiece() {
    if (currentPiece.type < 0 || currentPiece.type >= 7) return;

    const auto& rotations = TETROMINOES[currentPiece.type];
    if (rotations.empty()) return;
    if (static_cast<size_t>(currentPiece.rotation) >= rotations.size()) return;

    const auto& shape = rotations[currentPiece.rotation];
    int color = COLORS[currentPiece.type];

    for (size_t y = 0; y < shape.size(); ++y) {
        for (size_t x = 0; x < shape[y].size(); ++x) {
            if (shape[y][x]) {
                int bx = currentPiece.x + static_cast<int>(x);
                int by = currentPiece.y + static_cast<int>(y);
                if (by >= 0 && by < BOARD_HEIGHT && bx >= 0 && bx < BOARD_WIDTH) {
                    board[by][bx] = '#';
                    colorBoard[by][bx] = color;
                }
            }
        }
    }
}

// === Limpiar líneas completas ===
void clearLines() {
    int linesCleared = 0;
    for (int y = BOARD_HEIGHT - 1; y >= 0; --y) {
        bool full = true;
        for (int x = 0; x < BOARD_WIDTH; ++x) {
            if (board[y][x] == EMPTY_CELL) {
                full = false;
                break;
            }
        }

        if (full) {
            ++linesCleared;
            for (int yy = y; yy > 0; --yy) {
                board[yy] = board[yy - 1];
                colorBoard[yy] = colorBoard[yy - 1];
            }
            fill(board[0].begin(), board[0].end(), EMPTY_CELL);
            fill(colorBoard[0].begin(), colorBoard[0].end(), 7);
            ++y; // Revisar la misma fila nuevamente
        }
    }

    if (linesCleared > 0) {
        score += (linesCleared == 1 ? 100 : linesCleared == 2 ? 300 : linesCleared == 3 ? 500 : 800);
    }
}

// === Generar nueva pieza ===
void spawnPiece() {
    int type = rand() % 7;
    currentPiece = { BOARD_WIDTH / 2 - 1, 0, type, 0 };

    if (type < 0 || type >= 7 || TETROMINOES[type].empty()) {
        gameOver = true;
        return;
    }

    if (checkCollision(currentPiece.x, currentPiece.y, TETROMINOES[type][0])) {
        gameOver = true;
    }
}

// === Renderizar o eliminar temporalmente la pieza ===
void renderPiece(bool place) {
    if (currentPiece.type < 0 || currentPiece.type >= 7) return;

    const auto& rotations = TETROMINOES[currentPiece.type];
    if (rotations.empty()) return;
    if (static_cast<size_t>(currentPiece.rotation) >= rotations.size()) return;

    const auto& shape = rotations[currentPiece.rotation];
    int color = place ? COLORS[currentPiece.type] : 7;

    for (size_t y = 0; y < shape.size(); ++y) {
        for (size_t x = 0; x < shape[y].size(); ++x) {
            if (shape[y][x]) {
                int bx = currentPiece.x + static_cast<int>(x);
                int by = currentPiece.y + static_cast<int>(y);
                if (by >= 0 && by < BOARD_HEIGHT && bx >= 0 && bx < BOARD_WIDTH) {
                    board[by][bx] = place ? '#' : EMPTY_CELL;
                    colorBoard[by][bx] = color;
                }
            }
        }
    }
}

// === Evaluar calidad del tablero (heurística) ===
double evaluateBoard() {
    vector<int> heights(BOARD_WIDTH, 0);
    int totalHeight = 0;
    int holes = 0;
    int bumpiness = 0;

    // Calcular altura de cada columna
    for (int x = 0; x < BOARD_WIDTH; ++x) {
        for (int y = 0; y < BOARD_HEIGHT; ++y) {
            if (board[y][x] != EMPTY_CELL) {
                heights[x] = BOARD_HEIGHT - y;
                break;
            }
        }
    }

    // Altura total
    for (int h : heights) totalHeight += h;

    // Contar agujeros
    for (int x = 0; x < BOARD_WIDTH; ++x) {
        bool blockFound = false;
        for (int y = 0; y < BOARD_HEIGHT; ++y) {
            if (blockFound && board[y][x] == EMPTY_CELL) holes++;
            if (board[y][x] != EMPTY_CELL) blockFound = true;
        }
    }

    // Desigualdad entre columnas adyacentes
    for (int i = 0; i < BOARD_WIDTH - 1; ++i) {
        bumpiness += abs(heights[i] - heights[i+1]);
    }

    // Heurística ajustada (valores clásicos de literatura)
    return -0.51 * totalHeight - 0.76 * holes - 0.36 * bumpiness;
}

// === Encontrar mejor movimiento posible ===
Piece findBestMove() {
    double bestScore = -1e9;
    Piece bestPiece = currentPiece;
    Piece original = currentPiece;

    for (int rot = 0; rot < 4; ++rot) {
        for (int offsetX = -5; offsetX <= 5; ++offsetX) {
            currentPiece = original;
            currentPiece.rotation = rot % TETROMINOES[currentPiece.type].size();
            currentPiece.x += offsetX;

            // Simular caída hasta el fondo
            while (!checkCollision(currentPiece.x, currentPiece.y + 1, TETROMINOES[currentPiece.type][currentPiece.rotation])) {
                currentPiece.y++;
            }

            // Si es un movimiento válido, evaluarlo
            if (!checkCollision(currentPiece.x, currentPiece.y, TETROMINOES[currentPiece.type][currentPiece.rotation])) {
                renderPiece(true);
                double score = evaluateBoard();
                renderPiece(false);

                if (score > bestScore) {
                    bestScore = score;
                    bestPiece = currentPiece;
                }
            }
        }
    }

    return bestPiece;
}

// === Función principal ===
int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    cout << "🟦🟧🟥🟨🟩🟪🟫 TETRIS AUTOJUGABLE 🟫🟪🟩🟨🟥🟧🟦\n";
    cout << "El bot está jugando solo...\n";
    Sleep(2000);
    system("cls");

    while (true) {
        if (gameOver) {
            cout << "¡JUEGO TERMINADO! Puntuación: " << score << "\n";
            cout << "Reiniciando en 2 segundos...\n";
            Sleep(2000);

            // Reiniciar juego
            board.assign(BOARD_HEIGHT, vector<char>(BOARD_WIDTH, EMPTY_CELL));
            colorBoard.assign(BOARD_HEIGHT, vector<int>(BOARD_WIDTH, 7));
            score = 0;
            gameOver = false;
            currentPiece = {-1, -1, -1, -1};
            system("cls");
            continue;
        }

        if (currentPiece.type == -1) {
            spawnPiece();
        }

        // Solo actuar si hay una pieza activa
        if (currentPiece.type >= 0 && currentPiece.type < 7) {
            Piece target = findBestMove();

            // Aplicar rotaciones necesarias
            while (currentPiece.rotation != target.rotation) {
                int numRot = static_cast<int>(TETROMINOES[currentPiece.type].size());
                currentPiece.rotation = (currentPiece.rotation + 1) % numRot;
                Sleep(100);
                drawBoard();
            }

            // Mover horizontalmente
            while (currentPiece.x < target.x) {
                currentPiece.x++;
                Sleep(100);
                drawBoard();
            }
            while (currentPiece.x > target.x) {
                currentPiece.x--;
                Sleep(100);
                drawBoard();
            }

            // Caída rápida al fondo
            while (!checkCollision(currentPiece.x, currentPiece.y + 1, TETROMINOES[currentPiece.type][currentPiece.rotation])) {
                currentPiece.y++;
                Sleep(50);
                drawBoard();
            }

            // Fijar pieza
            renderPiece(true);
            lockPiece();
            clearLines();
            currentPiece = {-1, -1, -1, -1};
        }

        drawBoard();
        Sleep(100); // Pequeña pausa visual
    }

    return 0;
}
