/*

//
TOOLCHAIN : C:\msys64\uctr64\bin (CHANGE PATH)

// INSTALLERS
https://www.tensorflow.org/install/lang_c?hl=es-419


// NON STATIC - REFERENCE - INHERITANCE

g++ -std=c++20 -I"include" -L"lib" -shared -m64 -o TensorFlowAppCPP.dll tensorFlowApp.cpp -ltensorflow -lAlgorithm -Wl,--subsystem,windows -DALGORITHM_EXPORTS

// COMPILE FROM ROOT (above _tensorflow folder)

g++ -std=c++20 -I"include" -L"lib" -shared -m64 -o "__dist/TensorFlowAppCPP.dll" "_tensorflow/tensorFlowApp.cpp"  -ltensorflow -lAlgorithm -Wl,--subsystem,windows -DALGORITHM_EXPORTS   


*/

#include "../include/tensorFlowApp.h"

//
TensorFlowApp::TensorFlowApp(): Algorithm(false)
{
     //
     this->ReadConfigFile("tensorflow.ini");
}
//
TensorFlowApp::~TensorFlowApp()
{
    //
}

//
const char*  TensorFlowApp::GetTensorFlowAPIVersion()
{
	//
  	return TF_Version(); // Return the TensorFlow version directly;
}

//
std::string TensorFlowApp::GetTensorFlowAppVersion()
{
    auto it = this->configMap.find("DLL_VERSION");
    if (it != this->configMap.end()) {
        return it->second;
    }
    return "UNKNOWN"; 
}

/////////////////////////////////////////////////////////////////////
// DLL ENTRY POINTS
/////////////////////////////////////////////////////////////////////
//
DLL_EXPORT const char* GetTensorFlowAPIVersion() 
{
    static std::string versionCache;
    {
        std::unique_ptr<TensorFlowApp> app = std::make_unique<TensorFlowApp>();
        versionCache = app->GetTensorFlowAPIVersion(); // TF_Version()
    }
    return versionCache.c_str();
}
//
DLL_EXPORT const char* GetTensorFlowAppVersion() {
    static std::string version;
    if (version.empty()) {
        TensorFlowApp app;
        version = app.GetTensorFlowAppVersion();
    }
    return version.c_str();
}
//
DLL_EXPORT const char* GetCPPSTDVersion()
{
    static std::string version;
    if (version.empty()) {
        TensorFlowApp app;
        version = app.GetCPPSTDVersion(__cplusplus);
    }
    return version.c_str();
}
//
DLL_EXPORT bool PlayTicTacToeGameWithHistory(TicTacToeResultOnline* result, int aiMode, double temperature) 
{
    try {
    	//
        if (!result) return false;
        //
        return RunTicTacToeSelfPlay(*result, aiMode, temperature);
    } catch (...) {
        return false;
    }
}
////////////////////////////////////////////////////////////////////////////////
// TETRIS END POINTS
////////////////////////////////////////////////////////////////////////////////

// ==================== Constants ====================
DLL_EXPORT int TETRIS_GetBoardWidth() { return BOARD_WIDTH; }
DLL_EXPORT int TETRIS_GetBoardHeight() { return BOARD_HEIGHT; }

// ==================== Game Session ====================
DLL_EXPORT TETRIS_Game TETRIS_CreateGame() {
    auto* game = new GameState();
    // Load default AI weights or random
    game->aiWeights = HeuristicWeights::RandomWeights();
    return game;
}

DLL_EXPORT void TETRIS_DestroyGame(TETRIS_Game game) {
    delete static_cast<GameState*>(game);
}

// ==================== Game Control ====================
DLL_EXPORT void TETRIS_Reset(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    game->grid = {}; // Clear grid
    game->score = 0;
    game->lines = 0;
    game->level = 1;
    game->gameOver = false;
}

DLL_EXPORT void TETRIS_Step(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    if (game->gameOver) return;

    // Generate piece if needed
    if (game->nextPiece == 0) {
        game->nextPiece = Engine::Random::Int(1, 7);
    }

    int currentPiece = game->nextPiece;
    game->nextPiece = Engine::Random::Int(1, 7);
    
    Piece p{currentPiece, 0, 3, 0};
    if (Engine::IsGameOver(game->grid, p)) {
        game->gameOver = true;
        return;
    }

    Move m = Engine::FindBestMove(game->grid, currentPiece, game->aiWeights);
    p.rotation = m.rotation; p.x = m.x;
    
    int y = 0;
    while (!Engine::IsValid(game->grid, {currentPiece, m.rotation, m.x, y})) y--;
    p.y = y;
    while (Engine::IsValid(game->grid, {currentPiece, m.rotation, m.x, p.y + 1})) p.y++;

    game->grid = Engine::PlacePiece(game->grid, p);
    int cleared = Engine::ClearLines(game->grid);
    if (cleared) {
        game->lines += cleared;
        game->score += cleared * cleared * 100 * game->level;
        game->level = 1 + (game->lines / 10);
    }
}

DLL_EXPORT void TETRIS_ToggleAutoPlay(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    game->autoPlay = !game->autoPlay;
    // In a real implementation, you'd run a loop in another thread
}

// ==================== State Query ====================
DLL_EXPORT int TETRIS_GetScore(TETRIS_Game gameHandle) {
    return static_cast<GameState*>(gameHandle)->score;
}

DLL_EXPORT int TETRIS_GetLines(TETRIS_Game gameHandle) {
    return static_cast<GameState*>(gameHandle)->lines;
}

DLL_EXPORT int TETRIS_GetLevel(TETRIS_Game gameHandle) {
    return static_cast<GameState*>(gameHandle)->level;
}

DLL_EXPORT int TETRIS_GetNextPiece(TETRIS_Game gameHandle) {
    return static_cast<GameState*>(gameHandle)->nextPiece;
}

DLL_EXPORT int TETRIS_IsGameOver(TETRIS_Game gameHandle) {
    return static_cast<GameState*>(gameHandle)->gameOver ? 1 : 0;
}

DLL_EXPORT const int* TETRIS_GetBoardMatrix(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    // Return pointer to first element of 2D array
    return &game->grid[0][0];
}

// ==================== AI Functions ====================
DLL_EXPORT void TETRIS_TrainAI(const char* weightsFile, int generations) {
    if (generations <= 0) generations = NUM_GENERATIONS;
    auto weights = Engine::RunGeneticAlgorithm();
    Engine::SaveWeights(weights, weightsFile);
}

DLL_EXPORT void TETRIS_LoadAI(TETRIS_Game gameHandle, const char* weightsFile) {
    auto* game = static_cast<GameState*>(gameHandle);
    Engine::LoadWeights(game->aiWeights, weightsFile);
}

DLL_EXPORT void TETRIS_GetAIWeights(TETRIS_Game gameHandle, double* weightsOut) {
    auto* game = static_cast<GameState*>(gameHandle);
    weightsOut[0] = game->aiWeights.w_lines;
    weightsOut[1] = game->aiWeights.w_height;
    weightsOut[2] = game->aiWeights.w_holes;
    weightsOut[3] = game->aiWeights.w_bumpiness;
}

DLL_EXPORT void TETRIS_SetAIWeights(TETRIS_Game gameHandle, const double* weightsIn) {
    auto* game = static_cast<GameState*>(gameHandle);
    game->aiWeights.w_lines = weightsIn[0];
    game->aiWeights.w_height = weightsIn[1];
    game->aiWeights.w_holes = weightsIn[2];
    game->aiWeights.w_bumpiness = weightsIn[3];
}
