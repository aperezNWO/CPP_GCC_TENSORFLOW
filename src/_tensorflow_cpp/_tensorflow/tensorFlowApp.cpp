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
    try {
        auto* game = new GameState();
        game->aiWeights = HeuristicWeights::RandomWeights();
        return game;
    } catch (...) {
        return nullptr; // Return null on failure
    }
}

DLL_EXPORT void TETRIS_DestroyGame(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    if (game) delete game;
}

// ==================== Game Control ====================
DLL_EXPORT void TETRIS_Reset(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    if (!game) return;
    
    game->grid = {};
    game->score = 0;
    game->lines = 0;
    game->level = 1;
    game->gameOver = false;
    game->nextPiece = 0;
}

DLL_EXPORT void TETRIS_Step(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    if (!game || game->gameOver) return;

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
    if (game) game->autoPlay = !game->autoPlay;
}

// ==================== State Query ====================
DLL_EXPORT int TETRIS_GetScore(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    return game ? game->score : 0;
}

DLL_EXPORT int TETRIS_GetLines(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    return game ? game->lines : 0;
}

DLL_EXPORT int TETRIS_GetLevel(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    return game ? game->level : 1;
}

DLL_EXPORT int TETRIS_GetNextPiece(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    return game ? game->nextPiece : 0;
}

DLL_EXPORT int TETRIS_IsGameOver(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    return (game && game->gameOver) ? 1 : 0;
}

DLL_EXPORT const int* TETRIS_GetBoardMatrix(TETRIS_Game gameHandle) {
    auto* game = static_cast<GameState*>(gameHandle);
    return game ? &game->grid[0][0] : nullptr;
}

// ==================== AI Functions ====================
DLL_EXPORT void TETRIS_TrainAI(const char* weightsFile, int generations) {
    if (!weightsFile) return;
    if (generations <= 0) generations = NUM_GENERATIONS;
    try {
        auto weights = Engine::RunGeneticAlgorithm();
        Engine::SaveWeights(weights, weightsFile);
    } catch (...) {
        // Silently fail - in production, add error handling
    }
}

DLL_EXPORT void TETRIS_LoadAI(TETRIS_Game gameHandle, const char* weightsFile) {
    auto* game = static_cast<GameState*>(gameHandle);
    if (!game || !weightsFile) return;
    Engine::LoadWeights(game->aiWeights, weightsFile);
}

DLL_EXPORT void TETRIS_GetAIWeights(TETRIS_Game gameHandle, double* weightsOut) {
    if (!weightsOut) return;
    auto* game = static_cast<GameState*>(gameHandle);
    if (!game) return;
    
    weightsOut[0] = game->aiWeights.w_lines;
    weightsOut[1] = game->aiWeights.w_height;
    weightsOut[2] = game->aiWeights.w_holes;
    weightsOut[3] = game->aiWeights.w_bumpiness;
}

DLL_EXPORT void TETRIS_SetAIWeights(TETRIS_Game gameHandle, const double* weightsIn) {
    if (!weightsIn) return;
    auto* game = static_cast<GameState*>(gameHandle);
    if (!game) return;
    
    game->aiWeights.w_lines = weightsIn[0];
    game->aiWeights.w_height = weightsIn[1];
    game->aiWeights.w_holes = weightsIn[2];
    game->aiWeights.w_bumpiness = weightsIn[3];
}
