/*
SELF PLAYING TETRIS WITH GENETIC ALGORITHM SOLVER

Compile from root with:

g++ -std=c++20  -o __test/TetrisBoard.exe  _tetris/TetrisBoard.cpp

*/

#include "TetrisEngine.h"

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


