/*

g++ -std=c++17 -o test_sortbenchmark_exe.exe test_sortbenchmark_exe.cpp

*/
// sort_demo_with_progress.cpp
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <chrono>   // For ETA
#include <limits>   // For input cleanup

// For cross-platform screen clear
#ifdef _WIN32
    #include <windows.h>
    void clearScreen() { system("cls"); }
#else
    void clearScreen() { system("clear"); }
#endif

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

// Sigmoid function with clamping
double sigmoid(double x) {
    if (x < -700.0) return 0.0;
    if (x > 700.0) return 1.0;
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of sigmoid
double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

class NeuralNetwork {
public:
    std::vector<double> input, hidden, output;
    std::vector<std::vector<double>> weights_ih, weights_ho;
    std::vector<double> bias_h, bias_o;
    double learningRate = 0.01;

    NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        : input(inputSize), hidden(hiddenSize), output(outputSize),
          weights_ih(hiddenSize, std::vector<double>(inputSize)),
          weights_ho(outputSize, std::vector<double>(hiddenSize)),
          bias_h(hiddenSize), bias_o(outputSize) {

        if (inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0) {
            throw std::invalid_argument("Layer sizes must be positive");
        }

        for (auto& row : weights_ih)
            for (double& w : row)
                w = dis(gen) * 2.0 - 1.0;

        for (auto& row : weights_ho)
            for (double& w : row)
                w = dis(gen) * 2.0 - 1.0;

        for (double& b : bias_h)
            b = dis(gen) * 2.0 - 1.0;

        for (double& b : bias_o)
            b = dis(gen) * 2.0 - 1.0;
    }

    void forward(const std::vector<double>& x) {
        if (x.size() != input.size())
            throw std::out_of_range("Input size mismatch");

        input = x;

        for (size_t i = 0; i < hidden.size(); ++i) {
            double activation = bias_h.at(i);
            const auto& w_row = weights_ih.at(i);
            for (size_t j = 0; j < input.size(); ++j) {
                activation += w_row.at(j) * input.at(j);
            }
            hidden[i] = sigmoid(activation);
        }

        for (size_t i = 0; i < output.size(); ++i) {
            double activation = bias_o.at(i);
            const auto& w_row = weights_ho.at(i);
            for (size_t j = 0; j < hidden.size(); ++j) {
                activation += w_row.at(j) * hidden.at(j);
            }
            output[i] = sigmoid(activation);
        }
    }

    void backprop(const std::vector<double>& target) {
        if (target.size() != output.size())
            throw std::out_of_range("Target size mismatch");

        std::vector<double> grad_output(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            double error = target.at(i) - output.at(i);
            grad_output[i] = error * sigmoidDerivative(output.at(i));
        }

        std::vector<double> grad_hidden(hidden.size(), 0.0);
        for (size_t i = 0; i < hidden.size(); ++i) {
            for (size_t j = 0; j < output.size(); ++j) {
                grad_hidden[i] += grad_output[j] * weights_ho[j][i];
            }
            grad_hidden[i] *= sigmoidDerivative(hidden.at(i));
        }

        for (size_t i = 0; i < output.size(); ++i) {
            bias_o.at(i) += learningRate * grad_output.at(i);
            for (size_t j = 0; j < hidden.size(); ++j) {
                weights_ho[i][j] += learningRate * grad_output.at(i) * hidden.at(j);
            }
        }

        for (size_t i = 0; i < hidden.size(); ++i) {
            bias_h.at(i) += learningRate * grad_hidden.at(i);
            for (size_t j = 0; j < input.size(); ++j) {
                weights_ih[i][j] += learningRate * grad_hidden.at(i) * input.at(j);
            }
        }
    }

    bool saveModel(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        auto writeMatrix = [&](const std::vector<std::vector<double>>& mat) {
            file << mat.size() << " " << mat[0].size() << "\n";
            for (const auto& row : mat) {
                for (double w : row) file << std::setprecision(10) << w << " ";
                file << "\n";
            }
        };

        auto writeVector = [&](const std::vector<double>& vec) {
            file << vec.size() << "\n";
            for (double v : vec) file << std::setprecision(10) << v << " ";
            file << "\n";
        };

        writeMatrix(weights_ih);
        writeMatrix(weights_ho);
        writeVector(bias_h);
        writeVector(bias_o);
        file.close();
        return true;
    }

    bool loadModel(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        auto readMatrix = [&](std::vector<std::vector<double>>& mat) -> bool {
            int rows, cols;
            if (!(file >> rows >> cols)) return false;
            mat.resize(rows, std::vector<double>(cols));
            for (auto& row : mat) {
                for (double& w : row) {
                    if (!(file >> w)) return false;
                }
            }
            return true;
        };

        auto readVector = [&](std::vector<double>& vec) -> bool {
            int size;
            if (!(file >> size)) return false;
            vec.resize(size);
            for (double& v : vec) {
                if (!(file >> v)) return false;
            }
            return true;
        };

        if (!readMatrix(weights_ih)) goto fail;
        if (!readMatrix(weights_ho)) goto fail;
        if (!readVector(bias_h))     goto fail;
        if (!readVector(bias_o))     goto fail;

        file.close();
        return true;

    fail:
        file.close();
        return false;
    }
};

// Normalize: [min..max] → [0..1]
std::vector<double> normalize(const std::vector<double>& arr) {
    if (arr.empty()) throw std::invalid_argument("Empty array");
    double min_val = *std::min_element(arr.begin(), arr.end());
    double max_val = *std::max_element(arr.begin(), arr.end());
    std::vector<double> norm = arr;
    if (max_val != min_val) {
        for (double& x : norm) x = (x - min_val) / (max_val - min_val);
    } else {
        std::fill(norm.begin(), norm.end(), 0.5);
    }
    return norm;
}

// Denormalize: [0..1] → [orig_min..orig_max]
std::vector<double> denormalize(const std::vector<double>& norm, double orig_min, double orig_max) {
    std::vector<double> orig = norm;
    double range = orig_max - orig_min;
    if (range > 0) {
        for (double& x : orig) x = x * range + orig_min;
    }
    return orig;
}

// Train with progress bar and ETA
void trainSortingModel(NeuralNetwork& nn, int inputSize, int numSamples = 1000, int total_epochs = 5000) {
    std::cout << "📊 Generating " << numSamples << " training samples...\n";

    std::vector<std::vector<double>> X_train, y_train;
    for (int n = 0; n < numSamples; ++n) {
        std::vector<double> arr(inputSize);
        for (int i = 0; i < inputSize; ++i) arr[i] = i + 1;
        std::shuffle(arr.begin(), arr.end(), gen);

        std::vector<double> sorted = arr;
        std::sort(sorted.begin(), sorted.end());

        X_train.push_back(normalize(arr));
        y_train.push_back(normalize(sorted));
    }

    std::cout << "🚀 Starting training (" << total_epochs << " epochs)...\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        double total_error = 0.0;
        std::vector<int> indices(X_train.size());
        for (int i = 0; i < indices.size(); ++i) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), gen);

        for (int idx : indices) {
            try {
                nn.forward(X_train[idx]);
                nn.backprop(y_train[idx]);

                for (size_t i = 0; i < y_train[idx].size(); ++i) {
                    double err = y_train[idx][i] - nn.output[i];
                    total_error += err * err;
                }
            } catch (...) { /* Ignore individual errors */ }
        }

        double avg_error = total_error / X_train.size();

        // Update every 250 epochs
        if (epoch % 250 == 0 || epoch == total_epochs - 1) {
            int percent = (epoch * 100) / total_epochs;

            // Progress bar (20 blocks)
            int bar_width = 20;
            std::string bar(bar_width, '░');
            for (int i = 0; i < percent * bar_width / 100; ++i) bar[i] = '█';

            // Estimate time remaining
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            int eta_seconds = (elapsed * (total_epochs - epoch)) / (epoch + 1);
            int mins = eta_seconds / 60;
            int secs = eta_seconds % 60;

            std::cout << "Epoch " << epoch << "/" << total_epochs
                      << " [" << bar << "] " << percent << "%"
                      << " | Error: " << avg_error
                      << " | ETA: ~" << mins << "m" << secs << "s\n";
        }
    }
}

// Run one sorting demo
void doSortDemo(NeuralNetwork& nn) {
    const int SIZE = 25;
    std::vector<double> test_input(SIZE);
    for (int i = 0; i < SIZE; ++i) test_input[i] = i + 1;
    std::shuffle(test_input.begin(), test_input.end(), gen);

    try {
        auto test_norm = normalize(test_input);
        nn.forward(test_norm);
        auto pred_raw = denormalize(nn.output, 1.0, 25.0);

        std::cout << "\n📥 Input:  ";
        for (int i = 0; i < SIZE; ++i) {
            std::cout << static_cast<int>(test_input[i]) << " ";
            if ((i + 1) % 10 == 0 && i > 0) std::cout << "\n           ";
        }

        std::cout << "\n📤 Sorted: ";
        for (int i = 0; i < SIZE; ++i) {
            std::cout << static_cast<int>(std::round(pred_raw[i])) << " ";
            if ((i + 1) % 10 == 0 && i > 0) std::cout << "\n           ";
        }
        std::cout << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ Prediction failed: " << e.what() << "\n";
    }
}

int main() {
    const std::string MODEL_FILE = "sort_model_25.txt";
    const int INPUT_SIZE = 25;
    const int HIDDEN_SIZE = 128;
    const int OUTPUT_SIZE = 25;

    try {
        NeuralNetwork nn(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

        clearScreen();
        std::cout << "🧠 Sorting Neural Network Demo\n";
        std::cout << "================================\n\n";

        // Load or train model
        if (std::ifstream(MODEL_FILE)) {
            std::cout << "💾 Loading pre-trained model '" << MODEL_FILE << "'...\n";
            if (nn.loadModel(MODEL_FILE)) {
                std::cout << "✅ Model loaded successfully!\n\n";
            } else {
                std::cout << "❌ Failed to load model. Training new one...\n";
                trainSortingModel(nn, INPUT_SIZE, 1000, 5000);
                if (nn.saveModel(MODEL_FILE)) {
                    std::cout << "✅ New model trained and saved.\n\n";
                } else {
                    std::cerr << "⚠️ Warning: Could not save model.\n\n";
                }
            }
        } else {
            std::cout << "⚠️ Model '" << MODEL_FILE << "' not found.\n";
            trainSortingModel(nn, INPUT_SIZE, 1000, 5000);
            if (nn.saveModel(MODEL_FILE)) {
                std::cout << "✅ Model trained and saved as '" << MODEL_FILE << "'.\n\n";
            } else {
                std::cerr << "⚠️ Warning: Could not save model.\n\n";
            }
        }

        char choice;
		do {
		    doSortDemo(nn);
		
		    std::cout << "🔁 Run another sort? (y/n): ";
		
		    if (!(std::cin >> choice)) {
		        std::cin.clear();
		        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		        std::cout << "\nInput error. Exiting.\n";
		        break;
		    }
		
		    choice = std::tolower(static_cast<unsigned char>(choice));
		    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Clear buffer
		
		    if (choice == 'y') {
		        clearScreen();
		    } else if (choice != 'n') {
		        std::cout << "⚠️  Invalid choice. Assuming 'n'. Goodbye!\n";
		        break;
		    }
		
		} while (choice == 'y');
		
		
    } catch (const std::exception& e) {
        std::cerr << "💥 An error occurred: " << e.what() << "\n";
        return 1;
    }

    return 0;
}