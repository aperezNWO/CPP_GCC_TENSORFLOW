/*

g++ -std=c++17 -o test_sortbenchmark_rank.exe test_sortbenchmark_rank.cpp

*/

// rank_based_sort.cpp
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>

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
    return 1.0 / (1.0 + exp(-x));
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
    double learningRate = 0.05; // Higher LR for faster convergence

    NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        : input(inputSize), hidden(hiddenSize), output(outputSize),
          weights_ih(hiddenSize, std::vector<double>(inputSize)),
          weights_ho(outputSize, std::vector<double>(hiddenSize)),
          bias_h(hiddenSize), bias_o(outputSize) {

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
        input = x;
        for (int i = 0; i < hidden.size(); ++i) {
            double activation = bias_h[i];
            for (int j = 0; j < input.size(); ++j) {
                activation += weights_ih[i][j] * input[j];
            }
            hidden[i] = sigmoid(activation);
        }

        for (int i = 0; i < output.size(); ++i) {
            double activation = bias_o[i];
            for (int j = 0; j < hidden.size(); ++j) {
                activation += weights_ho[i][j] * hidden[j];
            }
            output[i] = sigmoid(activation);
        }
    }

    void backprop(const std::vector<double>& target) {
        std::vector<double> grad_output(output.size());
        for (int i = 0; i < output.size(); ++i) {
            double error = target[i] - output[i];
            grad_output[i] = error * sigmoidDerivative(output[i]);
        }

        std::vector<double> grad_hidden(hidden.size(), 0);
        for (int i = 0; i < hidden.size(); ++i) {
            for (int j = 0; j < output.size(); ++j) {
                grad_hidden[i] += grad_output[j] * weights_ho[j][i];
            }
            grad_hidden[i] *= sigmoidDerivative(hidden[i]);
        }

        // Update output layer
        for (int i = 0; i < output.size(); ++i) {
            bias_o[i] += learningRate * grad_output[i];
            for (int j = 0; j < hidden.size(); ++j) {
                weights_ho[i][j] += learningRate * grad_output[i] * hidden[j];
            }
        }

        // Update hidden layer
        for (int i = 0; i < hidden.size(); ++i) {
            bias_h[i] += learningRate * grad_hidden[i];
            for (int j = 0; j < input.size(); ++j) {
                weights_ih[i][j] += learningRate * grad_hidden[i] * input[j];
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

// Normalize input to [0,1]
std::vector<double> normalize(const std::vector<double>& arr) {
    double min_val = *std::min_element(arr.begin(), arr.end());
    double max_val = *std::max_element(arr.begin(), arr.end());
    std::vector<double> norm = arr;
    if (max_val != min_val) {
        for (double& x : norm) x = (x - min_val) / (max_val - min_val);
    }
    return norm;
}

// Train the network to assign higher scores to smaller numbers
void trainRankingModel(NeuralNetwork& nn, int numSamples = 1000, int epochs = 2000) {
    const int SIZE = 25;
    std::cout << "?? Generating " << numSamples << " ranking training samples...\n";

    std::vector<std::vector<double>> X_train, y_train;

    for (int n = 0; n < numSamples; ++n) {
        std::vector<double> arr(SIZE);
        for (int i = 0; i < SIZE; ++i) arr[i] = i + 1;
        std::shuffle(arr.begin(), arr.end(), gen);

        // Target: higher score = earlier position
        std::vector<double> scores(SIZE, 0.0);
        std::vector<std::pair<double, int>> indexed;
        for (int i = 0; i < SIZE; ++i) indexed.emplace_back(arr[i], i);
        std::sort(indexed.begin(), indexed.end()); // sort by value

        for (int i = 0; i < SIZE; ++i) {
            int original_index = indexed[i].second;
            scores[original_index] = (SIZE - i); // higher score for smaller values
        }

        // Normalize both
        X_train.push_back(normalize(arr));
        y_train.push_back(normalize(scores));
    }

    std::cout << "?? Starting ranking model training (" << epochs << " epochs)...\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;
        std::vector<int> indices(numSamples);
        for (int i = 0; i < numSamples; ++i) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), gen);

        for (int idx : indices) {
            const auto& x = X_train[idx];
            const auto& y = y_train[idx];

            nn.forward(x);
            nn.backprop(y);

            for (int i = 0; i < y.size(); ++i) {
                double err = y[i] - nn.output[i];
                total_error += err * err;
            }
        }

        if (epoch % 250 == 0) {
            double avg_error = total_error / (numSamples * SIZE);
            std::cout << "Epoch " << epoch << ", Avg Score Error: " << avg_error << "\n";
        }
    }
}

// Sort an array using learned ranking scores
std::vector<double> rankSort(NeuralNetwork& nn, const std::vector<double>& input) {
    auto norm_input = normalize(input);
    nn.forward(norm_input);
    auto scores = nn.output;

    // Create index-score pairs and sort by descending score
    std::vector<std::pair<double, double>> ranked; // (score, value)
    for (int i = 0; i < input.size(); ++i) {
        ranked.emplace_back(scores[i], input[i]);
    }
    std::sort(ranked.rbegin(), ranked.rend()); // high score first

    std::vector<double> result;
    for (const auto& p : ranked) result.push_back(p.second);
    return result;
}

// Interactive demo loop
int main() {
    const std::string MODEL_FILE = "rank_model_25_ranky.txt";
    const int INPUT_SIZE = 25;
    const int HIDDEN_SIZE = 64;
    const int OUTPUT_SIZE = 25;

    NeuralNetwork nn(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    clearScreen();
    std::cout << "?? Rank-Based Sorting Demo\n";
    std::cout << "===========================\n\n";

    if (std::ifstream(MODEL_FILE)) {
        std::cout << "?? Loading pre-trained ranking model...\n";
        if (nn.loadModel(MODEL_FILE)) {
            std::cout << "? Model loaded!\n\n";
        } else {
            std::cout << "? Failed to load. Training new one...\n";
            trainRankingModel(nn, 1000, 2000);
            nn.saveModel(MODEL_FILE);
            std::cout << "? New model trained and saved.\n\n";
        }
    } else {
        std::cout << "?? Model not found. Training ranking model...\n";
        trainRankingModel(nn, 1000, 2000);
        nn.saveModel(MODEL_FILE);
        std::cout << "? Model trained and saved as '" << MODEL_FILE << "'.\n\n";
    }

    char choice;
    do {
        // Generate random shuffled array
        std::vector<double> test_input(25);
        for (int i = 0; i < 25; ++i) test_input[i] = i + 1;
        std::shuffle(test_input.begin(), test_input.end(), gen);

        auto sorted = rankSort(nn, test_input);

        std::cout << "?? Input:  ";
        for (int i = 0; i < 25; ++i) {
            std::cout << static_cast<int>(test_input[i]) << " ";
            if ((i + 1) % 10 == 0 && i > 0) std::cout << "\n           ";
        }

        std::cout << "\n?? Sorted:";
        for (int i = 0; i < 25; ++i) {
            std::cout << static_cast<int>(sorted[i]) << " ";
            if ((i + 1) % 10 == 0 && i > 0) std::cout << "\n           ";
        }
        std::cout << "\n\n";

        std::cout << "Run another? (y/n): ";
        std::cin >> choice;
        choice = std::tolower(static_cast<unsigned char>(choice));
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        if (choice == 'y') clearScreen();

    } while (choice == 'y');

    std::cout << "Thank you for using the Rank-Based Sorter!\n";
    return 0;
}