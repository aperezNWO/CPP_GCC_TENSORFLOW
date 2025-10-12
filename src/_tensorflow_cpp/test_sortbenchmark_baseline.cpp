/*

g++ -std=c++17 -o test_sortbenchmark_baseline.exe test_sortbenchmark_baseline.cpp

*/

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);
std::uniform_int_distribution<> moveDis(0, 8);

// Sigmoid function with clamping
double sigmoid(double x) {
    x = std::max(-700.0, std::min(700.0, x));
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid
double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

class NeuralNetwork {
public:
    std::vector<double> input, hidden, output;
    std::vector<std::vector<double>> weights_ih, weights_ho;
    std::vector<double> bias_h, bias_o;
    double learningRate = 0.01; // Lower LR for stability

    NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        : input(inputSize), hidden(hiddenSize), output(outputSize),
          weights_ih(hiddenSize, std::vector<double>(inputSize)),
          weights_ho(outputSize, std::vector<double>(hiddenSize)),
          bias_h(hiddenSize), bias_o(outputSize) {

        // Initialize weights and biases randomly in [-1, 1]
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
        std::vector<double> outputError(output.size());
        std::vector<double> hiddenError(hidden.size());

        // Output layer gradients
        std::vector<double> grad_output(output.size());
        for (int i = 0; i < output.size(); ++i) {
            double error = target[i] - output[i];
            grad_output[i] = error * sigmoidDerivative(output[i]);
        }

        // Hidden layer gradients
        std::vector<double> grad_hidden(hidden.size(), 0);
        for (int i = 0; i < hidden.size(); ++i) {
            for (int j = 0; j < output.size(); ++j) {
                grad_hidden[i] += grad_output[j] * weights_ho[j][i];
            }
            grad_hidden[i] *= sigmoidDerivative(hidden[i]);
        }

        // Update weights and biases (output layer)
        for (int i = 0; i < output.size(); ++i) {
            bias_o[i] += learningRate * grad_output[i];
            for (int j = 0; j < hidden.size(); ++j) {
                weights_ho[i][j] += learningRate * grad_output[i] * hidden[j];
            }
        }

        // Update weights and biases (hidden layer)
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
                for (double w : row) {
                    file << std::setprecision(10) << w << " ";
                }
                file << "\n";
            }
        };

        auto writeVector = [&](const std::vector<double>& vec) {
            file << vec.size() << "\n";
            for (double v : vec) {
                file << std::setprecision(10) << v << " ";
            }
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

// Normalize input: scale [min..max] ? [0..1]
std::vector<double> normalize(const std::vector<double>& arr) {
    double min_val = *std::min_element(arr.begin(), arr.end());
    double max_val = *std::max_element(arr.begin(), arr.end());
    std::vector<double> norm = arr;
    if (max_val != min_val) {
        for (double& x : norm) {
            x = (x - min_val) / (max_val - min_val);
        }
    }
    return norm;
}

// Denormalize output: scale back to original range
std::vector<double> denormalize(const std::vector<double>& norm, double orig_min, double orig_max) {
    std::vector<double> orig = norm;
    if (orig_max != orig_min) {
        for (double& x : orig) {
            x = x * (orig_max - orig_min) + orig_min;
        }
    }
    return orig;
}


int main() {
    try {
        const int INPUT_SIZE  = 9;
        const int HIDDEN_SIZE = 64;
        const int OUTPUT_SIZE = 9;

        NeuralNetwork nn(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        std::cout << "?? Sorting Network Initialized\n";

        // Generate training data
        std::cout << "?? Generating 1000 shuffled arrays...\n";
        std::vector<std::vector<double>> X_train, y_train;

        for (int n = 0; n < 1000; ++n) {
            std::vector<double> arr = {1,2,3,4,5,6,7,8,9};
            std::shuffle(arr.begin(), arr.end(), gen);

            std::vector<double> sorted = arr;
            std::sort(sorted.begin(), sorted.end());

            // Normalize both input and target
            auto x_norm = normalize(arr);
            auto y_norm = normalize(sorted);

            X_train.push_back(x_norm);
            y_train.push_back(y_norm);
        }

        // Training loop
        std::cout << "?? Starting training (5000 epochs)...\n";
        for (int epoch = 0; epoch < 5000; ++epoch) {
            double total_error = 0.0;

            // Shuffle training order
            std::vector<int> indices(X_train.size());
            for (int i = 0; i < indices.size(); ++i) indices[i] = i;
            std::shuffle(indices.begin(), indices.end(), gen);

            for (int idx : indices) {
                const auto& x = X_train[idx];
                const auto& y = y_train[idx];

                nn.forward(x);
                nn.backprop(y);

                // Compute MSE
                for (int i = 0; i < y.size(); ++i) {
                    double err = y[i] - nn.output[i];
                    total_error += err * err;
                }
            }

            double avg_error = total_error / X_train.size();

            if (epoch % 500 == 0) {
                std::cout << "Epoch " << epoch << ", Avg Error: " << avg_error << "\n";
            }
        }

        // Test on new input
        std::vector<double> test_input = {9, 4, 1, 7, 2, 8, 5, 3, 6};
        auto test_norm = normalize(test_input);

        nn.forward(test_norm);
        auto pred_norm = nn.output;
        auto pred_raw = denormalize(pred_norm, 1.0, 9.0); // map back to [1..9]

        std::cout << "\n?? Input:  ";
        for (double x : test_input) std::cout << static_cast<int>(x) << " ";

        std::cout << "\n?? Sorted: ";
        for (double x : pred_raw) std::cout << static_cast<int>(std::round(x)) << " ";
        std::cout << "\n";

        // Save trained model
        if (nn.saveModel("sort_model_9.txt")) {
            std::cout << "\n?? Model saved to sort_model.txt\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "?? Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

