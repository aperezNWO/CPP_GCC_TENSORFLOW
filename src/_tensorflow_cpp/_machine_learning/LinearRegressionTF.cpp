
/*

// COMPILE FROM ROOT (above _tensorflow folder)

g++ -std=c++20 -I"include" -L"lib" -shared -m64 -o "__test/linearregressionTF.exe" "_machine_learning/LinearRegressionTF.cpp"  -ltensorflow -Wl,--subsystem,windows   

*/

// Filename: LinearRegressionTF.cpp
// Requires TensorFlow C++ library to be linked (installation/compilation is complex)
// Example compilation command (adjust paths as needed):
// g++ -std=c++14 -I/path/to/tensorflow/include -L/path/to/tensorflow/lib -ltensorflow_cc -ltensorflow_framework LinearRegressionTF.cpp -o LinearRegressionTF

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <iostream>
#include <vector>

int main() {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    // 1. Prepare Data
    std::vector<float> mission_numbers = {8.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
    std::vector<float> mission_times = {147.0f, 193.0f, 195.0f, 244.0f, 142.0f, 217.0f, 295.0f, 265.0f, 301.0f};

    // 2. Define Constants and Hyperparameters
    const int num_samples = mission_numbers.size();
    const float learning_rate = 0.01f; // Adjust as needed
    const int num_iterations = 1000; // Adjust as needed

    // 3. Create Graph
    Scope root = Scope::NewRootScope();

    // Placeholders for input data (x) and target values (y)
    auto X = Placeholder(root, DT_FLOAT, Placeholder::Shape({num_samples, 1}));
    auto Y = Placeholder(root, DT_FLOAT, Placeholder::Shape({num_samples, 1}));

    // Variables for model parameters (slope 'm' and intercept 'b')
    auto m = Variable(root, {1, 1}, DT_FLOAT);
    auto b = Variable(root, {1, 1}, DT_FLOAT);

    // Initialize variables (or use Assign operations later)
    auto m_init = Assign(root, m, Const(root, {{0.0f}})); // Initial guess for slope
    auto b_init = Assign(root, b, Const(root, {{0.0f}})); // Initial guess for intercept

    // Define the linear model: Y_pred = m * X + b
    auto Y_pred = Add(root, MatMul(root, X, m), b);

    // Define the cost function (Mean Squared Error)
    auto diff = Sub(root, Y_pred, Y);
    auto squared_diff = Square(root, diff);
    auto cost = ReduceMean(root, squared_diff, {0});

    // Calculate gradients manually using TensorFlow ops (simplified)
    // dJ/dm = (2/N) * sum((Y_pred - Y) * X)
    // dJ/db = (2/N) * sum(Y_pred - Y)
    auto N_tensor = Const(root, static_cast<float>(num_samples));
    auto grad_m = Mul(root, Div(root, Const(root, 2.0f), N_tensor), ReduceSum(root, Mul(root, diff, X), {0}));
    auto grad_b = Mul(root, Div(root, Const(root, 2.0f), N_tensor), ReduceSum(root, diff, {0}));

    // Define update operations for gradient descent
    auto update_m = Assign(root, m, Sub(root, m, Mul(root, Const(root, learning_rate), grad_m)));
    auto update_b = Assign(root, b, Sub(root, b, Mul(root, Const(root, learning_rate), grad_b)));

    // 4. Create Session and Run Training
    ClientSession session(root);

    // Initialize variables first
    std::vector<Tensor> init_outputs;
    TF_CHECK_OK(session.Run({m_init, b_init}, &init_outputs));

    // Prepare input tensors
    Tensor x_tensor(DT_FLOAT, TensorShape({num_samples, 1}));
    Tensor y_tensor(DT_FLOAT, TensorShape({num_samples, 1}));
    std::copy_n(mission_numbers.begin(), num_samples, x_tensor.flat<float>().data());
    std::copy_n(mission_times.begin(), num_samples, y_tensor.flat<float>().data());

    // Training Loop
    for (int i = 0; i < num_iterations; ++i) {
        std::vector<Tensor> outputs;
        TF_CHECK_OK(session.Run({{X, x_tensor}, {Y, y_tensor}}, {update_m, update_b, cost}, &outputs));
        if (i % 100 == 0) { // Print cost every 100 iterations
            std::cout << "Iteration " << i << ", Cost: " << outputs[2].scalar<float>()() << std::endl;
        }
    }

    // Get final trained parameters (slope and intercept)
    std::vector<Tensor> final_outputs;
    TF_CHECK_OK(session.Run({m, b}, &final_outputs));
    float final_slope = final_outputs[0].matrix<float>()(0, 0);
    float final_intercept = final_outputs[1].matrix<float>()(0, 0);

    std::cout << "\nTrained Linear Regression Model (TensorFlow C++): Total_Time_Hours = " << final_slope << " * Mission_Number + " << final_intercept << std::endl;

    // Predict for Apollo 18 (mission number 18)
    float mission_number_to_predict = 18.0f;
    Tensor predict_x_tensor(DT_FLOAT, TensorShape({1, 1}));
    predict_x_tensor.flat<float>()(0) = mission_number_to_predict;

    std::vector<Tensor> prediction_outputs;
    TF_CHECK_OK(session.Run({{X, predict_x_tensor}}, {Y_pred}, &prediction_outputs));
    float predicted_time = prediction_outputs[0].matrix<float>()(0, 0);

    std::cout << "\nPredicted total mission time for Apollo " << static_cast<int>(mission_number_to_predict) << ": " << predicted_time << " hours." << std::endl;
    std::cout << "Approximate duration: " << predicted_time / 24.0f << " days." << std::endl; // Note: This is total mission time, not just travel time one way

    return 0;
}