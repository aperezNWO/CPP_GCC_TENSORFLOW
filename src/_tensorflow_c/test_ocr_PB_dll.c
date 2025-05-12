/*
    TOOLCHAIN : C:\msys64\mingwin\bin.
    
    gcc -I"include" -L"lib" -shared -o OCR_Tensorflow.dll OCR_Tensorflow.c -ltensorflow

 */

 #include <windows.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <tensorflow/c/c_api.h> // TensorFlow C API header

 
    #ifdef _WIN32
    #define EXPORT __declspec(dllexport)
    #else
    #define EXPORT
    #endif

 
 // Custom deallocator for TF_Buffer
 void custom_deallocator(void* data, size_t length) {
     free(data); // Free the allocated memory
 }
 
 // Custom deallocator for TF_Tensor
 void tensor_deallocator(void* data, size_t length, void* arg) {
     free(data); // Free the allocated memory
 }

EXPORT const char* performOCR() {

    const char* modelPath = "ocr_model.pb";  // Replace with your model path
    const char* imagePath = "test.png";      // Replace with your image path

     TF_Status* status = TF_NewStatus();
     TF_Graph* graph = TF_NewGraph();
 
     // Load the TensorFlow model
     FILE* file = fopen(modelPath, "rb");
     if (!file) {
         //fprintf(stderr, "Error: Unable to open TensorFlow model file.\n");
         //return NULL;
         return "Error : unable to open TensorFlow model file";
     }
 
     fseek(file, 0, SEEK_END);
     long fileSize = ftell(file);
     fseek(file, 0, SEEK_SET);
 
     void* modelData = malloc(fileSize);
     fread(modelData, 1, fileSize, file);
     fclose(file);
 
     TF_Buffer* buffer = TF_NewBuffer();
     buffer->data = modelData;
     buffer->length = fileSize;
 
     // Assign the custom deallocator
     buffer->data_deallocator = custom_deallocator;
 
     TF_ImportGraphDefOptions* importOpts = TF_NewImportGraphDefOptions();
     TF_GraphImportGraphDef(graph, buffer, importOpts, status);
     TF_DeleteImportGraphDefOptions(importOpts);
     TF_DeleteBuffer(buffer);
 
     if (TF_GetCode(status) != TF_OK) {
         //fprintf(stderr, "Error: Failed to load TensorFlow model: %s\n", TF_Message(status));
         TF_DeleteGraph(graph);
         TF_DeleteStatus(status);
         //return NULL;
         return "Error : Failed to load Tensorflow model";
     }
 
     // Create a TensorFlow session
     TF_SessionOptions* sessionOpts = TF_NewSessionOptions();
     TF_Session* session = TF_NewSession(graph, sessionOpts, status);
     TF_DeleteSessionOptions(sessionOpts);
 
     if (TF_GetCode(status) != TF_OK) {
         //fprintf(stderr, "Error: Failed to create TensorFlow session: %s\n", TF_Message(status));
         TF_DeleteGraph(graph);
         TF_DeleteStatus(status);
         //return NULL;
         return "Error : failed to create TensorFlow Session";
     }
 
     // Prepare input tensor
     int inputHeight = 64; // Example input size (adjust based on your model)
     int inputWidth = 128;
     int channels = 1;
 
     // Simulate loading normalized image data (replace with real image processing)
     float* imageData = (float*)malloc(sizeof(float) * inputHeight * inputWidth * channels);
     for (size_t i = 0; i < inputHeight * inputWidth * channels; ++i) {
         imageData[i] = (float)rand() / RAND_MAX; // Random values for demonstration
     }
 
     int64_t inputDims[] = {1, inputHeight, inputWidth, channels}; // Batch size = 1
     TF_Tensor* inputTensor = TF_NewTensor(
         TF_FLOAT,
         inputDims,
         4, // Number of dimensions
         imageData,
         sizeof(float) * inputHeight * inputWidth * channels,
         tensor_deallocator, // Deallocator function
         NULL                // Deallocator argument
     );
 
     // Run inference
     const char* inputOpName = "input"; // Replace with your model's input operation name
     const char* outputOpName = "output"; // Replace with your model's output operation name
 
     TF_Output inputOp = {TF_GraphOperationByName(graph, inputOpName), 0};
     TF_Output outputOp = {TF_GraphOperationByName(graph, outputOpName), 0};
 
     TF_Tensor* outputTensor = NULL;
     TF_SessionRun(
         session,
         NULL, // Run options
         &inputOp, &inputTensor, 1, // Input operation and tensor
         &outputOp, &outputTensor, 1, // Output operation and tensor
         NULL, 0, // Target operations
         NULL, // Run metadata
         status
     );
 
     if (TF_GetCode(status) != TF_OK) {
         //fprintf(stderr, "Error: Inference failed: %s\n", TF_Message(status));
         TF_DeleteTensor(inputTensor);
         TF_DeleteSession(session, status);
         TF_DeleteGraph(graph);
         TF_DeleteStatus(status);
         //return NULL;
         return "Error : Interference failed";
     }
 
     // Process the output
     float* outputData = (float*)TF_TensorData(outputTensor);
     int numOutputs = TF_Dim(outputTensor, 1); // Number of output classes
     char result[256] = ""; // Placeholder for decoded text
 
     for (int i = 0; i < numOutputs; ++i) {
         int predictedClass = (int)(outputData[i] + 0.5); // Round to nearest integer
         char c = (char)predictedClass; // Map class index to character (depends on your model)
         strncat(result, &c, 1);
     }
 
     // Cleanup
     TF_DeleteTensor(inputTensor);
     TF_DeleteTensor(outputTensor);
     TF_DeleteSession(session, status);
     TF_DeleteGraph(graph);
     TF_DeleteStatus(status);
 
     return strdup(result); // Return dynamically allocated string
 }