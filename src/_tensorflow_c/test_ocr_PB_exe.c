/*

	// EJERCICIO INCOMPLETO, NO SE LOGRO CONSEGUIR ARCHIVO .PB

    TOOLCHAIN : C:\msys64\ucrt64\bin.
    
    gcc -o test_dll_ocr.exe tf_dll_test.c

*/

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

// Define the function prototype for the exported function from the DLL
typedef const char* (__cdecl* PerformOCRWithTensorFlow_t)();

int main() {
    // Paths to the TensorFlow model and input image
    const char* modelPath = "ocr_model.pb";  // Replace with your model path
    const char* imagePath = "test.png";      // Replace with your image path

    // Load the DLL
    HMODULE dllHandle = LoadLibrary("OCR_Tensorflow.dll");
    if (!dllHandle) {
        fprintf(stderr, "Error: Unable to load OCR_Tensorflow.dll\n");
        return 1;
    }

    // Get the address of the exported function
    PerformOCRWithTensorFlow_t PerformOCRWithTensorFlow =
        (PerformOCRWithTensorFlow_t)GetProcAddress(dllHandle, "performOCR");
    if (!PerformOCRWithTensorFlow) {
        fprintf(stderr, "Error: Unable to find 'PerformOCRWithTensorFlow' in the DLL\n");
        FreeLibrary(dllHandle);
        return 1;
    }

    // Call the function to perform OCR
    const char* ocrResult = PerformOCRWithTensorFlow();
    if (ocrResult) {
        printf("OCR Result: %s\n", ocrResult);

        // Free the dynamically allocated result string (if necessary)
        free((void*)ocrResult); // Use `free` only if the DLL uses `strdup` to allocate memory
    } else {
        fprintf(stderr, "Error: OCR failed.\n");
    }

    // Unload the DLL
    FreeLibrary(dllHandle);

    return 0;
}