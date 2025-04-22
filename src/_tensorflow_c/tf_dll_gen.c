/*

TOOLCHAIN : C:\msys64\ucrt64\bin.

gcc -I"include" -L"lib" -shared -o TensorFlowApp64_C.dll tf_dll_gen.c -ltensorflow


*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif


EXPORT const char* GetTensorFlowVersionInfo() {
    const char* version = TF_Version();
    const char* prefix  = "TensorFlow Version: ";

    // Allocate enough memory for the combined string
    size_t buffer_size = strlen(prefix) + strlen(version) + 1;
    char* msg = (char*)malloc(buffer_size);

    if (msg == NULL) {
        return "Memory allocation failed!";
    }

    // Build the combined string
    strcpy(msg, prefix);
    strcat(msg, version);

    return msg;  // Remember: caller must free() this!
}

EXPORT const char* GetTensorFlowVersion() {
  const char* msg = TF_Version();
  return msg; // Return the TensorFlow version directly
}