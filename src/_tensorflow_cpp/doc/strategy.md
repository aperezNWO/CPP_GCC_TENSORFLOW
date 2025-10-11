==================================
?? PROJECT STRATEGY
==================================

?? VISION
A full-stack demo that teaches the difference between deterministic algorithms (QuickSort) 
and learning-based approaches (Neural Networks), visualized through interactive animations.

?? SCOPE
- C++ DLL: Sorting methods + NN inference
- .NET Core: API endpoints via P/Invoke
- Angular: Step-by-step animation UI
- Educational focus: Show *how* different paradigms solve problems

----------------------------------
?? IDEAS (Future Possibilities)
----------------------------------
[ ] Add TensorFlow model loading from .pb
[ ] Compare performance: time, accuracy, "smoothness"
[ ] Export trained weights to C++ array for zero-dep NN
[ ] Support NxN matrix sorting (2D)
[ ] Let user draw input bar chart ? sort it
[ ] Add sound effects per swap (fun!)

----------------------------------
? QUESTIONS (Need Answers)
----------------------------------
[ ] How to pass temperature to NeuralSort via DllImport?
[ ] Can I pre-load the TF model when .NET starts?
[ ] Best way to avoid memory leaks when returning JSON strings?
[ ] Should I rename 'OpenCvDll' to 'AlgorithmCore'?

----------------------------------
? TODO (Next Actions)
----------------------------------
[ ] Rename /cpp_old ? /native
[ ] Move sorting functions to /src/native/sorting/
[ ] Update Makefile paths
[ ] Test BubbleSort endpoint after move
[ ] Document renaming decisions in DECISIONS.md

----------------------------------
?? DECISIONS (Why We Did It This Way)
----------------------------------
- Renamed OpenCvDll ? AlgorithmCore because it no longer depends on OpenCV.
- Use JSON strings over struct P/Invoke because nested arrays are hard to marshal.
- Simulate NN steps instead of real training so frontend has animation data.