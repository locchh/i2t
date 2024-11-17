Using a PyTorch-trained image-captioning model in C++ typically involves exporting the trained model to a format that can be loaded in C++ (e.g., TorchScript) and writing C++ code to handle model inference. Here’s a step-by-step guide:

---

### **Steps to Use the Model in C++**

---

#### 1. **Export the Model to TorchScript**

TorchScript allows you to serialize and run PyTorch models in a language-agnostic way.

In your Python script, save the trained model as a TorchScript file:

```python
import torch

# Assume `model` is your trained image-captioning model
model.eval()

# Dummy input to trace the model (use the expected input shape)
dummy_image = torch.randn(1, 3, 224, 224)  # Example input for the encoder
dummy_caption = torch.tensor([[101]])      # Example input for the decoder (<CLS> token)

# Export the model to TorchScript
scripted_model = torch.jit.trace(model, (dummy_image, dummy_caption))
scripted_model.save("image_captioning_model.pt")
print("Model saved as TorchScript!")
```

This saves your model as `image_captioning_model.pt`.

---

#### 2. **Set Up a C++ Project**

Create a C++ project with the following directory structure:

```plaintext
image-captioning-cpp/
├── CMakeLists.txt      # Build system file for CMake
├── main.cpp            # Main C++ file for inference
├── image_captioning_model.pt  # TorchScript model
└── assets/             # Folder for input images
```

---

#### 3. **Install LibTorch**

Download and install LibTorch, the C++ distribution of PyTorch, from [PyTorch.org](https://pytorch.org/cppdocs/installing.html).

1. Choose the appropriate version for your platform (e.g., CPU/GPU).
2. Extract the archive and set the `Torch_DIR` environment variable to point to the `libtorch` directory.

---

#### 4. **Write the C++ Code**

**`main.cpp`**: A simple C++ script to load the TorchScript model and run inference.

```cpp
#include <torch/script.h> // TorchScript header
#include <torch/torch.h>
#include <opencv2/opencv.hpp> // For image preprocessing
#include <iostream>
#include <memory>

// Function to preprocess the image
torch::Tensor preprocess_image(const std::string& image_path) {
    // Load image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    // Convert to RGB and resize to (224, 224)
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(224, 224));

    // Convert to tensor
    torch::Tensor img_tensor = torch::from_blob(
        image.data, {1, image.rows, image.cols, 3}, torch::kUInt8
    );
    img_tensor = img_tensor.permute({0, 3, 1, 2}); // Change to CxHxW
    img_tensor = img_tensor.to(torch::kFloat32).div(255); // Normalize to [0, 1]

    // Normalize using ImageNet mean and std
    img_tensor[0][0] = (img_tensor[0][0] - 0.485) / 0.229;
    img_tensor[0][1] = (img_tensor[0][1] - 0.456) / 0.224;
    img_tensor[0][2] = (img_tensor[0][2] - 0.406) / 0.225;

    return img_tensor;
}

int main() {
    // Load the TorchScript model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("image_captioning_model.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    std::cout << "Model loaded successfully!\n";

    // Preprocess input image
    std::string image_path = "assets/example.jpg";
    torch::Tensor image_tensor = preprocess_image(image_path);

    // Dummy caption input (e.g., CLS token ID = 101)
    torch::Tensor caption_tensor = torch::tensor({{101}}, torch::kLong);

    // Run inference
    auto output = model.forward({image_tensor, caption_tensor});
    auto generated_tokens = output.toTensor().argmax(2);

    // Print the generated tokens
    std::cout << "Generated caption tokens:\n" << generated_tokens << "\n";

    return 0;
}
```

---

#### 5. **CMake Build Configuration**

**`CMakeLists.txt`**: Configure CMake to compile the project with LibTorch.

```cmake
cmake_minimum_required(VERSION 3.10)
project(ImageCaptioning)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Specify Torch directory
set(Torch_DIR "/path/to/libtorch/share/cmake/Torch")

# Find the Torch package
find_package(Torch REQUIRED)

# Add executable
add_executable(image_captioning main.cpp)

# Link Torch library
target_link_libraries(image_captioning "${TORCH_LIBRARIES}")
set_property(TARGET image_captioning PROPERTY CXX_STANDARD 14)
```

---

#### 6. **Build and Run**

1. **Build the project**:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

2. **Run the executable**:
   ```bash
   ./image_captioning
   ```

---

### Notes on Inference in C++
- The tokenizer (e.g., `BertTokenizer`) will need to be replicated in C++ or replaced with a simple Python script that processes captions offline.
- Ensure preprocessing steps (e.g., normalization) are consistent between Python and C++.
- If your model uses GPU, ensure LibTorch is compiled with CUDA support and linked appropriately.

Let me know if you'd like help with tokenizer replication or other parts of this pipeline!
