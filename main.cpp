#include <numeric>
#include <dxgi1_2.h>
#include <d3d11.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>


#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
OrtValue* input_tensors = nullptr;      //推理的输入
OrtValue* output_tensors = nullptr;     //推理的输出
std::vector<std::string> input_names;
std::vector<std::string> output_names;
std::vector<int64_t> output_dims;
size_t num_output_dims = 0;
OrtAllocator* allocator;
OrtRunOptions* run_options;   //添加一个新的全局变量
std::vector<cv::Rect> boxes;
std::vector<double> confidences;
std::vector<int> classes;

// 预处理函数
void Preprocess(cv::Mat& img, float* inputTensorValues)
{
    // 转换到RGB通道
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 裁剪/缩放图像到模型需要的输入大小
    int IMG_WIDTH = 416;
    int IMG_HEIGHT = 416;
    cv::resize(img, img, cv::Size(IMG_WIDTH, IMG_HEIGHT));

    // 将RGB图像转换为浮点数类型
    img.convertTo(img, CV_32F);

    // 归一化，范围为 [0, 1]
    img = img / 255.0;

    // 将WHC图像转为NCHW（batch，channel，height，weight）形式
    //这个过程在原先提供的Yolox_Pre_Processing函数中有实现，这是合适的。

    int i = 0;
    for (int row = 0; row < img.rows; ++row)
    {
        uchar* uc_pixel = img.data + row * img.step;
        for (int col = 0; col < img.cols; ++col)
        {
            inputTensorValues[i] = uc_pixel[2] / 255.0f;
            inputTensorValues[i + IMG_HEIGHT * IMG_WIDTH] = uc_pixel[1] / 255.0f;
            inputTensorValues[i + 2 * IMG_HEIGHT * IMG_WIDTH] = uc_pixel[0] / 255.0f;
            uc_pixel += 3;
            ++i;
        }

    }
}

void Postprocess(float* output_tensors, const std::vector<int64_t>& output_dims) {

    // 大小为 batchSize x (85 * h * w) 或 batchSize x (5 * h * w)
    int stride = output_dims[2] / output_dims[1];

    for (int i = 0; i < output_dims[1]; i++) {

        const int basic_pos = i * (5 + output_dims[2]);

        float x_center = output_tensors[basic_pos + 0] * stride;
        float y_center = output_tensors[basic_pos + 1] * stride;
        float w = exp(output_tensors[basic_pos + 2]) * stride;
        float h = exp(output_tensors[basic_pos + 3]) * stride;

        float conf = output_tensors[basic_pos + 4]; // confidence score
        std::vector<float> class_probabilities(output_dims[2] - 5);  // Initializing with size C  

        // 把分类概率存储为另外的向量
        for (int class_idx = 0; class_idx < output_dims[2] - 5; ++class_idx) {
            class_probabilities[class_idx] = output_tensors[basic_pos + 5 + class_idx];
        }

        // Get the class index with the maximum probability
        int class_idx = std::distance(class_probabilities.begin(), std::max_element(class_probabilities.begin(), class_probabilities.end()));
        float box_prob = conf * class_probabilities[class_idx];  // Final probability by multiplying confidence and class probability

        if (box_prob >= 0.5)
        {
            cv::Rect rect(x_center - w / 2, y_center - h / 2, w, h);
            boxes.push_back(rect);
            confidences.push_back(box_prob);
            classes.push_back(class_idx);
        }

        std::vector<int> indices;
        std::vector<cv::Rect2d> boxes_f(boxes.begin(), boxes.end());
        std::vector<float> confidences_f(confidences.begin(), confidences.end());

        cv::dnn::NMSBoxes(boxes_f, confidences_f, 0.7, 0.1, indices);

        for (size_t idx : indices) {
            cv::Rect box = boxes[idx];
            std::cout << "Object " << idx
                << ", class id: " << classes[idx]
                << ", confidence: " << confidences[idx]
                << ", bbox: [" << box.x << "," << box.y << "," << box.width << "," << box.height << "]" << std::endl;
        }
    }
}


Ort::Session SetupOnnxSession()
{
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "Test");
    Ort::SessionOptions session_options;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
    Ort::Session session(env, L"CF-yolox.onnx", session_options);
    std::cout << "模型加载成功" << std::endl;

    // ----------- 获取输入/输出层信息 ----------- //
    OrtStatus* status = nullptr;
    OrtAllocator* allocator;
    g_ort->GetAllocatorWithDefaultOptions(&allocator); // 在使用allocator之前，需要进行初始化

    // ----------- 获取输入的数量 ----------- //
    size_t num_input_nodes = 0;
    status = g_ort->SessionGetInputCount(session, &num_input_nodes);

    char* input_name_cstr = nullptr;
    g_ort->SessionGetInputName(session, num_input_nodes - 1, allocator, &input_name_cstr);
    input_names.resize(1);
    input_names[0] = std::string(input_name_cstr);
    g_ort->AllocatorFree(allocator, input_name_cstr);

    // ----------- 获取输入名称 ----------- //
    char* input_names_temp = nullptr;
    status = g_ort->SessionGetInputName(session, num_input_nodes - 1, allocator, &input_names_temp);
    input_names[0] = std::string(input_names_temp);
    g_ort->AllocatorFree(allocator, input_names_temp);
    std::cout << "输入节点名称: " << input_names[0] << std::endl;
    // ----------- 获取输入维度 ----------- //
    size_t num_intput_dims;                     //维度
    OrtTypeInfo* input_typeinfo = nullptr;     //输入类型信息
    const OrtTensorTypeAndShapeInfo* Input_tensor_info = nullptr;   //输入的tensor信息
    g_ort->SessionGetInputTypeInfo(session, num_input_nodes - 1, &input_typeinfo);
    status = g_ort->SessionGetInputTypeInfo(session, num_input_nodes - 1, &input_typeinfo); //获取第0个输入信息，
    g_ort->CastTypeInfoToTensorInfo(input_typeinfo, &Input_tensor_info);       //获取输入tensor信息
    g_ort->GetDimensionsCount(Input_tensor_info, &num_intput_dims);            //获取输入维度大小 4
    std::cout << "输入维度大小：" << num_intput_dims << std::endl;
    //每个输入维度的信息
    std::vector<int64_t> input_dims;   //输入维度容器数组
    input_dims.resize(num_intput_dims); //容器的大小 4
    g_ort->GetDimensions(Input_tensor_info, input_dims.data(), num_intput_dims);    //获取每个维度的大小 [1 3 416 416]
    for (size_t i = 0; i < num_intput_dims; i++)
        std::cout << "输入维度 " << i << " = " << input_dims[i] << std::endl;     //打印每个维度

    // 获取输入tensor大小
    size_t input_tensor_size = 1;
    for (const auto& dim : input_dims) {
        input_tensor_size *= dim;
    }

    // 创建输入张量
    OrtMemoryInfo* memory_info;
    g_ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info);
    // Replace appropriate values here from input_dims
    int64_t local_input_dims[4] = { 1, 3, 416, 416 };

    // 创建输入张量
    g_ort->CreateTensorAsOrtValue(allocator, local_input_dims, num_intput_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors);

    // 创建运行选项
    g_ort->CreateRunOptions(&run_options);

    g_ort->ReleaseTypeInfo(input_typeinfo);

    g_ort->GetAllocatorWithDefaultOptions(&allocator); // 在使用allocator之前，需要进行初始化
    // 计算输入内存大小
    int intput_size = std::accumulate(input_dims.begin(), input_dims.end(), 1, std::multiplies<int64_t>()) * sizeof(float);

    // ----------- 获取输出的数量 ----------- //
    size_t num_output_nodes = 0;
    status = g_ort->SessionGetOutputCount(session, &num_output_nodes);

    char* output_name_cstr = nullptr;
    g_ort->SessionGetOutputName(session, num_output_nodes - 1, allocator, &output_name_cstr);
    output_names.resize(1);
    output_names[0] = std::string(output_name_cstr);
    g_ort->AllocatorFree(allocator, output_name_cstr);
    std::cout << "输出节点名称: " << output_names[0] << std::endl;
    // ----------- 输出维度 ----------- //
    size_t num_output_dims = 0;    //维度
    OrtTypeInfo* output_typeinfo = nullptr;     //输出类型信息
    const OrtTensorTypeAndShapeInfo* output_tensor_info = nullptr;   //输入tensor信息
    status = g_ort->SessionGetOutputTypeInfo(session, num_output_nodes - 1, &output_typeinfo);  //获取第x-1个输出的信息，yolox只有一个输出，索引为0
    //yolov5有4个输出,只有第4个输出有用，索引为4-1 = 3
    g_ort->CastTypeInfoToTensorInfo(output_typeinfo, &output_tensor_info);         //获取输出tensor信息
    g_ort->GetDimensionsCount(output_tensor_info, &num_output_dims);               //获取输出维度大小 4 
    std::cout << "输出维度大小：" << num_output_dims << std::endl;

    // 获取每个输出维度信息
    std::vector<int64_t> output_dims(num_output_dims);       //输出容器, 立即初始化大小，一次性分配
    g_ort->GetDimensions(output_tensor_info, output_dims.data(), num_output_dims);  //获取每个输出维度大小 [1 3549 6] 类别：6 - 5 = 1 

    for (size_t i = 0; i < num_output_dims; i++)
        std::cout << "输出维度 " << i << " = " << output_dims[i] << std::endl;//打印每个维度

    // 在确定output_dims被正确初始化后，再释放 TypeInfo
    g_ort->ReleaseTypeInfo(output_typeinfo);

    // 计算输出内存大小
    int output_size = std::accumulate(output_dims.begin(), output_dims.end(), 1, std::multiplies<int64_t>()) * sizeof(float);

    // 创建输出张量
    int64_t local_output_dims[4] = { 1, 3549, 6 };
    g_ort->CreateTensorAsOrtValue(allocator, local_output_dims, num_output_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &output_tensors);

    return session;  
}


void dxgi_opencv(Ort::Session& session)
{
    //1.创建设备和上下文
    ID3D11Device* device;
    ID3D11DeviceContext* context;
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &featureLevel, 1, D3D11_SDK_VERSION, &device, nullptr, &context)))
        return;

    //2.获取 DXGI设备
    IDXGIDevice* dxgiDevice;
    device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);

    //3.获取适配器
    IDXGIAdapter* dxgiAdapter;
    dxgiDevice->GetParent(__uuidof(IDXGIAdapter), (void**)&dxgiAdapter);
    dxgiDevice->Release();

    //4.获取输出
    IDXGIOutput* dxgiOutput;
    dxgiAdapter->EnumOutputs(0, &dxgiOutput);
    dxgiAdapter->Release();

    //5.获取输出的桌面图像
    IDXGIOutput1* dxgiOutput1;
    dxgiOutput->QueryInterface(__uuidof(IDXGIOutput1), (void**)&dxgiOutput1);
    dxgiOutput->Release();

    IDXGIOutputDuplication* desktopDupl;
    dxgiOutput1->DuplicateOutput(device, &desktopDupl);
    dxgiOutput1->Release();

    DXGI_OUTDUPL_FRAME_INFO frameInfo;
    IDXGIResource* desktopResource = nullptr;
    ID3D11Texture2D* acquiredTex = nullptr;
    ID3D11Texture2D* newTexture = nullptr;

    // iterate
    for (;;) {
        HRESULT hr = desktopDupl->AcquireNextFrame(100, &frameInfo, &desktopResource);
        if (SUCCEEDED(hr)) {

            // Start timing
            auto start = std::chrono::high_resolution_clock::now();

            desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&acquiredTex);
            desktopResource->Release();

            D3D11_TEXTURE2D_DESC desc;
            acquiredTex->GetDesc(&desc);

            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            desc.Usage = D3D11_USAGE_STAGING;
            desc.BindFlags = 0;
            desc.MiscFlags = 0;

            HRESULT hr = device->CreateTexture2D(&desc, nullptr, &newTexture);
            if (FAILED(hr)) {
                acquiredTex->Release();
                desktopDupl->ReleaseFrame();
                continue;
            }

            context->CopyResource(newTexture, acquiredTex);

            D3D11_MAPPED_SUBRESOURCE resource;
            UINT subresource = D3D11CalcSubresource(0, 0, 0);
            context->Map(newTexture, subresource, D3D11_MAP_READ, 0, &resource);

            auto pointer = static_cast<BYTE*>(resource.pData);

            int imgWidth = desc.Width;
            int imgHeight = desc.Height;
            cv::Mat img(imgHeight, imgWidth, CV_8UC4, pointer);

            acquiredTex->Release();
            newTexture->Release();
            desktopDupl->ReleaseFrame();

            // End timing and calculate duration
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;

            // Print the duration
            std::cout << "截图: " << elapsed.count() << " ms" << std::endl;

            cv::Mat imgBGR;
            cv::cvtColor(img, imgBGR, cv::COLOR_RGB2BGR);
            // 在图像中心裁剪一个500x500的区域
            int centerX = imgBGR.cols / 2;
            int centerY = imgBGR.rows / 2;
            cv::Rect roi(centerX - 250, centerY - 250, 500, 500);  // 修改裁剪尺寸为500x500
            imgBGR = imgBGR(roi);
            // 预处理函数
            cv::Mat inputBlob = imgBGR;

            float* inputTensorValues = new float[imgBGR.rows * imgBGR.cols * imgBGR.channels()];
            Preprocess(imgBGR, inputTensorValues);
            // 替换下面的代码中用于输入和输出张量的维度。
            const std::array<int64_t, 4> input_shape{ 1, 3, 416, 416 }; // 模型的输入维度，根据你的模型修改
            const std::array<int64_t, 3> output_shape{ 1, 3549, 6 }; // 模型的输出维度，根据你的模型修改

            // 将数据拷贝到input_tensors
            void* input_tensor_raw_data;
            size_t input_tensor_raw_data_length = sizeof(float) * input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
            g_ort->GetTensorMutableData(input_tensors, &input_tensor_raw_data);
            memcpy(input_tensor_raw_data, inputTensorValues, input_tensor_raw_data_length);

            // 执行推理
            std::array<Ort::Value, 1> input_container = { Ort::Value(input_tensors) };
            std::array<Ort::Value, 1> output_container = { Ort::Value(output_tensors) };
            std::vector<const char*> input_node_names_cstr;
            std::vector<const char*> output_node_names_cstr;

            for (const auto& name : input_names)
                input_node_names_cstr.push_back(name.c_str());

            for (const auto& name : output_names)
                output_node_names_cstr.push_back(name.c_str());

            session.Run(Ort::RunOptions{ nullptr },
                input_node_names_cstr.data(),
                input_container.data(),
                input_container.size(), // 注意这里使用容器的实际大小
                output_node_names_cstr.data(),
                output_container.data(),
                output_container.size()); // 注意这里使用容器的实际大小

            // 获取模型输出
            float* output_data;
            g_ort->GetTensorMutableData(output_tensors, reinterpret_cast<void**>(&output_data));



            // 后处理和可视化
            std::vector<float> inference_output(output_data, output_data + std::accumulate(std::begin(output_shape), std::end(output_shape), 1, std::multiplies<>{}));
            Postprocess(inference_output.data(), output_dims);

            // Draw bounding boxes and labels onto the image
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                cv::rectangle(imgBGR, boxes[i], cv::Scalar(0, 0, 255), 2);
                std::string label = std::to_string(classes[i]) + ": " + std::to_string(confidences[i]);
                cv::putText(imgBGR, label, boxes[i].tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
            }

            cv::namedWindow("Captured Image with bbox", cv::WINDOW_AUTOSIZE);
            cv::imshow("Captured Image with bbox", imgBGR);
            cv::waitKey(1);

            delete[] inputTensorValues;
        }
        else if (hr == DXGI_ERROR_WAIT_TIMEOUT)
        {
            std::cout << "获取下一帧超时..正在重试" << std::endl;
            continue; //当桌面没有更新会返回这个错误码，跳过这个帧继续下一次捕获。
        }
        else
        {
            std::cout << "无法获取下一帧..正在重试" << std::endl;
            break;
        }
    }
    desktopDupl->Release();
    context->Release();

}




int main() {
        Ort::Session session = SetupOnnxSession();
        dxgi_opencv(session);

    return 0;
}
