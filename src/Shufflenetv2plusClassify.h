#ifndef MXBASE_SHUFFLENETV2PLUSCLASSIFY_H
#define MXBASE_SHUFFLENETV2PLUSCLASSIFY_H

#include <string>
#include <vector>
#include <memory>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/postprocess/include/ClassPostProcessors/Resnet50PostProcess.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

#define CHANNEL 3
#define NET_HEIGHT 224
#define NET_WIDTH 224

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    uint32_t classNum;
    uint32_t topk;
    bool softmax;
    bool checkTensor;
    std::string modelPath;
};

class Shufflenetv2plusClassify {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR ReadTensorFromFile(const std::string &file, float *data);
    APP_ERROR ReadInputTensor(const std::string &fileName,
        std::vector<MxBase::TensorBase> *inputs);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs,
        std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
        std::vector<std::vector<MxBase::ClassInfo>> &clsInfos);
    APP_ERROR ShowResult(const std::string &imgPath,
        std::vector<std::vector<MxBase::ClassInfo>> &BatchClsInfos);
    APP_ERROR Process(const std::string &imgPath);
    APP_ERROR DeInit();
 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::Resnet50PostProcess> post_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    std::vector<uint32_t> inputDataShape_ = {1, 3, 224, 224};
    uint32_t inputDataSize_;
};
#endif