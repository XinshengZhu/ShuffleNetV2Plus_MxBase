#include <iostream>
#include <experimental/filesystem>
#include <vector>
#include "Shufflenetv2plusClassify.h"
#include "MxBase/Log/Log.h"

namespace fs = std::experimental::filesystem;
namespace {
const uint32_t CLASS_NUM = 1000;
}

int main()
{
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../model/shufflenetv2plus_clsidx_to_labels.names";
    initParam.topk = 5;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = "../model/shufflenetv2plus_npu.om";

    auto shufflenetv2plus = std::make_shared<Shufflenetv2plusClassify>();
    APP_ERROR ret = shufflenetv2plus->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Shufflenetv2plusClassify init failed, ret=" << ret << ".";
        return ret;
    }
    std::string binDir = "../test.bin";
    LogInfo << "read image path " << binDir;
    ret = shufflenetv2plus->Process(binDir);
    if (ret != APP_ERR_OK) {
        LogError << "Shufflenetv2plusClassify process failed, ret=" << ret << ".";
        shufflenetv2plus->DeInit();
        return ret;
    }
    shufflenetv2plus->DeInit();
    return APP_ERR_OK;
}