#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <cstdlib>

// 模型接口类
class ModelInterface {
public:
    ModelInterface(const std::string& pythonScriptPath) 
        : pythonScriptPath_(pythonScriptPath) {
    }
    
    // 训练模型
    bool trainModel(const std::string& symbol, const std::string& modelType = "rf") {
        std::string command = "python3 " + pythonScriptPath_ + " train " + symbol + " " + modelType;
        std::cout << "Executing command: " << command << std::endl;
        
        int result = std::system(command.c_str());
        return result == 0;
    }
    
    // 预测价格
    std::map<std::string, double> predict(const std::string& symbol, const std::string& modelType = "rf") {
        std::string outputFile = "/tmp/prediction_output.json";
        std::string command = "python3 " + pythonScriptPath_ + " predict " + symbol + " " + modelType + " > " + outputFile;
        std::cout << "Executing command: " << command << std::endl;
        
        int result = std::system(command.c_str());
        if (result != 0) {
            std::cerr << "Failed to execute prediction command" << std::endl;
            return {};
        }
        
        // 读取输出文件
        std::ifstream file(outputFile);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file" << std::endl;
            return {};
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string output = buffer.str();
        file.close();
        
        // 解析输出（简化版，实际应该使用JSON库）
        std::map<std::string, double> prediction;
        
        // 查找预测值
        size_t predictedPricePos = output.find("predicted_price");
        if (predictedPricePos != std::string::npos) {
            size_t colonPos = output.find(":", predictedPricePos);
            size_t endPos = output.find(",", colonPos);
            if (endPos == std::string::npos) {
                endPos = output.find("}", colonPos);
            }
            std::string priceStr = output.substr(colonPos + 1, endPos - colonPos - 1);
            try {
                prediction["predicted_price"] = std::stod(priceStr);
            } catch (...) {
                std::cerr << "Failed to parse predicted price" << std::endl;
            }
        }
        
        size_t predictedReturnPos = output.find("predicted_return");
        if (predictedReturnPos != std::string::npos) {
            size_t colonPos = output.find(":", predictedReturnPos);
            size_t endPos = output.find(",", colonPos);
            if (endPos == std::string::npos) {
                endPos = output.find("}", colonPos);
            }
            std::string returnStr = output.substr(colonPos + 1, endPos - colonPos - 1);
            try {
                prediction["predicted_return"] = std::stod(returnStr);
            } catch (...) {
                std::cerr << "Failed to parse predicted return" << std::endl;
            }
        }
        
        return prediction;
    }
    
    // 批量预测
    std::map<std::string, std::map<std::string, double>> batchPredict(const std::vector<std::string>& symbols, const std::string& modelType = "rf") {
        std::map<std::string, std::map<std::string, double>> predictions;
        
        for (const auto& symbol : symbols) {
            auto prediction = predict(symbol, modelType);
            if (!prediction.empty()) {
                predictions[symbol] = prediction;
            }
        }
        
        return predictions;
    }
    
private:
    std::string pythonScriptPath_;
};

// 主函数（用于测试）
int main() {
    ModelInterface modelInterface("../../python/models/model_runner.py");
    
    // 测试训练模型
    std::cout << "Training model for 000001.SZ..." << std::endl;
    bool trainResult = modelInterface.trainModel("000001.SZ");
    std::cout << "Train result: " << (trainResult ? "success" : "failure") << std::endl;
    
    // 测试预测
    std::cout << "Predicting for 000001.SZ..." << std::endl;
    auto prediction = modelInterface.predict("000001.SZ");
    std::cout << "Prediction result:" << std::endl;
    for (const auto& [key, value] : prediction) {
        std::cout << key << ": " << value << std::endl;
    }
    
    // 测试批量预测
    std::vector<std::string> symbols = {"000001.SZ", "000002.SZ"};
    std::cout << "Batch predicting..." << std::endl;
    auto batchPredictions = modelInterface.batchPredict(symbols);
    std::cout << "Batch prediction results:" << std::endl;
    for (const auto& [symbol, pred] : batchPredictions) {
        std::cout << "Symbol: " << symbol << std::endl;
        for (const auto& [key, value] : pred) {
            std::cout << "  " << key << ": " << value << std::endl;
        }
    }
    
    return 0;
}