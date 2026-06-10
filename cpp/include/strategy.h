#ifndef STRATEGY_H
#define STRATEGY_H

#include <string>
#include <vector>
#include <map>
#include <memory>

// 交易类型
enum class TradeType {
    BUY,
    SELL,
    HOLD
};

// 交易记录结构
struct Trade {
    std::string date;
    std::string symbol;
    TradeType type;
    double price;
    int shares;
    double amount;
};

// 投资组合结构
struct Portfolio {
    double cash;
    std::map<std::string, int> stocks;
    double totalValue;
};

// 策略结果结构
struct StrategyResult {
    std::string date;
    double portfolioValue;
    double cash;
    double stockValue;
    Trade trade;
};

// 策略接口
class Strategy {
public:
    virtual ~Strategy() = default;
    
    // 初始化策略
    virtual void initialize(double initialCash) = 0;
    
    // 执行策略
    virtual StrategyResult execute(const std::string& symbol, double currentPrice, double predictedReturn) = 0;
    
    // 获取投资组合
    virtual Portfolio getPortfolio() const = 0;
    
    // 获取交易记录
    virtual std::vector<Trade> getTrades() const = 0;
    
    // 获取策略名称
    virtual std::string getName() const = 0;
};

// 基于预测的策略
class PredictionBasedStrategy : public Strategy {
public:
    PredictionBasedStrategy(double buyThreshold = 0.01, double sellThreshold = -0.01);
    
    void initialize(double initialCash) override;
    StrategyResult execute(const std::string& symbol, double currentPrice, double predictedReturn) override;
    Portfolio getPortfolio() const override;
    std::vector<Trade> getTrades() const override;
    std::string getName() const override;
    
private:
    double buyThreshold_;
    double sellThreshold_;
    Portfolio portfolio_;
    std::vector<Trade> trades_;
};

// 策略工厂
class StrategyFactory {
public:
    static std::unique_ptr<Strategy> createStrategy(const std::string& strategyName, const std::map<std::string, double>& params = {});
};

#endif // STRATEGY_H