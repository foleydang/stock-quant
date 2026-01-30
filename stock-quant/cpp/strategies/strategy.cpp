#include "strategy.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

// 获取当前日期
std::string getCurrentDate() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_c);
    
    std::stringstream ss;
    ss << std::put_time(now_tm, "%Y-%m-%d");
    return ss.str();
}

// PredictionBasedStrategy 实现
PredictionBasedStrategy::PredictionBasedStrategy(double buyThreshold, double sellThreshold)
    : buyThreshold_(buyThreshold),
      sellThreshold_(sellThreshold) {
}

void PredictionBasedStrategy::initialize(double initialCash) {
    portfolio_.cash = initialCash;
    portfolio_.totalValue = initialCash;
    portfolio_.stocks.clear();
    trades_.clear();
}

StrategyResult PredictionBasedStrategy::execute(const std::string& symbol, double currentPrice, double predictedReturn) {
    StrategyResult result;
    result.date = getCurrentDate();
    
    // 计算股票价值
    double stockValue = 0.0;
    for (const auto& [sym, shares] : portfolio_.stocks) {
        if (sym == symbol) {
            stockValue += shares * currentPrice;
        }
    }
    
    // 更新投资组合价值
    portfolio_.totalValue = portfolio_.cash + stockValue;
    
    // 初始化交易记录
    Trade trade;
    trade.date = result.date;
    trade.symbol = symbol;
    trade.type = TradeType::HOLD;
    trade.price = currentPrice;
    trade.shares = 0;
    trade.amount = 0.0;
    
    // 执行交易决策
    if (predictedReturn > buyThreshold_ && portfolio_.cash > currentPrice * 10) {
        // 买入
        int sharesToBuy = static_cast<int>((portfolio_.cash * 0.1) / currentPrice);
        double cost = sharesToBuy * currentPrice;
        
        portfolio_.cash -= cost;
        portfolio_.stocks[symbol] += sharesToBuy;
        portfolio_.totalValue = portfolio_.cash + stockValue + cost;
        
        trade.type = TradeType::BUY;
        trade.shares = sharesToBuy;
        trade.amount = cost;
        
        trades_.push_back(trade);
    } else if (predictedReturn < sellThreshold_ && portfolio_.stocks.find(symbol) != portfolio_.stocks.end() && portfolio_.stocks[symbol] > 0) {
        // 卖出
        int sharesToSell = portfolio_.stocks[symbol] / 2;
        double revenue = sharesToSell * currentPrice;
        
        portfolio_.cash += revenue;
        portfolio_.stocks[symbol] -= sharesToSell;
        if (portfolio_.stocks[symbol] == 0) {
            portfolio_.stocks.erase(symbol);
        }
        portfolio_.totalValue = portfolio_.cash + stockValue - revenue;
        
        trade.type = TradeType::SELL;
        trade.shares = sharesToSell;
        trade.amount = revenue;
        
        trades_.push_back(trade);
    }
    
    // 填充结果
    result.portfolioValue = portfolio_.totalValue;
    result.cash = portfolio_.cash;
    result.stockValue = stockValue;
    result.trade = trade;
    
    return result;
}

Portfolio PredictionBasedStrategy::getPortfolio() const {
    return portfolio_;
}

std::vector<Trade> PredictionBasedStrategy::getTrades() const {
    return trades_;
}

std::string PredictionBasedStrategy::getName() const {
    return "PredictionBasedStrategy";
}

// StrategyFactory 实现
std::unique_ptr<Strategy> StrategyFactory::createStrategy(const std::string& strategyName, const std::map<std::string, double>& params) {
    if (strategyName == "prediction_based") {
        double buyThreshold = 0.01;
        double sellThreshold = -0.01;
        
        // 从参数中获取阈值
        auto buyIt = params.find("buy_threshold");
        if (buyIt != params.end()) {
            buyThreshold = buyIt->second;
        }
        
        auto sellIt = params.find("sell_threshold");
        if (sellIt != params.end()) {
            sellThreshold = sellIt->second;
        }
        
        return std::make_unique<PredictionBasedStrategy>(buyThreshold, sellThreshold);
    }
    
    // 默认返回基于预测的策略
    return std::make_unique<PredictionBasedStrategy>();
}