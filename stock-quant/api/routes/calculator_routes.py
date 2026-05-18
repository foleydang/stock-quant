"""补仓成本计算器路由"""

from flask import Blueprint, jsonify, request

calculator_bp = Blueprint('calculator', __name__)


@calculator_bp.route('/calculator/cost', methods=['GET', 'POST'])
def calculator_cost():
    """补仓成本计算
    
    参数:
        cost_price: 成本价格(每股)
        shares: 当前持有股数
        current_price: 当前股票价格
        target_cost: 目标成本价(期望降到多少)
        add_price: 补仓价格(可以等于current_price或自定义)
    
    返回:
        补仓方案详细数据
    """
    if request.method == 'POST':
        data = request.get_json() or {}
    else:
        data = request.args.to_dict()

    try:
        cost_price = float(data.get('cost_price', 0))
        shares = int(data.get('shares', 0))
        current_price = float(data.get('current_price', 0))
        target_cost = float(data.get('target_cost', 0))
        add_price = float(data.get('add_price', current_price))
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': '参数格式错误，请输入有效数字'}), 400

    if cost_price <= 0 or shares <= 0 or current_price <= 0 or target_cost <= 0 or add_price <= 0:
        return jsonify({'status': 'error', 'message': '所有参数必须为正数'}), 400

    # 计算需要补仓的数量
    # 新成本价 = (cost_price * shares + add_price * add_shares) / (shares + add_shares) = target_cost
    # 解方程: add_shares = (cost_price * shares - target_cost * shares) / (target_cost - add_price)
    numerator = cost_price * shares - target_cost * shares
    denominator = target_cost - add_price

    if denominator == 0:
        # 如果目标成本等于补仓价格，无法通过补仓达到
        return jsonify({
            'status': 'success',
            'message': '目标成本价等于补仓价格，无法通过补仓达到目标',
            'addShares': None,
            'addAmount': None,
            'currentLoss': round((current_price - cost_price) * shares, 2),
            'currentLossRate': round((current_price - cost_price) / cost_price * 100, 2),
        })

    add_shares = numerator / denominator

    if add_shares < 0:
        # 目标成本低于补仓价格或目标成本高于当前成本，无法通过此补仓价达到
        if target_cost < add_price and target_cost < cost_price:
            message_text = f'目标成本价({target_cost})低于补仓价({add_price})，无法通过补仓降成本到此价格'
        elif target_cost > cost_price:
            message_text = f'目标成本价({target_cost})高于当前成本价({cost_price})，无需补仓'
        else:
            message_text = '无法通过此补仓价达到目标成本'
        return jsonify({
            'status': 'success',
            'message': message_text,
            'addShares': None,
            'addAmount': None,
            'currentLoss': round((current_price - cost_price) * shares, 2),
            'currentLossRate': round((current_price - cost_price) / cost_price * 100, 2),
        })

    # 补仓数量必须是100的整数倍（A股规则）
    add_shares_round = int((add_shares // 100 + 1) * 100)  # 向上取整到100的倍数

    add_amount = round(add_price * add_shares_round, 2)
    new_total_cost = round(cost_price * shares + add_price * add_shares_round, 2)
    new_shares = shares + add_shares_round
    new_cost_price = round(new_total_cost / new_shares, 4)

    # 盈亏计算
    current_loss = round((current_price - cost_price) * shares, 2)
    current_loss_rate = round((current_price - cost_price) / cost_price * 100, 2)
    new_loss = round((current_price - new_cost_price) * new_shares, 2)
    new_loss_rate = round((current_price - new_cost_price) / new_cost_price * 100, 2)

    total_invest = round(cost_price * shares + add_price * add_shares_round, 2)
    total_value = round(current_price * new_shares, 2)

    # 补仓方案对比（不同补仓数量下的效果）
    comparisons = []
    for ratio in [0.5, 1.0, 1.5, 2.0, 3.0]:
        comp_shares = int(add_shares_round * ratio)
        if comp_shares < 100:
            comp_shares = 100
        comp_amount = round(add_price * comp_shares, 2)
        comp_total_cost = round(cost_price * shares + add_price * comp_shares, 2)
        comp_new_shares = shares + comp_shares
        comp_cost_price = round(comp_total_cost / comp_new_shares, 4)
        comp_loss = round((current_price - comp_cost_price) * comp_new_shares, 2)
        comp_loss_rate = round((current_price - comp_cost_price) / comp_cost_price * 100, 2)
        comparisons.append({
            'label': f'{ratio}x基准补仓',
            'addShares': comp_shares,
            'addAmount': comp_amount,
            'newCostPrice': comp_cost_price,
            'newShares': comp_new_shares,
            'newLoss': comp_loss,
            'newLossRate': comp_loss_rate,
            'totalInvest': round(cost_price * shares + add_price * comp_shares, 2),
            'totalValue': round(current_price * comp_new_shares, 2),
        })

    return jsonify({
        'status': 'success',
        'addShares': add_shares_round,
        'addAmount': add_amount,
        'newTotalCost': new_total_cost,
        'newCostPrice': new_cost_price,
        'newShares': new_shares,
        'currentLoss': current_loss,
        'currentLossRate': current_loss_rate,
        'newLoss': new_loss,
        'newLossRate': new_loss_rate,
        'totalInvest': total_invest,
        'totalValue': total_value,
        'targetCost': target_cost,
        'actualNewCost': new_cost_price,
        'comparisons': comparisons,
    })