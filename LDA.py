import os
import json
import math
import random
from openai import OpenAI
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

STOCK_STYLE_CONFIG = {
    "blue_chip": {
        "name": "Blue Chip",
        "core_feature": "Stable performance, huge market value, strong dividend capacity, industry leader, high risk resistance",
        "pe_range": (8, 18),
        "price_volatility_base": 0.12,
        "news_bias": ["Stable performance", "Dividend plan", "Industry leader", "Policy favorable", "Reasonable valuation"],
        "decision_weight": "Valuation safety>Performance stability>Market sentiment>Short-term theme",
        "risk_level": "Low Risk"
    },
    "emerging": {
        "name": "Emerging Industry Stock (New Productive Forces)",
        "core_feature": "In line with national strategy, growth industry, high performance growth, mainly ChiNext and STAR Market, intensive policy favorable",
        "pe_range": (35, 60),
        "price_volatility_base": 0.30,
        "news_bias": ["Technological breakthrough", "Policy support", "Performance pre-increase", "Industry prosperity", "Capacity expansion"],
        "decision_weight": "Policy orientation>Industry growth>Technological breakthrough>Valuation level>Short-term sentiment",
        "risk_level": "Medium-High Risk"
    },
    "sunset": {
        "name": "Sunset Industry Stock (Cyclical Industry)",
        "core_feature": "Traditional industry, slow growth, performance greatly affected by macroeconomic cycle, low valuation, average dividend",
        "pe_range": (10, 22),
        "price_volatility_base": 0.20,
        "news_bias": ["Raw material price increase", "Macroeconomy", "Overcapacity", "Performance pressure", "Industry integration"],
        "decision_weight": "Cycle inflection point>Raw material price>Performance forecast>Valuation level>Market sentiment",
        "risk_level": "Medium Risk"
    },
    "theme": {
        "name": "Theme Stock (Concept Sector)",
        "core_feature": "Driven by short-term hot spots, policy favorable/technical trend as core, weak performance realization, strong capital speculation attribute",
        "pe_range": (40, 80),
        "price_volatility_base": 0.45,
        "news_bias": ["Policy favorable", "Technical hot spot", "Capital inflow", "Concept fermentation", "Industry outlet"],
        "decision_weight": "Short-term theme>Capital sentiment>Policy favorable>Industry trend>Valuation level",
        "risk_level": "High Risk"
    },
    "growth": {
        "name": "Growth Stock",
        "core_feature": "Revenue/profit growth rate far exceeds industry average, growth enterprise, high P/E ratio, high performance elasticity",
        "pe_range": (25, 45),
        "price_volatility_base": 0.18,
        "news_bias": ["High performance growth", "Capacity expansion", "Market share increase", "New product launch", "Industry prosperity"],
        "decision_weight": "Performance growth>Industry prosperity>Valuation level>Market sentiment>Short-term theme",
        "risk_level": "Medium Risk"
    },
    "value": {
        "name": "Value Stock",
        "core_feature": "Undervalued stock price, low P/E ratio, low P/B ratio, stable performance, high dividend rate, high margin of safety",
        "pe_range": (6, 15),
        "price_volatility_base": 0.08,
        "news_bias": ["Undervaluation", "High dividend rate", "High quality assets", "Stable performance", "Value regression"],
        "decision_weight": "Valuation safety>Dividend capacity>Asset quality>Performance stability>Market sentiment",
        "risk_level": "Ultra-Low Risk"
    }
}

RETAIL_CAPITAL_LEVEL = {
    "small": {"min": 50000, "max": 300000, "ratio": 0.8, "risk": "Aggressive", "trade_freq": 0.9},
    "mid": {"min": 300000, "max": 1000000, "ratio": 0.15, "risk": "Steady", "trade_freq": 0.6},
    "large": {"min": 1000000, "max": 5000000, "ratio": 0.05, "risk": "Conservative", "trade_freq": 0.3}
}

RETAIL_TRADING_STYLE = {
    0: {"name": "Blind Follow Type", "core_logic": "Only watch stock price rise/fall + forum sentiment, blindly chase rise and kill fall, ignore valuation, the most mainstream retail investors in A-share market"},
    1: {"name": "Value Anchored Type", "core_logic": "Only watch PE valuation + performance news, buy at low valuation and sell at high valuation, resolutely not touch high valuation targets"},
    2: {"name": "Game Arbitrage Type", "core_logic": "Understand market game, reverse operation, harvest by using others' emotions, can control all stock styles"},
    3: {"name": "Policy Theme Type", "core_logic": "Only watch policy favorable + concept hot spots, theme is more important than everything, valuation is not important, buy when rising and sell when falling, unique retail investors in A-share market"}
}

STOCK_PREFER_WEIGHT = {
    0: {"blue_chip": 0.2, "emerging": 0.6, "sunset": 0.3, "theme": 1.0, "growth": 0.5, "value": 0.1},
    1: {"blue_chip": 1.0, "emerging": 0.4, "sunset": 0.7, "theme": 0.1, "growth": 0.6, "value": 1.0},
    2: {"blue_chip": 0.9, "emerging": 0.9, "sunset": 0.9, "theme": 0.9, "growth": 0.9, "value": 0.9},
    3: {"blue_chip": 0.3, "emerging": 1.0, "sunset": 0.2, "theme": 1.0, "growth": 0.8, "value": 0.2}
}

INITIAL_POSITION_CONFIG = {
    "capital_position": {
        "small": {"min": 0, "max": 500},
        "mid": {"min": 300, "max": 1500},
        "large": {"min": 1000, "max": 3000},
        "inst": {"min": 3000, "max": 8000}
    },
    "style_hold_prob": {0: 0.3, 1: 0.8, 2: 0.6, 3: 0.5},
    "stock_type_bonus": {"blue_chip": 1.2, "value": 1.3, "growth": 1.0, "emerging": 0.9, "theme": 0.8, "sunset": 0.7}
}

print("===== Please select the A-share stock type for this simulation =====")
print("1 - Blue Chip (Stable performance/Low volatility/Low valuation)")
print("2 - Emerging Industry Stock (New Productive Forces) (Policy favorable/High growth/High valuation)")
print("3 - Sunset Industry Stock (Cyclical Industry) (Cycle volatility/Medium valuation/Traditional industry)")
print("4 - Theme Stock (Concept Sector) (Hot speculation/Ultra-high volatility/Ultra-high valuation)")
print("5 - Growth Stock (High performance growth/Medium volatility/Medium-high valuation)")
print("6 - Value Stock (Undervalued safety/Ultra-low volatility/Ultra-low valuation)")
choice = input("Please enter number 1-6 to select the stock type: ")
choice_map = {"1": "blue_chip", "2": "emerging", "3": "sunset", "4": "theme", "5": "growth", "6": "value"}
SELECTED_STOCK_STYLE = choice_map.get(choice, "blue_chip")
CURRENT_STOCK_CONFIG = STOCK_STYLE_CONFIG[SELECTED_STOCK_STYLE]
print(f"\nConfirmed for this simulation: {CURRENT_STOCK_CONFIG['name']}")
print(f"Core features of the target: {CURRENT_STOCK_CONFIG['core_feature']}")
print(f"Risk level: {CURRENT_STOCK_CONFIG['risk_level']}\n")

def call_trading_llm(prompt):
    try:
        response = client.chat.completions.create(
            model='deepseek-chat',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"action": "wait", "reason": "Network fluctuation", "price_flex": 0}

class GameAgent:
    def __init__(self, agent_id, agent_type, level_k, learning_rate):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.level_k = level_k
        self.learning_rate = learning_rate
        self.total_profit = 0
        self.last_return = 0
        self.s2_trigger_prob = 0.9 if agent_type == "Inst" else 0.3
        self.sentiment_decay = 0.3

        if agent_type == "Inst":
            self.initial_capital = random.randint(1000000, 5000000)
            self.cap_level = "Institutional Capital"
            self.risk_type = "Institutional Steady Type"
            self.trade_freq = 1.0
        else:
            cap_choice = random.choices(["small", "mid", "large"], weights=[0.8, 0.15, 0.05], k=1)[0]
            self.cap_level = cap_choice
            self.risk_type = RETAIL_CAPITAL_LEVEL[cap_choice]["risk"]
            self.trade_freq = RETAIL_CAPITAL_LEVEL[cap_choice]["trade_freq"]
            self.initial_capital = random.randint(RETAIL_CAPITAL_LEVEL[cap_choice]["min"],
                                                  RETAIL_CAPITAL_LEVEL[cap_choice]["max"])

        self.cash = self.initial_capital
        self.shares = 0
        self.available_shares = 0
        self.today_buy_shares = 0

        base_hold_prob = INITIAL_POSITION_CONFIG["style_hold_prob"][self.level_k]
        stock_bonus = INITIAL_POSITION_CONFIG["stock_type_bonus"][SELECTED_STOCK_STYLE]
        final_hold_prob = base_hold_prob * stock_bonus

        if random.random() < final_hold_prob:
            if agent_type == "Inst":
                pos_range = INITIAL_POSITION_CONFIG["capital_position"]["inst"]
            else:
                pos_range = INITIAL_POSITION_CONFIG["capital_position"][self.cap_level]
            self.shares = random.randint(pos_range["min"], pos_range["max"])
            self.available_shares = self.shares
            self.today_buy_shares = 0
            cost_price = round(self.initial_capital / 100 * random.uniform(0.9, 1.1), 2)
            total_cost = cost_price * self.shares
            self.cash = round(self.initial_capital - total_cost, 2)
            if self.cash < 0:
                self.cash = round(self.initial_capital * 0.3, 2)
                self.shares = round(self.cash / cost_price, 0)

        self.update_trading_style()

    def update_trading_style(self):
        self.trading_style = RETAIL_TRADING_STYLE[self.level_k]["name"] if self.agent_type == "Retail" else "Institutional Rational Type"
        self.style_logic = RETAIL_TRADING_STYLE[self.level_k][
            "core_logic"] if self.agent_type == "Retail" else "Comprehensive dimensional analysis of valuation + fundamentals + game"
        self.stock_prefer = STOCK_PREFER_WEIGHT[self.level_k] if self.agent_type == "Retail" else 1.0

    def decide(self, market_context):
        level_instructions = {
            0: "You only trade randomly or blindly follow the trend, do not consider others' intentions, only watch stock price and sentiment. Blind follow type retail investors like small batch high frequency trading (100-500 shares per order), ignore valuation, chase rise and kill fall.",
            1: "You believe in value investment, think the market is rational, pay attention to PE and fundamentals, low valuation is the kingly way. Value anchored type retail investors like medium batch trading (200-800 shares per order), only increase positions when cash is sufficient.",
            2: "You think the market is full of Level-1 rational people, you will create a 'short trap' or reverse harvest by using their rational retreat. Game arbitrage type investors like to flexibly adjust the number of shares (300-1000 shares per order) and operate in reverse according to market sentiment.",
            3: "You only watch policy favorable + concept hot spots, theme is more important than everything, valuation is not important, buy when rising and sell when falling. Policy theme type retail investors like medium batch trading (200-600 shares per order) and stop loss quickly when hot spots fade."
        }
        trade_willing = self.stock_prefer[SELECTED_STOCK_STYLE] if self.agent_type == "Retail" else 1.0
        if random.random() > trade_willing * self.trade_freq and self.agent_type == "Retail":
            return {"side": "Wait and See", "price": round(market_context['current_price'], 2), "qty": 0,
                    "reason": f"[{self.risk_type}] Low stock selection preference, choose to wait and see"}

        s1_prompt = f"""
        Role: You are a [{self.risk_type}] {self.agent_type} investor with trading style={self.trading_style} and style logic={self.style_logic}.
        Your account status: Current holding shares={self.shares} shares, available shares for sale={self.available_shares} shares, current available cash={self.cash:.2f} CNY, current stock price={market_context['current_price']:.2f} CNY.
        Trading target: [{market_context['stock_style']}], target features: {market_context['stock_feature']}.
        Environment: News[{market_context['news']}], Forum[{market_context['forum_opinion']}], Trend[{market_context['trend']}].
        Trading rules: The minimum trading unit of A-shares is 100 shares (1 lot), the number of shares must be an integer multiple of 100; the number of shares bought cannot exceed the upper limit of cash available, and the number of shares sold cannot exceed the available shares for sale.
        Style preference: {level_instructions[self.level_k]}
        Task: Give the intuitive trading decision (Buy/Sell/Wait and See), specific number of shares (0 when waiting and seeing), decision reason (limited to 5 words).
        Important constraints:
        1. When buying, the maximum purchasable shares = available cash ÷ current stock price, the number of shares must be ≤ the maximum purchasable shares and an integer multiple of 100;
        2. When selling, the maximum salable shares = {self.available_shares} shares, the number of shares must be ≤ the maximum salable shares and an integer multiple of 100;
        3. Absolutely cannot choose to sell when holding 0 shares, and cannot buy when cash is insufficient for 1 lot;
        4. [{self.risk_type}] investors have a {self.trade_freq * 100}% probability to trade, try not to wait and see!
        JSON format: {{"action": "...", "qty": 0, "reason": "..."}}
        """
        s1_res = call_trading_llm(s1_prompt)
        intuition_action = s1_res.get("action", "wait")
        intuition_qty = s1_res.get("qty", 0)
        intuition_reason = s1_res.get("reason", "Intuition")

        final_action = intuition_action
        final_qty = intuition_qty
        final_reason = f"[Intuition-{self.trading_style}] {intuition_reason}"
        price_flex = 0.0

        flex = price_flex if self.agent_type == "Inst" else random.uniform(-0.05,
                                                                           0.05) if self.cap_level == "small" else random.uniform(
            -0.03, 0.03)

        if random.random() < self.s2_trigger_prob:
            max_buy_qty = int(self.cash / market_context['current_price']) // 100 * 100
            max_sell_qty = self.available_shares // 100 * 100

            s2_prompt = f"""
            Role: A [{self.risk_type}] {self.agent_type} investor with Level-{self.level_k} game ability, decision weight: {market_context['stock_decision_weight']}.
            Background logic: {level_instructions[self.level_k]}, your trading style={self.trading_style}.
            Your account status: Current holding shares={self.shares} shares, available shares for sale={self.available_shares} shares, current available cash={self.cash:.2f} CNY, current stock price={market_context['current_price']:.2f} CNY.
            Current market: Price{market_context['current_price']}, PE{market_context['pe']:.2f}(reasonable range{market_context['pe_min']}-{market_context['pe_max']}), Forum[{market_context['forum_opinion']}].
            Objective constraints:
            - Maximum purchasable shares when buying={max_buy_qty} shares (integer multiple of 100, cash upper limit);
            - Maximum salable shares when selling={max_sell_qty} shares (integer multiple of 100, upper limit of available holdings for sale);
            - The minimum trading unit of A-shares is 100 shares, the number of shares must be an integer multiple of 100.
            Your intuitive suggestion: Action={intuition_action}, Quantity={intuition_qty} shares.
            Target attributes: {market_context['stock_style']}({market_context['risk_level']})
            Task: Based on your game logic+valuation+trading style+risk preference+own account status+objective constraints, reflect and revise the decision:
            1. Is the action reasonable? Does it conform to the current market and own situation?
            2. Is the number of shares within the objective constraints? Does it fit your trading style preference?
            Must output after revision: Action (Buy/Sell/Wait and See), specific number of shares (0 when waiting and seeing), revision logic (limited to 10 words).
            Hard rules:
            - Purchased shares ≤{max_buy_qty}, sold shares ≤{max_sell_qty}, the number of shares must be an integer multiple of 100;
            - Cannot sell when holding 0 shares, cannot buy when cash is insufficient for 1 lot;
            - Revision logic must fit your trading style!
            JSON format: {{"action": "Buy/Sell/Wait and See", "qty": 0, "logic": "Revision logic within 10 words", "price_flex": 0.0}}
            """
            s2_res = call_trading_llm(s2_prompt)
            final_action = s2_res.get("action", intuition_action)
            final_qty = s2_res.get("qty", intuition_qty)
            final_reason = f"[Reflection-{self.trading_style}] {s2_res.get('logic', 'Game revision')}"
            price_flex = s2_res.get("price_flex", 0.0)
        else:
            price_flex = random.uniform(-0.01, 0.01)
            flex = price_flex if self.agent_type == "Inst" else random.uniform(-0.05,
                                                                               0.05) if self.cap_level == "small" else random.uniform(
                -0.03, 0.03)

            final_qty = int(final_qty)
            if final_action == "Buy":
                max_buy_qty = int(self.cash / round(market_context['current_price'] * (1 + flex), 2)) // 100 * 100
                final_qty = final_qty // 100 * 100
                final_qty = max(0, min(final_qty, max_buy_qty))
                if final_qty == 0 and intuition_action == "Buy":
                    final_action = "Wait and See"
                    final_reason += "| Insufficient cash"
            elif final_action == "Sell":
                max_sell_qty = self.available_shares // 100 * 100
                final_qty = final_qty // 100 * 100
                final_qty = max(0, min(final_qty, max_sell_qty))
                if final_qty == 0 and intuition_action == "Sell":
                    final_action = "Wait and See"
                    final_reason += "| No available holdings"
            else:
                final_qty = 0

        return {
            "side": final_action,
            "price": round(market_context['current_price'] * (1 + flex), 2),
            "qty": final_qty,
            "reason": final_reason
        }

    def publish_forum_post(self, market_context):
        if self.agent_type != "Retail":
            return ""
        post_prob = 0.18 if self.cap_level == "small" else 0.12 if self.cap_level == "mid" else 0.06
        if random.random() > post_prob:
            return ""

        level_post_prompt = {
            0: "You are a stock market novice, follow the trend to speak, short and emotional views, limited to 10 words, e.g. 'Buy quickly when rising'/'Sell quickly when falling'",
            1: "You understand value investment, mention PE and fundamentals, rational views, limited to 15 words, e.g. 'High PE, buy cautiously'/'Reasonable valuation, holdable'",
            2: "You are a stock market veteran, understand game reverse, in-depth views, limited to 20 words, e.g. 'Good news out is bad news, guard against pullback'",
            3: "You are a theme lover, only watch policy hot spots, radical views, limited to 10 words, e.g. 'Policy favorable, buy boldly'"
        }

        prompt = f"""
        Role: [{self.risk_type}] retail investor, Level-{self.level_k}, {level_post_prompt[self.level_k]}
        Current market: Stock price{market_context['current_price']}, PE{market_context['pe']:.2f}, News[{market_context['news']}], Trend[{market_context['trend']}]
        Trading target: {market_context['stock_style']}, your trading behavior today: {market_context['my_action']}
        Task: Post a forum post with views in line with your style and risk preference.
        JSON format: {{"content": "Your view content"}}
        """
        try:
            res = call_trading_llm(prompt)
            post_content = res.get("content", "")
            return f"{self.trading_style}[{self.agent_id}]: {post_content}" if post_content else ""
        except:
            return ""

class EvolutionMarket:
    def __init__(self):
        self.price = 100.0
        self.last_close = 100.0
        self.stock_style = CURRENT_STOCK_CONFIG["name"]
        self.stock_core_feature = CURRENT_STOCK_CONFIG["core_feature"]
        self.pe_min, self.pe_max = CURRENT_STOCK_CONFIG["pe_range"]
        self.price_volatility_base = CURRENT_STOCK_CONFIG["price_volatility_base"]
        self.stock_decision_weight = CURRENT_STOCK_CONFIG["decision_weight"]
        self.stock_risk = CURRENT_STOCK_CONFIG["risk_level"]

        self.market_sentiment = 0.0
        self.sentiment_decay = 0.2

        self.agents = [GameAgent(f"R_{i}", "Retail", random.choice([0, 1, 2, 3]), random.uniform(0.05, 0.3)) for i in
                       range(90)]
        self.agents += [GameAgent(f"I_{i}", "Inst", 2, 0.05) for i in range(1)]

        self.news_pool = {
            "No News": "Market trading is flat, no major news",
            "Weak Positive": "Industry leader's performance pre-increases, sector sentiment warms up",
            "Weak Negative": "Short-term industry inventory increases, sector under pressure",
            "Strong Positive": "Liquidity favorable, ample market liquidity",
            "Strong Negative1": "Fundamentals deteriorate, corporate profits decline",
            "Strong Negative2": "Regulatory cooling, theme speculation restricted"
        }
        self.style_news_pool = CURRENT_STOCK_CONFIG["news_bias"]
        self.news_prob = {
            "No News": 0.6, "Weak Positive": 0.15, "Weak Negative": 0.15, "Strong Positive": 0.06, "Strong Negative1": 0.02, "Strong Negative2": 0.02
        }
        self.last_news_type = "No News"
        self.forum_posts = []
        self.today_forum_opinion = "No Forum Opinion"

        self.days = []
        self.open_prices = []
        self.close_prices = []
        self.change_rates = []
        self.style_0 = []
        self.style_1 = []
        self.style_2 = []
        self.style_3 = []

    def generate_daily_news(self):
        news_types = list(self.news_prob.keys())
        news_probs = list(self.news_prob.values())
        selected_type = random.choices(news_types, weights=news_probs, k=1)[0]
        strong_news = ["Strong Positive", "Strong Negative1", "Strong Negative2"]
        if self.last_news_type in strong_news and selected_type in strong_news:
            selected_type = "No News"
        self.last_news_type = selected_type
        base_news = self.news_pool[selected_type]
        if random.random() < 0.2 and selected_type not in strong_news:
            base_news = random.choice(self.style_news_pool)
        self.market_sentiment *= (1 - self.sentiment_decay)
        if "Strong Positive" in selected_type:
            self.market_sentiment += 4.5
        elif "Strong Negative" in selected_type:
            self.market_sentiment -= 3.6
        elif "Weak Positive" in selected_type:
            self.market_sentiment += 1.5
        elif "Weak Negative" in selected_type:
            self.market_sentiment -= 1.2
        self.market_sentiment = max(min(self.market_sentiment, 5.0), -5.0)
        return base_news

    def summary_forum_opinion(self):
        if not self.forum_posts:
            self.today_forum_opinion = "No forum opinion, retail investors mainly wait and see"
            return
        prompt = f"""
        Role: Forum sentiment analyst
        Today's retail investor forum post list: {self.forum_posts}
        Task: Summarize the overall sentiment of the forum today, judge bullish/bearish/neutral, concise views, limited to 20 words.
        JSON format: {{"opinion": "Your summary content"}}
        """
        res = call_trading_llm(prompt)
        self.today_forum_opinion = res.get("opinion", "Forum views are messy, no unified sentiment")
        self.forum_posts = []

    def run_step(self, news, day):
        print(
            f"\n=== Day {day} | Target Type: {self.stock_style} | News: {news} | Forum Sentiment: {self.today_forum_opinion} ===")
        initial_price = self.price
        price_limit_up = round(initial_price * 1.1, 2)
        price_limit_down = round(initial_price * 0.9, 2)
        results = []

        pe_high_threshold = self.pe_max
        pe_low_threshold = self.pe_min
        if "Strong Positive" in news:
            pe_high_threshold = self.pe_max + 5
        elif "Strong Negative" in news:
            pe_low_threshold = self.pe_min - 3

        random.shuffle(self.agents)
        for agent in self.agents:
            context = {
                "current_price": round(self.price, 2),
                "last_close": self.last_close,
                "pe": self.price / 2.5,
                "news": news,
                "forum_opinion": self.today_forum_opinion,
                "trend": "Rise" if self.price > self.last_close else "Fall",
                "stock_style": self.stock_style,
                "stock_feature": self.stock_core_feature,
                "pe_min": self.pe_min,
                "pe_max": self.pe_max,
                "stock_decision_weight": self.stock_decision_weight,
                "risk_level": self.stock_risk
            }
            dec = agent.decide(context)

            side = dec['side']
            agent.last_return = 0

            news_intensity = 0.5
            if "Strong Positive" in news:
                news_intensity = 3.0
            elif "Strong Negative" in news:
                news_intensity = 2.8
            elif "Weak Positive" in news:
                news_intensity = 1.8
            elif "Weak Negative" in news:
                news_intensity = 1.6

            capital_factor = 1.0
            if agent.agent_type == "Inst":
                capital_factor = 2.0
            elif agent.cap_level == "large":
                capital_factor = 1.5

            sentiment_factor = 1.0
            if "bullish" in self.today_forum_opinion and "obvious" in self.today_forum_opinion:
                sentiment_factor = 2.0
            elif "bearish" in self.today_forum_opinion and "panic" in self.today_forum_opinion:
                sentiment_factor = 2.2

            market_factor = 1.0 + (self.market_sentiment * 0.35)

            impact_base = (math.sqrt(dec['qty']) / 100) * self.price_volatility_base
            impact = impact_base * news_intensity * capital_factor * sentiment_factor * market_factor

            max_single_impact = self.price * 0.05
            impact = min(max(impact, -max_single_impact), max_single_impact)

            current_pe = self.price / 2.5

            if agent.agent_type == "Inst":
                if current_pe > pe_high_threshold and side == "Buy" and random.random() < 0.8:
                    side = "Wait and See"
                elif current_pe < pe_low_threshold and side == "Sell" and random.random() < 0.8:
                    side = "Wait and See"
            else:
                val_sensi = 0.05 if agent.level_k in [0, 3] else 0.3 if agent.level_k == 1 else 0.2
                if current_pe > pe_high_threshold and side == "Buy" and random.random() < val_sensi:
                    side = "Wait and See"
                elif current_pe < pe_low_threshold and side == "Sell" and random.random() < val_sensi:
                    side = "Wait and See"

            required_fund = dec['price'] * dec['qty']

            if side == "Buy" and agent.cash < required_fund:
                side = "Wait and See"
                print(
                    f"[Insufficient Cash] {agent.agent_id} Cannot buy -> Wait and See | Available Cash: {agent.cash:.0f} | Required Fund: {required_fund:.0f}")
            elif side == "Sell" and agent.shares < dec['qty']:
                side = "Wait and See"
                print(f"[Insufficient Holdings] {agent.agent_id} Cannot sell -> Wait and See | Holding Shares: {agent.shares} | Required Shares: {dec['qty']}")

            if side == "Sell":
                if dec['qty'] > agent.available_shares:
                    dec['qty'] = agent.available_shares
                    print(
                        f"[T+1 Rule Effective] {agent.agent_id} Shares bought today cannot be sold -> Revised selling shares: {dec['qty']} (Available: {agent.available_shares}, Total Holdings: {agent.shares})")
                    if dec['qty'] == 0:
                        side = "Wait and See"
                        print(f"[T+1 Rule Effective] {agent.agent_id} No available holdings for sale -> Forced to wait and see")

            if side == "Buy":
                self.price += impact
                if self.price >= price_limit_up:
                    self.price = price_limit_up
                cost = dec['price'] * dec['qty']
                agent.cash -= cost
                agent.shares += dec['qty']
                agent.today_buy_shares += dec['qty']
                agent.last_return = self.price - initial_price

            elif side == "Sell":
                self.price -= impact
                if self.price <= price_limit_down:
                    self.price = price_limit_down
                revenue = dec['price'] * dec['qty']
                agent.cash += revenue
                agent.shares -= dec['qty']
                agent.available_shares -= dec['qty']
                agent.last_return = initial_price - self.price
            else:
                agent.last_return = 0

            agent.total_profit += agent.last_return
            agent_market_context = context.copy()
            agent_market_context["my_action"] = side
            post_content = agent.publish_forum_post(agent_market_context)
            if post_content:
                self.forum_posts.append(post_content)

            results.append(
                f"{agent.agent_id}(L{agent.level_k}-{agent.trading_style}): {side} - {dec['reason']} | Quotation: {dec['price']} | Quantity: {dec['qty']} | Cash: {agent.cash:.0f} | Holdings: {agent.shares}")

        for res in results:
            print(res)

        level_perf = {l: [] for l in [0, 1, 2, 3]}
        for a in self.agents:
            if a.last_return != 0:
                level_perf[a.level_k].append(a.last_return)
        avg_perf = {l: (sum(p) / len(p) if p else -99) for l, p in level_perf.items()}
        best_level = max(avg_perf, key=avg_perf.get)
        print(
            f"Optimal strategy for this round: Level-{best_level}({RETAIL_TRADING_STYLE[best_level]['name']}) (Average Return: {avg_perf[best_level]:.4f})")

        for a in self.agents:
            if a.level_k != best_level and random.random() < a.learning_rate:
                old_l = a.level_k
                a.level_k = best_level
                a.update_trading_style()
                print(
                    f"Evolution generated: {a.agent_id} From {RETAIL_TRADING_STYLE[old_l]['name']} -> {RETAIL_TRADING_STYLE[best_level]['name']}")

        self.summary_forum_opinion()
        self.price = round(self.price, 2)
        self.last_close = self.price
        daily_change = round(((self.price - initial_price) / initial_price) * 100, 2)

        self.days.append(day)
        self.open_prices.append(initial_price)
        self.close_prices.append(self.price)
        self.change_rates.append(daily_change)
        style_count = Counter([a.level_k for a in self.agents])
        self.style_0.append(style_count.get(0, 0))
        self.style_1.append(style_count.get(1, 0))
        self.style_2.append(style_count.get(2, 0))
        self.style_3.append(style_count.get(3, 0))

        for agent in self.agents:
            agent.available_shares += agent.today_buy_shares
            agent.today_buy_shares = 0

        print(
            f"Today's Opening Price: {initial_price:.2f} | Closing Price: {self.price:.2f} | Today's Change Rate: {daily_change}% (Upper Limit: {price_limit_up}, Lower Limit: {price_limit_down})")
        print(f"Strategy Distribution: {style_count}")

    def plot_all_charts(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'A-share {self.stock_style} Market Simulation Results ({len(self.days)} Trading Days)', fontsize=16,
                     fontweight='bold')

        color1 = '#E63946'
        ax1.plot(self.days, self.close_prices, color=color1, linewidth=2.5, marker='o', markersize=4, label='Closing Price')
        ax1.plot(self.days, self.open_prices, color='#457B9D', linewidth=1.5, linestyle='--', marker='s', markersize=3,
                 label='Opening Price')
        ax1.set_xlabel('Trading Day', fontsize=12)
        ax1.set_ylabel('Stock Price (CNY)', fontsize=12, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.legend(loc='upper left')
        ax1.set_title('Stock Price Trend & Daily Change Rate', fontsize=14, fontweight='bold')

        ax1_twin = ax1.twinx()
        colors = ['#E63946' if x > 0 else '#2A9D8F' for x in self.change_rates]
        ax1_twin.bar(self.days, self.change_rates, alpha=0.6, color=colors, width=0.6, label='Change Rate (%)')
        ax1_twin.set_ylabel('Change Rate (%)', fontsize=12, color='#264653')
        ax1_twin.tick_params(axis='y', labelcolor='#264653')
        ax1_twin.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1_twin.legend(loc='upper right')

        ax2.bar(self.days, self.style_0, label='Blind Follow Type(L0)', color='#E76F51', alpha=0.9)
        ax2.bar(self.days, self.style_1, bottom=self.style_0, label='Value Anchored Type(L1)', color='#F4A261', alpha=0.9)
        ax2.bar(self.days, self.style_2, bottom=np.array(self.style_0) + np.array(self.style_1), label='Game Arbitrage Type(L2)',
                color='#2A9D8F', alpha=0.9)
        ax2.bar(self.days, self.style_3,
                bottom=np.array(self.style_0) + np.array(self.style_1) + np.array(self.style_2), label='Policy Theme Type(L3)',
                color='#264653', alpha=0.9)
        ax2.set_xlabel('Trading Day', fontsize=12)
        ax2.set_ylabel('Number of Investors', fontsize=12)
        ax2.set_title('Trading Strategy Distribution Evolution', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    lab = EvolutionMarket()
    trading_days = 1
    for d in range(1, trading_days + 1):
        daily_news = lab.generate_daily_news()
        lab.run_step(daily_news, d)

    print("\nSimulation completed, generating visual charts...")
    lab.plot_all_charts()