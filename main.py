"""
AgenticTraders - Main Server & Backtesting Engine
FastAPI server with backtesting logic and yfinance market data.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import func
from database import Database, Agent
from agents import AgentBrain, generate_manager_feedback

# Initialize
app = FastAPI(title="AgenticTraders", version="1.0")

# Enable CORS for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for local development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

db = Database('agentic_traders.db')

# Stock universe - 32 popular stocks across major sectors that existed throughout 2010-2025
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA",  # Mega-cap tech
    "META", "NFLX", "AMD", "INTC", "ORCL", "CSCO",    # Tech & semiconductors  
    "CRM", "ADBE", "QCOM",                             # Software & telecom
    "V", "MA", "JPM", "BAC",                           # Financial services
    "JNJ", "PFE", "UNH", "MRK",                        # Healthcare & pharma
    "WMT", "PG", "KO", "NKE",                          # Consumer goods & retail
    "XOM", "CVX",                                      # Energy
    "BA", "CAT",                                       # Industrials
    "DIS", "^NDX"                                      # Entertainment & benchmark index
]

# Agent configurations
AGENT_CONFIGS = [
    {
        "name": "Momentum Mike",
        "personality": "aggressive",
        "allowed_factors": ["momentum", "volatility"],
        "preferred_indicators": ["RSI", "MACD", "Volume"]
    },
    {
        "name": "Value Vicky",
        "personality": "conservative",
        "allowed_factors": ["mean_reversion", "sentiment"],
        "preferred_indicators": ["Bollinger", "Moving_Averages"]
    },
    {
        "name": "Contrarian Carl",
        "personality": "contrarian",
        "allowed_factors": ["mean_reversion", "volatility"],
        "preferred_indicators": ["RSI", "Bollinger"]
    },
    {
        "name": "Trend Tina",
        "personality": "trend_follower",
        "allowed_factors": ["momentum", "sentiment"],
        "preferred_indicators": ["MACD", "Moving_Averages", "Volume"]
    },
    {
        "name": "Steady Steve",
        "personality": "balanced",
        "allowed_factors": ["momentum", "mean_reversion", "volatility"],
        "preferred_indicators": ["RSI", "MACD", "Bollinger"]
    },
    {
        "name": "Volatile Vera",
        "personality": "risk_seeking",
        "allowed_factors": ["volatility", "momentum"],
        "preferred_indicators": ["Bollinger", "Volume"]
    },
    {
        "name": "Patient Paula",
        "personality": "patient",
        "allowed_factors": ["mean_reversion", "sentiment"],
        "preferred_indicators": ["Moving_Averages", "RSI"]
    },
    {
        "name": "Momentum Mary",
        "personality": "aggressive",
        "allowed_factors": ["momentum", "volatility"],
        "preferred_indicators": ["MACD", "Volume"]
    },
    {
        "name": "Defensive Dan",
        "personality": "defensive",
        "allowed_factors": ["mean_reversion", "sentiment"],
        "preferred_indicators": ["Bollinger", "Moving_Averages"]
    },
    {
        "name": "Adaptive Alex",
        "personality": "adaptive",
        "allowed_factors": ["momentum", "mean_reversion", "volatility", "sentiment"],
        "preferred_indicators": ["RSI", "MACD", "Bollinger", "Moving_Averages"]
    }
]


class BacktestRequest(BaseModel):
    start_year: int = 2015
    end_year: int = 2025
    learning_years: int = 5  # First N years are learning phase
    
    # Experiment configuration flags for ablation studies
    enable_memory: bool = True       # RAG memory retrieval
    enable_reflection: bool = True   # Strategy notes updates
    enable_manager: bool = True      # Manager feedback
    description: str = ""            # Human-readable experiment name


class MarketDataEngine:
    """Fetch and process market data using yfinance"""
    
    def __init__(self):
        self.cache = {}
    
    def get_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download market data for all stocks + SPY"""
        tickers = UNIVERSE + ["SPY"]
        
        cache_key = f"{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        print(f"Downloading data from {start_date} to {end_date}...")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        self.cache[cache_key] = data
        return data
    
    def calculate_monthly_returns(self, data: pd.DataFrame, month_end: str) -> Dict:
        """Calculate returns for each stock in the month"""
        returns = {}
        
        for ticker in UNIVERSE:
            try:
                # Try Close first (more widely available), fall back to Adj Close
                if ('Close', ticker) in data.columns:
                    ticker_data = data[('Close', ticker)]
                elif ('Adj Close', ticker) in data.columns:
                    ticker_data = data[('Adj Close', ticker)]
                else:
                    returns[ticker] = 0.0
                    continue
                
                # Get month data and drop NaN
                month_data = ticker_data[:month_end].dropna()
                
                if len(month_data) < 20:  # Need at least 20 days
                    returns[ticker] = 0.0
                    continue
                
                # Calculate monthly return
                start_price = month_data.iloc[-20]  # ~1 month ago
                end_price = month_data.iloc[-1]
                
                returns[ticker] = ((end_price / start_price) - 1) * 100
            except Exception as e:
                returns[ticker] = 0.0
        
        return returns
    
    def get_market_context(self, data: pd.DataFrame, month_end: str) -> Dict:
        """Analyze market regime, volatility, SPY returns"""
        try:
            # Access SPY data using MultiIndex tuple
            if ('Close', 'SPY') in data.columns:
                spy_data = data[('Close', 'SPY')][:month_end].dropna()
            elif ('Adj Close', 'SPY') in data.columns:
                spy_data = data[('Adj Close', 'SPY')][:month_end].dropna()
            else:
                return self._default_context()
            
            if len(spy_data) < 60:
                return self._default_context()
            
            # Calculate returns
            current_price = spy_data.iloc[-1]
            price_1m = spy_data.iloc[-20]
            price_3m = spy_data.iloc[-60]
            
            spy_1m = ((current_price / price_1m) - 1) * 100
            spy_3m = ((current_price / price_3m) - 1) * 100
            
            # Volatility (std of daily returns)
            daily_returns = spy_data.pct_change().dropna()
            volatility = daily_returns.tail(20).std() * np.sqrt(252) * 100
            
            # Regime detection
            regime = self._detect_regime(spy_1m, spy_3m, volatility)
            
            return {
                'regime': regime,
                'volatility': volatility,
                'spy_1m': spy_1m,
                'spy_3m': spy_3m
            }
        except Exception as e:
            print(f"Error calculating market context: {e}")
            return self._default_context()
    
    def _detect_regime(self, return_1m: float, return_3m: float, volatility: float) -> str:
        """Simple regime classification"""
        if return_3m > 5 and volatility < 20:
            return "bull_low_vol"
        elif return_3m > 5 and volatility >= 20:
            return "bull_high_vol"
        elif return_3m < -5 and volatility < 20:
            return "bear_low_vol"
        elif return_3m < -5 and volatility >= 20:
            return "bear_high_vol"
        else:
            return "choppy"
    
    def _default_context(self) -> Dict:
        return {
            'regime': 'unknown',
            'volatility': 20.0,
            'spy_1m': 0.0,
            'spy_3m': 0.0
        }


class BacktestEngine:
    """Run multi-agent backtest simulation"""
    
    def __init__(self, db: Database):
        self.db = db
        self.market_engine = MarketDataEngine()
        self.run_id = None  # Will be set at start of each run
        # Experiment configuration (set by run())
        self.enable_memory = True
        self.enable_reflection = True
        self.enable_manager = True
    
    def run(self, start_year: int, end_year: int, learning_years: int,
            enable_memory: bool = True, enable_reflection: bool = True, 
            enable_manager: bool = True, description: str = ""):
        """Execute full backtest with configurable experiment settings"""
        
        # Generate unique run_id for this backtest
        self.run_id = self.db.get_next_run_id()
        
        # Store experiment configuration
        self.enable_memory = enable_memory
        self.enable_reflection = enable_reflection
        self.enable_manager = enable_manager
        
        # Save run config to database
        self.db.save_run_config(self.run_id, {
            'enable_memory': enable_memory,
            'enable_reflection': enable_reflection,
            'enable_manager': enable_manager,
            'start_year': start_year,
            'end_year': end_year,
            'learning_years': learning_years,
            'description': description
        })
        
        # Describe experiment configuration
        config_str = []
        config_str.append(f"🧠 Memory: {'ON' if enable_memory else 'OFF'}")
        config_str.append(f"📝 Reflection: {'ON' if enable_reflection else 'OFF'}")
        config_str.append(f"👨‍💼 Manager: {'ON' if enable_manager else 'OFF'}")
        
        print(f"\n🚀 Starting AgenticTraders Backtest")
        print(f"Run ID: {self.run_id}")
        if description:
            print(f"Experiment: {description}")
        print(f"Period: {start_year}-{end_year}")
        print(f"Learning Phase: First {learning_years} years")
        print(f"Config: {' | '.join(config_str)}\n")
        
        # Initialize agents
        self.db.init_agents(AGENT_CONFIGS)
        agents_db = self.db.get_all_agents()
        agent_brains = [AgentBrain(a, self.db) for a in agents_db]
        
        # Download all data
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        data = self.market_engine.get_market_data(start_date, end_date)
        
        # Generate monthly dates
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Track performance
        agent_portfolios = {agent.id: 100.0 for agent in agent_brains}  # Start with $100
        
        learning_end_year = start_year + learning_years
        
        for idx, month_start in enumerate(date_range):
            month_num = idx + 1
            month_str = month_start.strftime("%Y-%m")
            year = month_start.year
            is_learning = year < learning_end_year
            
            phase = "📚 LEARNING" if is_learning else "💰 DEPLOYMENT"
            print(f"{phase} | Month {month_num} ({month_str})")
            
            # Get market context
            month_end = (month_start + timedelta(days=31)).strftime("%Y-%m-%d")
            market_context = self.market_engine.get_market_context(data, month_end)
            
            # Pre-calculate returns once for all agents
            returns = self.market_engine.calculate_monthly_returns(data, month_end)
            
            # Parallel agent decisions
            monthly_results = []
            enable_memory = self.enable_memory  # Capture for closure
            
            def process_agent(agent):
                """Process single agent decision (runs in parallel)"""
                decision = agent.make_decision(market_context, month_num, is_learning, enable_memory=enable_memory)
                result = self._calculate_portfolio_return(decision, returns)
                allocation = self._get_allocation(decision['confidence'])
                
                return {
                    'agent': agent,
                    'decision': decision,
                    'result': result,
                    'allocation': allocation
                }
            
            # Execute all agent decisions in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_agent, agent) for agent in agent_brains]
                
                # Collect results as they complete
                for future in as_completed(futures):
                    agent_result = future.result()
                    agent = agent_result['agent']
                    decision = agent_result['decision']
                    result = agent_result['result']
                    allocation = agent_result['allocation']
                    
                    # Update portfolio
                    portfolio_return = result * (allocation / 100)
                    agent_portfolios[agent.id] *= (1 + portfolio_return / 100)
                    
                    # Save trade
                    self.db.save_trade({
                        'run_id': self.run_id,
                        'agent_id': agent.id,
                        'month': month_str,
                        'year': year,
                        'month_num': month_num,
                        'factor_choice': decision['factor_choice'],
                        'action': decision['action'],
                        'confidence': decision['confidence'],
                        'stocks_long': decision.get('long_picks', []),
                        'stocks_short': decision.get('short_picks', []),
                        'market_regime': market_context['regime'],
                        'volatility': market_context['volatility'],
                        'spy_return_1m': market_context['spy_1m'],
                        'spy_return_3m': market_context['spy_3m'],
                        'return_pct': result,
                        'allocation': allocation
                    })
                    
                    # Create memory
                    agent.create_memory(market_context, decision, result, month_str, self.run_id)
                    
                    # Track for manager feedback
                    monthly_results.append({
                        'agent_id': agent.id,
                        'name': agent.name,
                        'return': result,
                        'cumulative_return': ((agent_portfolios[agent.id] / 100) - 1) * 100
                    })
                    
                    print(f"  {agent.name}: {decision['action'].upper()} {decision.get('long_picks', []) + decision.get('short_picks', [])} → {result:+.2f}% (Portfolio: ${agent_portfolios[agent.id]:.2f})")
            
            # Manager feedback (only in deployment phase, if enabled)
            if self.enable_manager and not is_learning and month_num % 3 == 0:  # Every 3 months
                generate_manager_feedback(monthly_results, month_str, self.db, self.run_id)
            
            # Update agent notes every 6 months (if reflection is enabled)
            if self.enable_reflection and month_num % 6 == 0:
                for agent in agent_brains:
                    trades = self.db.get_agent_trades(agent.id)
                    
                    # Get best/worst months
                    trade_results = [{'month': t.month, 'return': t.return_pct, 'factor': t.factor_choice} 
                                    for t in trades]
                    best_months = sorted(trade_results, key=lambda x: x['return'], reverse=True)[:3]
                    worst_months = sorted(trade_results, key=lambda x: x['return'])[:3]
                    
                    # Get latest manager feedback
                    from database import ManagerFeedback
                    session = self.db.get_session()
                    try:
                        manager_fb = session.query(ManagerFeedback).filter_by(
                            agent_id=agent.id
                        ).order_by(ManagerFeedback.id.desc()).first()
                        feedback_text = manager_fb.feedback_text if manager_fb else None
                    finally:
                        session.close()
                    
                    # Update notes
                    agent.update_notes(
                        market_context, 
                        decision, 
                        result, 
                        best_months, 
                        worst_months, 
                        feedback_text, 
                        month_num,
                        month_str=month_str,
                        run_id=self.run_id
                    )
        
        print("\n✅ Backtest Complete!\n")
        return {
            'run_id': self.run_id,
            'final_portfolios': agent_portfolios,
            'agents': [{
                'id': a.id,
                'name': a.name,
                'final_value': agent_portfolios[a.id],
                'return_pct': ((agent_portfolios[a.id] / 100) - 1) * 100
            } for a in agent_brains]
        }
    
    def _calculate_portfolio_return(self, decision: Dict, stock_returns: Dict) -> float:
        """Calculate portfolio return based on decision"""
        if decision['action'] == 'cash':
            return 0.0
        
        if decision['action'] == 'long':
            picks = decision.get('long_picks', [])
            if not picks:
                return 0.0
            avg_return = np.mean([stock_returns.get(stock, 0.0) for stock in picks])
            return avg_return
        
        if decision['action'] == 'short':
            picks = decision.get('short_picks', [])
            if not picks:
                return 0.0
            avg_return = np.mean([stock_returns.get(stock, 0.0) for stock in picks])
            return -avg_return  # Inverse for short
        
        return 0.0
    
    def _get_allocation(self, confidence: float) -> float:
        """Convert confidence to portfolio allocation %"""
        if confidence < 0.7:
            return 0.0
        elif confidence < 0.8:
            return 50.0
        elif confidence < 0.9:
            return 75.0
        else:
            return 100.0


# API Endpoints

@app.get("/")
def root():
    """Serve the Research Lab HTML page"""
    return FileResponse("research.html", media_type="text/html")


@app.get("/dashboard")
def dashboard():
    """Serve the Dashboard HTML page"""
    return FileResponse("dashboard.html", media_type="text/html")


@app.post("/backtest/run")
def run_backtest(request: BacktestRequest):
    """Run full backtest simulation with experiment configuration"""
    try:
        engine = BacktestEngine(db)
        results = engine.run(
            request.start_year, 
            request.end_year, 
            request.learning_years,
            enable_memory=request.enable_memory,
            enable_reflection=request.enable_reflection,
            enable_manager=request.enable_manager,
            description=request.description
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
def get_agents():
    """Get all agents"""
    agents = db.get_all_agents()
    return [
        {
            'id': a.id,
            'name': a.name,
            'personality': a.personality,
            'allowed_factors': a.allowed_factors,
            'preferred_indicators': a.preferred_indicators
        }
        for a in agents
    ]


@app.get("/agents/{agent_id}/performance")
def get_agent_performance(agent_id: int, run_id: Optional[int] = None, 
                         start_month: Optional[str] = None, end_month: Optional[str] = None):
    """Get agent performance metrics, optionally filtered by run_id and date range"""
    trades = db.get_agent_trades(agent_id)
    
    # Filter by run_id if provided
    if run_id is not None:
        trades = [t for t in trades if t.run_id == run_id]
    
    # Filter by date range if provided
    if start_month:
        trades = [t for t in trades if t.month >= start_month]
    if end_month:
        trades = [t for t in trades if t.month <= end_month]
    
    if not trades:
        return {"error": "No trades found"}
    
    returns = [t.return_pct for t in trades]
    cumulative = 100.0
    portfolio_values = []
    
    for t in trades:
        allocation = t.allocation / 100
        cumulative *= (1 + t.return_pct / 100 * allocation)
        portfolio_values.append(cumulative)
    
    return {
        'agent_id': agent_id,
        'total_trades': len(trades),
        'winning_trades': sum(1 for r in returns if r > 0),
        'losing_trades': sum(1 for r in returns if r < 0),
        'win_rate': sum(1 for r in returns if r > 0) / len(returns) * 100 if returns else 0,
        'avg_return': np.mean(returns) if returns else 0,
        'total_return': ((cumulative / 100) - 1) * 100,
        'sharpe_ratio': (np.mean(returns) / np.std(returns) * np.sqrt(12)) if len(returns) > 1 and np.std(returns) > 0 else 0,
        'max_return': max(returns) if returns else 0,
        'min_return': min(returns) if returns else 0
    }


@app.get("/agents/{agent_id}/trades")
def get_agent_trades_api(agent_id: int, limit: Optional[int] = None, run_id: Optional[int] = None):
    """Get agent trade history, optionally filtered by run_id"""
    trades = db.get_agent_trades(agent_id, limit)
    
    # Filter by run_id if provided
    if run_id is not None:
        trades = [t for t in trades if t.run_id == run_id]
    
    return [
        {
            'month': t.month,
            'factor_choice': t.factor_choice,
            'action': t.action,
            'confidence': t.confidence,
            'stocks_long': t.stocks_long,
            'stocks_short': t.stocks_short,
            'return_pct': t.return_pct,
            'allocation': t.allocation,
            'market_regime': t.market_regime
        }
        for t in trades
    ]


@app.get("/agents/{agent_id}/notes")
def get_agent_notes(agent_id: int):
    """Get agent strategy notes"""
    session = db.get_session()
    try:
        agent = session.query(Agent).filter_by(id=agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"notes": agent.strategy_notes}
    finally:
        session.close()


@app.get("/agents/{agent_id}/memories")
def get_agent_memories(agent_id: int, run_id: Optional[int] = None):
    """Get agent memories, optionally filtered by run_id"""
    memories = db.get_agent_memories(agent_id)
    
    # Filter by run_id if provided
    if run_id is not None:
        memories = [m for m in memories if m.run_id == run_id]
    
    return [
        {
            'month': m.month,
            'situation': m.situation,
            'market_regime': m.market_regime,
            'factor_used': m.factor_used,
            'result': m.result
        }
        for m in memories
    ]


@app.get("/agents/{agent_id}/feedback")
def get_agent_feedback(agent_id: int, run_id: Optional[int] = None):
    """Get manager feedback history for an agent"""
    from database import ManagerFeedback
    session = db.get_session()
    try:
        query = session.query(ManagerFeedback).filter_by(agent_id=agent_id)
        
        if run_id is not None:
            query = query.filter_by(run_id=run_id)
        
        feedback_list = query.order_by(ManagerFeedback.month).all()
        
        return [
            {
                'month': f.month,
                'feedback': f.feedback_text,
                'rank': f.rank,
                'run_id': f.run_id
            }
            for f in feedback_list
        ]
    finally:
        session.close()


@app.get("/agents/{agent_id}/notes-history")
def get_agent_notes_history(agent_id: int, run_id: Optional[int] = None):
    """Get strategy notes evolution history for an agent"""
    history = db.get_strategy_notes_history(agent_id, run_id)
    return history


@app.get("/leaderboard")
def get_leaderboard(run_id: Optional[int] = None, start_month: Optional[str] = None, end_month: Optional[str] = None):
    """Get agent rankings, optionally filtered by run_id and date range"""
    agents = db.get_all_agents()
    
    leaderboard = []
    for agent in agents:
        trades = db.get_agent_trades(agent.id)
        
        # Filter by run_id if provided
        if run_id is not None:
            trades = [t for t in trades if t.run_id == run_id]
        
        # Filter by date range if provided
        if start_month:
            trades = [t for t in trades if t.month >= start_month]
        if end_month:
            trades = [t for t in trades if t.month <= end_month]
        
        if not trades:
            continue
        
        cumulative = 100.0
        for t in trades:
            allocation = t.allocation / 100
            cumulative *= (1 + t.return_pct / 100 * allocation)
        
        returns = [t.return_pct for t in trades]
        sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        leaderboard.append({
            'agent_id': agent.id,
            'name': agent.name,
            'personality': agent.personality,
            'total_return': ((cumulative / 100) - 1) * 100,
            'sharpe_ratio': sharpe * np.sqrt(12),  # Annualized Sharpe
            'total_trades': len(trades),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns) * 100 if returns else 0
        })
    
    # Sort by total return
    leaderboard.sort(key=lambda x: x['total_return'], reverse=True)
    
    return leaderboard


@app.get("/runs")
def get_runs():
    """Get list of all backtest runs with metadata and experiment config"""
    from database import Trade
    session = db.get_session()
    try:
        # Get unique run_ids with date ranges
        runs = session.query(
            Trade.run_id,
            func.min(Trade.month).label('start_month'),
            func.max(Trade.month).label('end_month'),
            func.count(Trade.id).label('total_trades')
        ).group_by(Trade.run_id).order_by(Trade.run_id).all()
        
        result = []
        for r in runs:
            run_data = {
                'run_id': r.run_id,
                'start_month': r.start_month,
                'end_month': r.end_month,
                'total_trades': r.total_trades
            }
            # Add experiment config if available
            config = db.get_run_config(r.run_id)
            if config:
                run_data['config'] = config
            result.append(run_data)
        
        return result
    finally:
        session.close()


# ============== RESEARCH ENDPOINTS ==============

@app.get("/research/compare")
def compare_runs(run_ids: str, start_month: Optional[str] = None, end_month: Optional[str] = None):
    """
    Compare multiple runs for ablation study.
    run_ids: Comma-separated list of run IDs (e.g., "1,2,3")
    Returns aggregated metrics for each run for easy comparison.
    """
    run_id_list = [int(x.strip()) for x in run_ids.split(',')]
    
    comparison = []
    for run_id in run_id_list:
        # Get config
        config = db.get_run_config(run_id)
        
        # Get leaderboard data
        agents = db.get_all_agents()
        returns = []
        sharpes = []
        win_rates = []
        volatilities = []
        
        for agent in agents:
            trades = db.get_agent_trades(agent.id)
            trades = [t for t in trades if t.run_id == run_id]
            
            if start_month:
                trades = [t for t in trades if t.month >= start_month]
            if end_month:
                trades = [t for t in trades if t.month <= end_month]
            
            if not trades:
                continue
            
            trade_returns = [t.return_pct for t in trades]
            cumulative = 100.0
            for t in trades:
                allocation = t.allocation / 100
                cumulative *= (1 + t.return_pct / 100 * allocation)
            
            total_return = ((cumulative / 100) - 1) * 100
            sharpe = (np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(12)) if len(trade_returns) > 1 and np.std(trade_returns) > 0 else 0
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
            vol = np.std(trade_returns) if len(trade_returns) > 1 else 0
            
            returns.append(total_return)
            sharpes.append(sharpe)
            win_rates.append(win_rate)
            volatilities.append(vol)
        
        if returns:
            comparison.append({
                'run_id': run_id,
                'config': config,
                'metrics': {
                    'avg_return': np.mean(returns),
                    'median_return': np.median(returns),
                    'std_return': np.std(returns),
                    'best_return': max(returns),
                    'worst_return': min(returns),
                    'avg_sharpe': np.mean(sharpes),
                    'avg_win_rate': np.mean(win_rates),
                    'avg_volatility': np.mean(volatilities)
                },
                'agent_returns': returns  # For distribution charts
            })
    
    return comparison


@app.get("/research/psychology")
def analyze_psychology(run_id: int):
    """
    Analyze agent behavioral psychology - do rankings affect risk-taking?
    Returns data for "Desperation Index" and "Complacency" analysis.
    """
    from database import Trade
    session = db.get_session()
    
    try:
        agents = db.get_all_agents()
        psychology_data = []
        
        for agent in agents:
            # Get all trades for this agent in this run, ordered by month
            trades = session.query(Trade).filter_by(
                agent_id=agent.id, run_id=run_id
            ).order_by(Trade.year, Trade.month_num).all()
            
            if len(trades) < 2:
                continue
            
            # Calculate monthly rankings and risk metrics
            monthly_analysis = []
            
            for i, trade in enumerate(trades):
                # Get this month's trades for all agents to calculate rank
                month_trades = session.query(Trade).filter_by(
                    run_id=run_id, month=trade.month
                ).all()
                
                # Calculate cumulative returns up to this month for ranking
                agent_cumulative = {}
                for mt in month_trades:
                    prev_trades = [t for t in session.query(Trade).filter_by(
                        agent_id=mt.agent_id, run_id=run_id
                    ).all() if t.month <= trade.month]
                    
                    cumulative = 100.0
                    for pt in prev_trades:
                        allocation = pt.allocation / 100
                        cumulative *= (1 + pt.return_pct / 100 * allocation)
                    agent_cumulative[mt.agent_id] = cumulative
                
                # Rank agents
                sorted_agents = sorted(agent_cumulative.items(), key=lambda x: x[1], reverse=True)
                rank = next((idx + 1 for idx, (aid, _) in enumerate(sorted_agents) if aid == agent.id), 0)
                
                # Risk metric: confidence * number of stocks picked
                num_picks = len(trade.stocks_long or []) + len(trade.stocks_short or [])
                risk_score = trade.confidence * num_picks if trade.action != 'cash' else 0
                
                monthly_analysis.append({
                    'month': trade.month,
                    'rank': rank,
                    'total_agents': len(sorted_agents),
                    'rank_percentile': (len(sorted_agents) - rank + 1) / len(sorted_agents) * 100,
                    'confidence': trade.confidence,
                    'action': trade.action,
                    'num_picks': num_picks,
                    'risk_score': risk_score,
                    'return': trade.return_pct,
                    'is_bottom_tier': rank > len(sorted_agents) * 0.7,
                    'is_top_tier': rank <= len(sorted_agents) * 0.3
                })
            
            # Calculate psychology metrics
            bottom_tier_months = [m for m in monthly_analysis if m['is_bottom_tier']]
            top_tier_months = [m for m in monthly_analysis if m['is_top_tier']]
            
            avg_risk_when_losing = np.mean([m['risk_score'] for m in bottom_tier_months]) if bottom_tier_months else 0
            avg_risk_when_winning = np.mean([m['risk_score'] for m in top_tier_months]) if top_tier_months else 0
            
            psychology_data.append({
                'agent_id': agent.id,
                'name': agent.name,
                'personality': agent.personality,
                'desperation_index': avg_risk_when_losing - avg_risk_when_winning,  # Positive = gambles when losing
                'avg_risk_when_bottom_tier': avg_risk_when_losing,
                'avg_risk_when_top_tier': avg_risk_when_winning,
                'months_in_bottom_tier': len(bottom_tier_months),
                'months_in_top_tier': len(top_tier_months),
                'monthly_analysis': monthly_analysis  # Full detail for charts
            })
        
        # Sort by desperation index (highest = most desperate behavior)
        psychology_data.sort(key=lambda x: x['desperation_index'], reverse=True)
        
        return {
            'run_id': run_id,
            'config': db.get_run_config(run_id),
            'agents': psychology_data
        }
    
    finally:
        session.close()


@app.get("/research/impact")
def analyze_manager_impact(baseline_run_id: int, manager_run_id: int):
    """
    Analyze the specific impact of the Manager on agent behavior.
    1. "The Rescue Effect": How did the bottom 3 agents from the Baseline run perform in the Manager run?
    2. "Coachability": Did agents actually lower risk after receiving negative feedback?
    """
    from database import Trade, ManagerFeedback
    session = db.get_session()
    
    try:
        # --- PART 1: THE RESCUE EFFECT ---
        # Identify the "Losers" in the Baseline run (Bottom 3 by total return)
        baseline_trades = session.query(Trade).filter_by(run_id=baseline_run_id).all()
        
        agent_returns = {}
        for t in baseline_trades:
            if t.agent_id not in agent_returns:
                agent_returns[t.agent_id] = []
            agent_returns[t.agent_id].append(t.return_pct)
            
        # Calculate total return for ranking
        agent_totals = []
        for aid, rets in agent_returns.items():
            total = 1.0
            for r in rets:
                total *= (1 + r/100)
            agent_totals.append({'agent_id': aid, 'total_return': (total-1)*100})
            
        # Sort and pick bottom 3
        agent_totals.sort(key=lambda x: x['total_return'])
        bottom_3_ids = [a['agent_id'] for a in agent_totals[:3]]
        
        # Get names
        agents = db.get_all_agents()
        agent_map = {a.id: a.name for a in agents}
        bottom_3_names = [agent_map[aid] for aid in bottom_3_ids]
        
        # Compare their performance in Baseline vs Manager
        rescue_data = []
        
        for aid in bottom_3_ids:
            # Baseline Equity Curve
            b_trades = [t for t in baseline_trades if t.agent_id == aid]
            b_trades.sort(key=lambda x: (x.year, x.month_num))
            b_equity = [100.0]
            for t in b_trades:
                b_equity.append(b_equity[-1] * (1 + t.return_pct/100 * t.allocation/100))
                
            # Manager Equity Curve
            m_trades = session.query(Trade).filter_by(run_id=manager_run_id, agent_id=aid).order_by(Trade.year, Trade.month_num).all()
            m_equity = [100.0]
            for t in m_trades:
                m_equity.append(m_equity[-1] * (1 + t.return_pct/100 * t.allocation/100))
            
            rescue_data.append({
                'agent_id': aid,
                'name': agent_map[aid],
                'baseline_final': b_equity[-1],
                'manager_final': m_equity[-1],
                'baseline_curve': b_equity,
                'manager_curve': m_equity
            })

        # --- PART 2: COACHABILITY (FEEDBACK COMPLIANCE) ---
        # Did they listen? Check trades immediately following "risk" feedback
        feedback_events = session.query(ManagerFeedback).filter_by(run_id=manager_run_id).all()
        
        compliance_stats = {'compliant': 0, 'non_compliant': 0, 'total_risk_feedback': 0}
        
        risk_keywords = ['risk', 'exposure', 'leverage', 'cut', 'reduce', 'careful', 'high volatility']
        
        for fb in feedback_events:
            # Check if feedback was about reducing risk
            if any(k in fb.feedback_text.lower() for k in risk_keywords):
                compliance_stats['total_risk_feedback'] += 1
                
                # Find the trade for this agent in the NEXT month
                # fb.month is like "2020-03". We need the trade for "2020-04" or later
                # Simplified: just look for the next trade chronologically
                
                next_trade = session.query(Trade).filter(
                    Trade.run_id == manager_run_id,
                    Trade.agent_id == fb.agent_id,
                    Trade.month > fb.month
                ).order_by(Trade.month).first()
                
                prev_trade = session.query(Trade).filter(
                    Trade.run_id == manager_run_id,
                    Trade.agent_id == fb.agent_id,
                    Trade.month <= fb.month
                ).order_by(Trade.month.desc()).first()
                
                if next_trade and prev_trade:
                    # Did they reduce allocation or confidence?
                    did_reduce = (next_trade.allocation < prev_trade.allocation) or \
                                 (next_trade.confidence < prev_trade.confidence) or \
                                 (next_trade.action == 'cash' and prev_trade.action != 'cash')
                    
                    if did_reduce:
                        compliance_stats['compliant'] += 1
                    else:
                        compliance_stats['non_compliant'] += 1

        return {
            'rescue_analysis': rescue_data,
            'compliance_stats': compliance_stats,
            'bottom_agents': bottom_3_names
        }
        
    except Exception as e:
        print(f"Error in impact analysis: {e}")
        return {"error": str(e)}
    finally:
        session.close()


@app.get("/research/ablation-summary")
def get_ablation_summary(run_ids: str):
    """
    Get a summary formatted for ablation study visualization.
    Returns the "value added" by each component (Memory, Reflection, Manager).
    """
    run_id_list = [int(x.strip()) for x in run_ids.split(',')]
    
    # Get comparison data
    runs_data = []
    for run_id in run_id_list:
        config = db.get_run_config(run_id)
        
        # Calculate average return for this run
        agents = db.get_all_agents()
        returns = []
        
        for agent in agents:
            trades = db.get_agent_trades(agent.id)
            trades = [t for t in trades if t.run_id == run_id]
            
            if not trades:
                continue
            
            cumulative = 100.0
            for t in trades:
                allocation = t.allocation / 100
                cumulative *= (1 + t.return_pct / 100 * allocation)
            
            returns.append(((cumulative / 100) - 1) * 100)
        
        if returns:
            runs_data.append({
                'run_id': run_id,
                'config': config,
                'avg_return': np.mean(returns),
                'std_return': np.std(returns)
            })
    
    # Sort by features enabled (baseline first)
    runs_data.sort(key=lambda x: (
        x['config']['enable_memory'] if x['config'] else 0,
        x['config']['enable_reflection'] if x['config'] else 0,
        x['config']['enable_manager'] if x['config'] else 0
    ) if x['config'] else (0, 0, 0))
    
    # Calculate value added by each component
    summary = {
        'runs': runs_data,
        'value_added': {}
    }
    
    if len(runs_data) >= 2:
        baseline_return = runs_data[0]['avg_return']
        for i, run in enumerate(runs_data[1:], 1):
            prev_return = runs_data[i-1]['avg_return']
            summary['value_added'][f'run_{run["run_id"]}_vs_run_{runs_data[i-1]["run_id"]}'] = {
                'absolute': run['avg_return'] - prev_return,
                'relative_pct': ((run['avg_return'] - prev_return) / abs(prev_return) * 100) if prev_return != 0 else 0
            }
    
    return summary


if __name__ == "__main__":
    import uvicorn
    print("Starting AgenticTraders server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
