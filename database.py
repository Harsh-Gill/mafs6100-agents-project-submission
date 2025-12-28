"""
AgenticTraders - Database Models
Simple SQLite database for storing agents, trades, memories, and performance.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class Agent(Base):
    __tablename__ = 'agents'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    personality = Column(String(50), nullable=False)  # Conservative, Aggressive, Contrarian, etc.
    allowed_factors = Column(JSON, nullable=False)  # ["momentum", "mean_reversion", "volatility"]
    preferred_indicators = Column(JSON, nullable=False)  # ["RSI", "MACD", "Bollinger"]
    strategy_notes = Column(Text, default="")  # Evolving strategy document
    created_at = Column(DateTime, default=datetime.now)

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, nullable=False, default=1)  # Track different backtest runs
    agent_id = Column(Integer, nullable=False)
    month = Column(String(20), nullable=False)  # "2018-01"
    year = Column(Integer, nullable=False)
    month_num = Column(Integer, nullable=False)
    
    # Decision details
    factor_choice = Column(String(50))
    action = Column(String(20))  # "long", "short", "cash"
    confidence = Column(Float)
    stocks_long = Column(JSON)  # ["TSLA", "NVDA"]
    stocks_short = Column(JSON)  # ["MSFT"]
    
    # Market conditions
    market_regime = Column(String(50))
    volatility = Column(Float)
    spy_return_1m = Column(Float)
    spy_return_3m = Column(Float)
    
    # Results
    return_pct = Column(Float)
    allocation = Column(Float)  # % of portfolio allocated
    
    created_at = Column(DateTime, default=datetime.now)

class Memory(Base):
    __tablename__ = 'memories'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, nullable=False, default=1)  # Track different backtest runs
    agent_id = Column(Integer, nullable=False)
    month = Column(String(20), nullable=False)
    
    # Memory content
    situation = Column(Text, nullable=False)  # Description of market + decision + result
    embedding = Column(JSON)  # Vector embedding for RAG retrieval
    
    # Context
    market_regime = Column(String(50))
    factor_used = Column(String(50))
    action = Column(String(20))
    result = Column(Float)
    
    created_at = Column(DateTime, default=datetime.now)

class Performance(Base):
    __tablename__ = 'performance'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, nullable=False)
    month = Column(String(20), nullable=False)
    
    # Cumulative metrics
    cumulative_return = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    
    # Rolling metrics
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    
    updated_at = Column(DateTime, default=datetime.now)

class ManagerFeedback(Base):
    __tablename__ = 'manager_feedback'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, nullable=False, default=1)  # Track different backtest runs
    agent_id = Column(Integer, nullable=False)
    month = Column(String(20), nullable=False)
    
    feedback_text = Column(Text, nullable=False)
    rank = Column(Integer)  # Rank among all agents (1 = best)
    
    created_at = Column(DateTime, default=datetime.now)

class StrategyNotesHistory(Base):
    __tablename__ = 'strategy_notes_history'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, nullable=False, default=1)  # Track different backtest runs
    agent_id = Column(Integer, nullable=False)
    month = Column(String(20), nullable=False)  # Month when notes were updated
    
    notes_text = Column(Text, nullable=False)  # Snapshot of strategy notes at this time
    notes_version = Column(Integer, default=1)  # Version number for this agent in this run
    
    created_at = Column(DateTime, default=datetime.now)


class RunConfig(Base):
    """Store experimental configuration for each run"""
    __tablename__ = 'run_configs'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, unique=True, nullable=False)
    
    # Experiment configuration flags
    enable_memory = Column(Integer, default=1)      # 1=True, 0=False - RAG memory retrieval
    enable_reflection = Column(Integer, default=1)  # 1=True, 0=False - Strategy notes updates
    enable_manager = Column(Integer, default=1)     # 1=True, 0=False - Manager feedback
    
    # Run metadata
    start_year = Column(Integer)
    end_year = Column(Integer)
    learning_years = Column(Integer)
    description = Column(String(200))  # Human-readable experiment name
    
    created_at = Column(DateTime, default=datetime.now)


class Database:
    """Simple database manager"""
    
    def __init__(self, db_path='agentic_traders.db'):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        return self.Session()
    
    def init_agents(self, agents_config):
        """Initialize agents from config"""
        session = self.get_session()
        try:
            for config in agents_config:
                existing = session.query(Agent).filter_by(name=config['name']).first()
                if not existing:
                    agent = Agent(**config)
                    session.add(agent)
            session.commit()
        finally:
            session.close()
    
    def get_all_agents(self):
        """Get all agents"""
        session = self.get_session()
        try:
            return session.query(Agent).all()
        finally:
            session.close()
    
    def get_agent_trades(self, agent_id, limit=None):
        """Get trades for an agent"""
        session = self.get_session()
        try:
            query = session.query(Trade).filter_by(agent_id=agent_id).order_by(Trade.year, Trade.month_num)
            if limit:
                query = query.limit(limit)
            return query.all()
        finally:
            session.close()
    
    def get_agent_memories(self, agent_id, run_id=None):
        """Get all memories for an agent, optionally filtered by run_id"""
        session = self.get_session()
        try:
            query = session.query(Memory).filter_by(agent_id=agent_id)
            if run_id is not None:
                query = query.filter_by(run_id=run_id)
            return query.all()
        finally:
            session.close()
    
    def save_trade(self, trade_data):
        """Save a trade"""
        session = self.get_session()
        try:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            return trade.id
        finally:
            session.close()
    
    def save_memory(self, memory_data):
        """Save a memory"""
        session = self.get_session()
        try:
            memory = Memory(**memory_data)
            session.add(memory)
            session.commit()
        finally:
            session.close()
    
    def update_agent_notes(self, agent_id, notes, month=None, run_id=1):
        """Update agent strategy notes and save history"""
        session = self.get_session()
        try:
            agent = session.query(Agent).filter_by(id=agent_id).first()
            if agent:
                agent.strategy_notes = notes
                
                # Save history snapshot if month is provided
                if month:
                    # Get the next version number for this agent in this run
                    max_version = session.query(func.max(StrategyNotesHistory.notes_version)).filter(
                        StrategyNotesHistory.agent_id == agent_id,
                        StrategyNotesHistory.run_id == run_id
                    ).scalar()
                    next_version = (max_version or 0) + 1
                    
                    # Create history record
                    history = StrategyNotesHistory(
                        agent_id=agent_id,
                        run_id=run_id,
                        month=month,
                        notes_text=notes,
                        notes_version=next_version
                    )
                    session.add(history)
                
                session.commit()
        finally:
            session.close()
    
    def update_performance(self, perf_data):
        """Update agent performance metrics"""
        session = self.get_session()
        try:
            perf = session.query(Performance).filter_by(
                agent_id=perf_data['agent_id'], 
                month=perf_data['month']
            ).first()
            
            if perf:
                for key, value in perf_data.items():
                    setattr(perf, key, value)
            else:
                perf = Performance(**perf_data)
                session.add(perf)
            
            session.commit()
        finally:
            session.close()
    
    def save_manager_feedback(self, feedback_data):
        """Save manager feedback"""
        session = self.get_session()
        try:
            feedback = ManagerFeedback(**feedback_data)
            session.add(feedback)
            session.commit()
        finally:
            session.close()
    
    def get_next_run_id(self):
        """Get the next run_id for a new backtest"""
        session = self.get_session()
        try:
            max_run_id = session.query(func.max(Trade.run_id)).scalar()
            return (max_run_id or 0) + 1
        finally:
            session.close()
    
    def get_strategy_notes_history(self, agent_id, run_id=None):
        """Get strategy notes history for an agent"""
        session = self.get_session()
        try:
            query = session.query(StrategyNotesHistory).filter_by(agent_id=agent_id)
            if run_id:
                query = query.filter_by(run_id=run_id)
            history = query.order_by(StrategyNotesHistory.month).all()
            return [{
                'month': h.month,
                'notes': h.notes_text,
                'version': h.notes_version,
                'run_id': h.run_id,
                'created_at': h.created_at.isoformat()
            } for h in history]
        finally:
            session.close()
    
    def save_run_config(self, run_id, config):
        """Save experimental configuration for a run"""
        session = self.get_session()
        try:
            run_config = RunConfig(
                run_id=run_id,
                enable_memory=1 if config.get('enable_memory', True) else 0,
                enable_reflection=1 if config.get('enable_reflection', True) else 0,
                enable_manager=1 if config.get('enable_manager', True) else 0,
                start_year=config.get('start_year'),
                end_year=config.get('end_year'),
                learning_years=config.get('learning_years'),
                description=config.get('description', '')
            )
            session.add(run_config)
            session.commit()
        finally:
            session.close()
    
    def get_run_config(self, run_id):
        """Get experimental configuration for a run"""
        session = self.get_session()
        try:
            config = session.query(RunConfig).filter_by(run_id=run_id).first()
            if config:
                return {
                    'run_id': config.run_id,
                    'enable_memory': bool(config.enable_memory),
                    'enable_reflection': bool(config.enable_reflection),
                    'enable_manager': bool(config.enable_manager),
                    'start_year': config.start_year,
                    'end_year': config.end_year,
                    'learning_years': config.learning_years,
                    'description': config.description,
                    'created_at': config.created_at.isoformat()
                }
            return None
        finally:
            session.close()
    
    def get_all_run_configs(self):
        """Get all run configurations"""
        session = self.get_session()
        try:
            configs = session.query(RunConfig).order_by(RunConfig.run_id).all()
            return [{
                'run_id': c.run_id,
                'enable_memory': bool(c.enable_memory),
                'enable_reflection': bool(c.enable_reflection),
                'enable_manager': bool(c.enable_manager),
                'start_year': c.start_year,
                'end_year': c.end_year,
                'learning_years': c.learning_years,
                'description': c.description,
                'created_at': c.created_at.isoformat()
            } for c in configs]
        finally:
            session.close()
