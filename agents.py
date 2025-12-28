"""
AgenticTraders - Agent Intelligence & Memory
Handles agent decision-making, RAG memory retrieval, and strategy note updates.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional
import requests
from sklearn.metrics.pairwise import cosine_similarity

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class AgentBrain:
    """AI-powered trading agent with memory and learning"""
    
    def __init__(self, agent_db_obj, database):
        self.id = agent_db_obj.id
        self.name = agent_db_obj.name
        self.personality = agent_db_obj.personality
        self.allowed_factors = agent_db_obj.allowed_factors
        self.preferred_indicators = agent_db_obj.preferred_indicators
        self.strategy_notes = agent_db_obj.strategy_notes
        self.db = database
    
    def make_decision(self, market_data: Dict, month_num: int, is_learning_phase: bool, 
                      enable_memory: bool = True) -> Dict:
        """
        Agent makes trading decision based on market data and memories.
        Returns: {factor_choice, action, confidence, long_picks, short_picks}
        
        Args:
            enable_memory: If False, RAG memory retrieval is disabled (ablation study)
        """
        # Retrieve relevant memories via RAG (if enabled)
        if enable_memory:
            memories = self._retrieve_relevant_memories(market_data, top_k=3)
        else:
            memories = []  # Ablation: No memory access
        
        # Build prompt
        prompt = self._build_decision_prompt(market_data, memories, month_num, is_learning_phase)
        
        # Get AI decision
        response = self._call_llm(prompt)
        decision = self._parse_decision(response)
        
        return decision
    
    def update_notes(self, market_data: Dict, decision: Dict, result: float, 
                     best_months: List[Dict], worst_months: List[Dict], 
                     manager_feedback: Optional[str], month_num: int,
                     month_str: str = None, run_id: int = 1) -> str:
        """
        Agent reflects on performance and updates strategy notes.
        Returns: Updated strategy notes
        """
        # Format best/worst months
        best_str = "\n".join([f"• Month {m['month']}: {m['return']:+.2f}% ({m['factor']})" 
                              for m in best_months])
        worst_str = "\n".join([f"• Month {m['month']}: {m['return']:+.2f}% ({m['factor']})" 
                               for m in worst_months])
        
        prompt = f"""
# YOUR CURRENT NOTES
{self.strategy_notes if self.strategy_notes else "No notes yet."}

# WHAT JUST HAPPENED (Month {month_num})
- Market: {market_data['regime']}
- Volatility: {market_data['volatility']:.1f}%
- SPY: {market_data['spy_1m']:+.1f}% (1m), {market_data['spy_3m']:+.1f}% (3m)
- Your Factor: {decision['factor_choice']}
- Your Action: {decision['action']}
- Your Confidence: {decision['confidence']:.2f}
- Your Longs: {decision.get('long_picks', [])}
- Your Shorts: {decision.get('short_picks', [])}
- **RESULT: {result:+.2f}%**

# YOUR 3 BEST MONTHS
{best_str if best_str else "None yet"}

# YOUR 3 WORST MONTHS
{worst_str if worst_str else "None yet"}

# MANAGER FEEDBACK
{f"Your manager says: {manager_feedback}" if manager_feedback else "No manager feedback this month."}

# TASK
Update your strategy notes. What did you learn? What patterns are emerging?
Incorporate manager feedback if provided. Focus on: When does {decision['factor_choice']} work? When does it fail?

Format:
---
WHAT WORKS FOR ME:
• [specific patterns that lead to gains]

WHAT FAILS:
• [specific patterns that lead to losses]

FACTOR INSIGHTS:
• [when to use each of my factors: {self.allowed_factors}]

MY RULES:
• [concrete rules for when to trade vs sit out]

RECENT LESSONS:
• [what I just learned this month]
---
"""
        
        new_notes = self._call_llm(prompt, system_prompt=f"You are {self.name}, a {self.personality} trader. Update your strategy notes based on recent performance.")
        
        # Save to database with history tracking
        self.db.update_agent_notes(self.id, new_notes, month=month_str, run_id=run_id)
        self.strategy_notes = new_notes
        
        return new_notes
    
    def create_memory(self, market_data: Dict, decision: Dict, result: float, month: str, run_id: int = 1):
        """
        Store experience as a memory with embedding for RAG retrieval.
        """
        # Create memory text
        situation = f"""
Market: {market_data['regime']}, Volatility: {market_data['volatility']:.1f}%, SPY 1m: {market_data['spy_1m']:+.1f}%
Factor: {decision['factor_choice']}, Action: {decision['action']}, Confidence: {decision['confidence']:.2f}
Picks: Long={decision.get('long_picks', [])}, Short={decision.get('short_picks', [])}
Result: {result:+.2f}%
"""
        
        # Get embedding
        embedding = self._get_embedding(situation)
        
        # Save to database
        memory_data = {
            'run_id': run_id,
            'agent_id': self.id,
            'month': month,
            'situation': situation,
            'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            'market_regime': market_data['regime'],
            'factor_used': decision['factor_choice'],
            'action': decision['action'],
            'result': result
        }
        
        self.db.save_memory(memory_data)
    
    def _retrieve_relevant_memories(self, market_data: Dict, top_k: int = 3) -> List[str]:
        """
        RAG: Retrieve top-k most similar past experiences.
        """
        # Create query embedding from current market situation
        query_text = f"Market: {market_data['regime']}, Volatility: {market_data['volatility']:.1f}%, SPY: {market_data['spy_1m']:+.1f}%"
        query_embedding = self._get_embedding(query_text)
        
        # Get all memories
        memories = self.db.get_agent_memories(self.id)
        
        if not memories:
            return []
        
        # Calculate similarities
        memory_embeddings = np.array([m.embedding for m in memories])
        query_emb_array = np.array(query_embedding).reshape(1, -1)
        
        similarities = cosine_similarity(query_emb_array, memory_embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [memories[i].situation for i in top_indices]
    
    def _build_decision_prompt(self, market_data: Dict, memories: List[str], month_num: int, is_learning: bool) -> str:
        """Build decision-making prompt"""
        
        phase_context = "LEARNING PHASE - Focus on exploration and building your strategy." if is_learning else "DEPLOYMENT PHASE - Execute your refined strategy."
        
        memories_str = "\n---\n".join(memories) if memories else "No relevant past experiences yet."
        
        return f"""
You are {self.name}, a {self.personality} trader.

{phase_context}

# YOUR STRATEGY NOTES
{self.strategy_notes if self.strategy_notes else "You haven't developed a strategy yet. This is your chance to learn."}

# RELEVANT PAST EXPERIENCES (via RAG)
{memories_str}

# CURRENT MARKET (Month {month_num})
- Regime: {market_data['regime']}
- Volatility: {market_data['volatility']:.1f}%
- SPY 1m return: {market_data['spy_1m']:+.1f}%
- SPY 3m return: {market_data['spy_3m']:+.1f}%

# YOUR TOOLS
- Allowed Factors: {', '.join(self.allowed_factors)}
- Preferred Indicators: {', '.join(self.preferred_indicators)}
- Universe: TSLA, NVDA, MSFT, AMZN, GOOGL, PLTR, NDX

# DECISION TASK
Decide your trade for this month. You can:
1. Go LONG (pick 1-3 stocks)
2. Go SHORT (pick 1-3 stocks)
3. Stay in CASH (sit out)

Provide confidence (0-1). Your allocation will be:
- 0.7-0.8 confidence → 50% portfolio
- 0.8-0.9 confidence → 75% portfolio
- 0.9+ confidence → 100% portfolio
- Below 0.7 → Stay cash

Respond in JSON:
{{
  "factor_choice": "momentum|mean_reversion|volatility|sentiment",
  "action": "long|short|cash",
  "confidence": 0.85,
  "long_picks": ["NVDA", "TSLA"],
  "short_picks": [],
  "reasoning": "Brief explanation"
}}
"""
    
    def _parse_decision(self, response: str) -> Dict:
        """Parse LLM response into decision dict"""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            decision = json.loads(json_str)
            
            # Validate
            if decision['action'] == 'cash':
                decision['long_picks'] = []
                decision['short_picks'] = []
            
            return decision
        except:
            # Fallback to cash if parsing fails
            return {
                'factor_choice': self.allowed_factors[0],
                'action': 'cash',
                'confidence': 0.5,
                'long_picks': [],
                'short_picks': [],
                'reasoning': 'Failed to parse decision, staying in cash'
            }
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for RAG (simple version using model)"""
        # For simplicity, use a mock embedding
        # In production, use OpenAI embeddings or similar
        return np.random.rand(384)  # Mock 384-dim embedding
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call OpenRouter API"""
        
        if not OPENROUTER_API_KEY:
            # Mock response for testing without API key
            print(f"⚠️  WARNING: No OPENROUTER_API_KEY found! Using mock decisions.")
            return '{"factor_choice": "momentum", "action": "cash", "confidence": 0.6, "long_picks": [], "short_picks": [], "reasoning": "Testing mode - no API key"}'
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "google/gemini-2.5-flash",
            "messages": messages
        }
        
        try:
            print(f"🤖 Calling API for {self.name}...")
            response = requests.post(OPENROUTER_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()['choices'][0]['message']['content']
            print(f"✅ API call successful for {self.name}")
            return result
        except Exception as e:
            print(f"❌ LLM call failed for {self.name}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text}")
            # Return safe default
            return '{"factor_choice": "momentum", "action": "cash", "confidence": 0.5, "long_picks": [], "short_picks": [], "reasoning": "API error"}'


def generate_manager_feedback(agents_performance: List[Dict], month: str, db, run_id: int = 1) -> None:
    """
    Manager AI evaluates all agents and provides feedback using LLM.
    agents_performance: [{'agent_id': 1, 'name': 'Agent A', 'return': 5.2, 'cumulative_return': 15.3, ...}]
    """
    # Rank agents
    ranked = sorted(agents_performance, key=lambda x: x['cumulative_return'], reverse=True)
    
    # Get recent trades for context
    for rank, agent_perf in enumerate(ranked, 1):
        agent_id = agent_perf['agent_id']
        
        # Get agent's recent trading history for context
        trades = db.get_agent_trades(agent_id)
        if trades:
            recent_trades = [t for t in trades if t.run_id == run_id][-5:]  # Last 5 trades
            trade_summary = "\n".join([
                f"  • {t.month}: {t.action.upper()} via {t.factor_choice} → {t.return_pct:+.2f}%"
                for t in recent_trades
            ])
        else:
            trade_summary = "No recent trades available."
        
        # Get agent's current strategy notes
        from database import Agent
        session = db.get_session()
        try:
            agent = session.query(Agent).filter_by(id=agent_id).first()
            strategy_notes = agent.strategy_notes if agent and agent.strategy_notes else "No strategy developed yet."
        finally:
            session.close()
        
        # Build performance context
        total_agents = len(ranked)
        performance_tier = "top tier" if rank <= 3 else "bottom tier" if rank >= total_agents - 2 else "mid tier"
        
        # Create manager feedback prompt
        prompt = f"""
You are evaluating trader: {agent_perf['name']}

# CURRENT PERFORMANCE
- Rank: {rank} out of {total_agents}
- This Month Return: {agent_perf['return']:+.2f}%
- Cumulative Return: {agent_perf['cumulative_return']:+.2f}%
- Performance Tier: {performance_tier}

# RECENT TRADING HISTORY
{trade_summary}

# TRADER'S CURRENT STRATEGY NOTES
{strategy_notes[:500]}{"..." if len(strategy_notes) > 500 else ""}

# YOUR TASK
As a senior hedge fund manager, provide constructive feedback to this trader. Be:
- **Balanced**: Acknowledge both strengths and weaknesses
- **Specific**: Reference their actual trades and strategy
- **Actionable**: Give concrete advice they can implement
- **Critical but supportive**: Push them to improve while maintaining morale

Keep it concise (2-4 sentences). Focus on what matters most for their development.
"""

        system_prompt = """You are a seasoned hedge fund manager with 20 years of experience. 
You've seen countless traders succeed and fail. You're known for being tough but fair - you give honest feedback 
that helps traders grow. You don't sugarcoat, but you also recognize good work when you see it. 
Your feedback is specific, actionable, and grounded in real performance data."""

        # Call LLM for personalized feedback
        if OPENROUTER_API_KEY:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
            "model": "google/gemini-2.5-flash",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }
            
            try:
                print(f"🏦 Manager evaluating {agent_perf['name']} (Rank {rank})...")
                response = requests.post(OPENROUTER_URL, headers=headers, json=data)
                response.raise_for_status()
                feedback = response.json()['choices'][0]['message']['content'].strip()
                print(f"✅ Feedback generated for {agent_perf['name']}")
            except Exception as e:
                print(f"⚠️  Manager feedback API failed: {e}")
                # Fallback to simple feedback
                if rank <= 3:
                    feedback = f"Top {rank} performer. Strong execution this period."
                elif rank >= total_agents - 2:
                    feedback = f"Bottom tier. Your strategy needs significant refinement."
                else:
                    feedback = f"Mid-pack at rank {rank}. Find your edge and exploit it consistently."
        else:
            # No API key - use simple feedback
            if rank <= 3:
                feedback = f"Top {rank} performer. Strong execution this period."
            elif rank >= total_agents - 2:
                feedback = f"Bottom tier. Your strategy needs significant refinement."
            else:
                feedback = f"Mid-pack at rank {rank}. Find your edge and exploit it consistently."
        
        # Save feedback
        db.save_manager_feedback({
            'run_id': run_id,
            'agent_id': agent_id,
            'month': month,
            'feedback_text': feedback,
            'rank': rank
        })
