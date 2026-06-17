import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init
import questionary
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.utils.progress import progress
from src.utils.visualize import save_graph_as_png
from src.cli.input import (
    parse_cli_inputs,
)

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None


##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-5.4",
    model_provider: str = "OpenAI",
):
    # Start progress tracking
    progress.start()

    try:
        # Build workflow (default to all analysts when none provided)
        workflow = create_workflow(selected_analysts if selected_analysts else None)
        agent = workflow.compile()

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow


def save_run_analysis(result: dict, inputs, tickers: list[str]) -> None:
    """Persist each agent verdict + run/data metadata to provider_cache.db."""
    import json
    import os
    import uuid
    import datetime as _dt

    from src.tools.providers.cache_db import save_analysis_run
    from src.tools.api import search_line_items, get_market_cap
    from src.agents.warren_buffett import calculate_intrinsic_value

    items_needed = [
        "capital_expenditure", "depreciation_and_amortization", "net_income",
        "outstanding_shares", "total_assets", "total_liabilities",
        "shareholders_equity", "revenue", "free_cash_flow",
    ]
    run_id = uuid.uuid4().hex[:12]
    run_time = _dt.datetime.now().isoformat(timespec="seconds")
    data_provider = os.environ.get("DATA_PROVIDER", "financialdatasets")
    decisions = result.get("decisions") or {}

    # Per-ticker data context (served from cache, so this is cheap and adds no network).
    ctx: dict[str, dict] = {}
    for t in tickers:
        try:
            items = search_line_items(t, items_needed, inputs.end_date, "ttm", 10)
            mc = get_market_cap(t, inputs.end_date)
            iv = calculate_intrinsic_value(items).get("intrinsic_value") if items else None
            mos = (iv - mc) / mc if (iv and mc) else None
            ctx[t] = {
                "intrinsic_value": iv, "market_cap": mc, "margin_of_safety": mos,
                "data_as_of": items[0].report_period if items else None,
                "data_periods": len(items),
                "period_basis": items[0].period if items else None,  # annual / quarterly / ttm
            }
        except Exception:
            ctx[t] = {}

    rows = []
    for agent_id, sigs in (result.get("analyst_signals") or {}).items():
        for t, s in sigs.items():
            # Only persist actual analyst verdicts (skip risk/portfolio bookkeeping entries).
            if not isinstance(s, dict) or "signal" not in s:
                continue
            reasoning = s.get("reasoning")
            if reasoning is not None and not isinstance(reasoning, str):
                reasoning = json.dumps(reasoning, default=str)
            try:
                confidence = float(s["confidence"]) if s.get("confidence") is not None else None
            except (TypeError, ValueError):
                confidence = None
            c = ctx.get(t, {})
            rows.append({
                "run_id": run_id, "ticker": t, "agent": agent_id,
                "signal": s.get("signal"), "confidence": confidence,
                "reasoning": reasoning,
                "intrinsic_value": c.get("intrinsic_value"), "market_cap": c.get("market_cap"),
                "margin_of_safety": c.get("margin_of_safety"),
                "data_as_of": c.get("data_as_of"), "data_periods": c.get("data_periods"),
                "period_basis": c.get("period_basis"), "data_provider": data_provider,
                "run_time": run_time,
                "metadata": json.dumps({"decision": decisions.get(t)}),
            })

    basis = next((c.get("period_basis") for c in ctx.values() if c.get("period_basis")), None)
    run_meta = {
        "run_id": run_id, "run_time": run_time, "tickers": tickers,
        "agents": inputs.selected_analysts, "model": inputs.model_name,
        "model_provider": inputs.model_provider, "data_provider": data_provider,
        "period_basis": basis, "start_date": inputs.start_date, "end_date": inputs.end_date,
    }
    save_analysis_run(run_meta, rows)
    print(f"\n\U0001F4C1 Saved {len(rows)} verdict(s) to provider_cache.db "
          f"(run_id={run_id}, provider={data_provider}, basis={basis})")


if __name__ == "__main__":
    inputs = parse_cli_inputs(
        description="Run the hedge fund trading system",
        require_tickers=True,
        default_months_back=None,
        include_graph_flag=True,
        include_reasoning_flag=True,
    )

    tickers = inputs.tickers
    selected_analysts = inputs.selected_analysts

    # Construct portfolio here
    portfolio = {
        "cash": inputs.initial_cash,
        "margin_requirement": inputs.margin_requirement,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            }
            for ticker in tickers
        },
    }

    result = run_hedge_fund(
        tickers=tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        portfolio=portfolio,
        show_reasoning=inputs.show_reasoning,
        selected_analysts=inputs.selected_analysts,
        model_name=inputs.model_name,
        model_provider=inputs.model_provider,
    )
    print_trading_output(result)

    if getattr(inputs, "save_analysis", True):
        try:
            save_run_analysis(result, inputs, tickers)
        except Exception as e:
            print(f"(warning: could not save analysis to db: {e})")
