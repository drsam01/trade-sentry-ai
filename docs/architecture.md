# Document Heading

This is the architecture of TradeSentry Strategy-as-a-Service (SaaS) AI platform

trade-sentry-ai/
│
├── apps/                                        # Individual strategy microservice
│   ├── breakout/                                # Breakout strategy (BB, ORB, London Open)
│   |   ├── scanner.py                           # Multi-symbol/session breakout scanner
│   |   ├── strategy.py                          # Strategy decision logic
│   |   ├── backtest_runner.py                   # Backtest runner for BB/ORB logic
│   |   ├── live_trader.py                       # Live trading executor for breakouts
│   |   └── config.yaml                          # Session mapping, confirmation filters
|   |
|   ├── drlfusion/
|   |   ├── strategy.py                          # Main strategy logic using Ensemble DRL agents
|   |   ├── helpers.py                           # Utility functions (reward, obs prep, etc.)
|   |   ├── orchestrator.py                      # Multi-symbol orchestration with DRL agents
|   |   ├── cli_live.py                          # CLI entry point for live trading
|   |   ├── configs/
|   |   │   └── drlfusion_config.yaml            # Strategy config
|   |   ├── backtest/
|   |   │   ├── cli_backtest.py                  # CLI for backtesting and walk-forward analysis
|   |   │   ├── backtest_runner.py               # Backtesting logic per agent/config
|   |   │   ├── walk_forward.py                  # Walk-forward testing logic
|   |   │   └── drlfusion_backtest_config.yaml   # Parameter + symbol configuration
|   |   ├── envs/
|   |   │   ├── simulation_env.py                # Custom trading gym environment
|   |   │   └── simulation_env_helpers.py        # Observation, reward, and episode logic
|   |   ├── agents/
|   |   │   ├── ppo_agent.py                     # PPO agent action inference logic
|   |   │   ├── dqn_agent.py                     # DQN action decision wrapper
|   |   │   ├── sac_agent.py                     # Soft Actor-Critic logic
|   |   │   └── a2c_agent.py                     # Advantage Actor-Critic inference logic
|   |   ├── trainers/
|   |   │   ├── ppo_trainer.py                   # PPO training implementation
|   |   │   ├── dqn_trainer.py                   # DQN training logic
|   |   │   ├── sac_trainer.py                   # SAC training loop
|   |   │   └── a2c_trainer.py                   # A2C model training
|   |   └── models/
|   |       ├── ppo_model.pt                     # Trained PPO PyTorch model
|   |       ├── dqn_model.pt                     # Trained DQN model
|   |       ├── sac_model.pt                     # Trained SAC model
|   |       └── a2c_model.pt                     # Trained A2C model
|   ├── smclassic/
│   │   ├── strategy.py                          # Core SMClassicStrategy class (live & backtest compatible)
│   │   ├── helpers.py                           # Grid and zone utilities (midpoint reset, ATR, etc.)
│   │   ├── orchestrator.py                      # SMClassicAsyncOrchestrator and PortfolioOrchestrator
│   │   ├── cli_live.py                          # CLI entry point for live trading execution
│   │   ├── configs/
│   │   │   └── smclassic_config.py              # Config
│   │   └── backtest/
│   │       ├── cli_backtest.py                  # CLI for launching backtests or walk-forward tests
│   │       ├── backtest_runner.py               # Batched backtest logic with multi-symbol and variant support
│   │       ├── walk_forward.py                  # Walk-forward execution with rebalancing and top-N filtering
│   │       └── smclassic_backtest_config.yaml   # Config dictionary including symbol list, variant grid, optimization space
│   │
│   ├── smedge/                     # Smart Money Concepts with trend + SMC zones
│   │   ├── signals.py              # Entry signal generator (ChoCH, BOS)
│   │   ├── strategy.py             # Strategy orchestration logic
│   │   ├── zones.py                # OB, FVG, and mitigation zone extractors
│   │   ├── backtest_runner.py      # Backtest entry point for SMEdge
│   │   ├── live_trader.py          # Async trader for SMEdge
│   │   └── config.yaml             # Parameters for zone logic, risk, timeframes
│   │
|   └── smgrid/
│       ├── strategy.py                         # Core SMGridStrategy class (live & backtest compatible)
│       ├── helpers.py                          # Grid and zone utilities (midpoint reset, ATR, etc.)
│       ├── orchestrator.py                     # SMGridAsyncOrchestrator and PortfolioOrchestrator
│       ├── cli_live.py                         # CLI entry point for live trading execution
│       ├── configs/
│       │   └── smgrid_config.py                # Config
│       └── backtest/
│           ├── cli_backtest.py                 # CLI for launching backtests or walk-forward tests
│           ├── backtest_runner.py              # Batched backtest logic with multi-symbol and variant support
│           ├── walk_forward.py                 # Walk-forward execution with rebalancing and top-N filtering
│           └── smgrid_backtest_config.yaml     # Config dictionary including symbol list, variant grid, optimization space
|
├── core/                           # Shared framework components
│   ├── broker/                     # Broker integrations
│   │   ├── base_broker.py          # Abstract interface: place_order(), get_balance()
│   │   ├── mt5.py                  # MetaTrader 5 connector
│   │   └── oanda.py                # OANDA broker implementation (future)
│   │
│   ├── execution/                  # Trade execution and risk logic
│   │   ├── order_manager.py        # Submit, update, cancel, multi-leg handling
│   │   ├── risk_manager.py         # SL/TP rules, dynamic lot size, break-even
│   │   └── portfolio_manager.py    # Multi-symbol/multi-strategy exposure control
│   │
│   ├── strategy/                   # Strategy framework interfaces
│   │   ├── base_strategy.py        # Abstract base for all strategies
│   │   └── strategy_registry.py    # Dynamic loader for strategies by name
│   │
│   ├── utils/                      # General-purpose tools
│   │   ├── config_loader.py        # YAML/JSON config loading + schema check
│   │   ├── notifier.py             # Telegram/email/webhook alert system
│   │   └── logger.py               # Centralized logger setup with rotation
│   │
│   └── monitoring/                 # Live and backtest monitoring tools
│       ├── metrics_collector.py    # Track win rate, PnL, drawdown
│       ├── trade_tracker.py        # Logs trades, equity, SL/TP hits, R-multiples
│       └── trade_watcher.py        # Monitors open trades, applies trailing SL, exits
│
├── backtest/                       # Generic backtest engine
│   ├── engine.py                   # Unified backtesting loop
│   ├── walk_forward.py             # Walk-forward + parameter grid support
│   ├── evaluation.py               # Metric calculation and visual summaries
│   └── cli.py                      # CLI interface to backtest strategy of choice
│
├── api_gateway/                    # REST API microservice
│   ├── main.py                     # FastAPI root app instance
│   ├── routes/                     # Modular route handlers
│   │   ├── strategies.py           # Launch, pause, configure strategies
│   │   ├── broker.py               # Broker link/validation endpoints
│   │   ├── wallet.py               # Balance, equity, performance fee data
│   │   └── auth.py                 # Signup/login/refresh/logout
│   └── auth/                       # JWT and password security
│       ├── jwt_handler.py          # Token issuance and decoding
│       └── password_hash.py        # Bcrypt-based password hashing & verification
│
├── ui/                             # Client-facing interfaces
│   ├── web/                        # Web dashboard (React)
│   │   ├── public/
│   │   ├── src/
│   │   │   ├── pages/              # Strategy marketplace, wallet, trades
│   │   │   ├── components/         # Strategy cards, live trade tables, charts
│   │   │   ├── services/           # Axios API bindings
│   │   │   └── App.tsx
│   │   └── package.json
│   └── mobile/                     # Optional: React Native or Flutter app
│       ├── android/
│       ├── ios/
│       └── lib/ or src/
│
├── database/                       # PostgreSQL schema + ORM
│   ├── schema.sql                  # Initial schema: users, trades, wallets, logs
│   ├── models.py                   # SQLAlchemy ORM models
│   └── migrations/                 # Alembic migration scripts
│
├── infra/                          # Infrastructure-as-code (IaC)
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.web
│   │   ├── Dockerfile.worker
│   ├── docker-compose.yaml         # Orchestration (API, DB, Web, Broker)
│   ├── nginx.conf                  # Reverse proxy + routing
│   └── .env.example                # Template .env for dev/staging
│
├── docs/                           # Documentation
│   ├── architecture.md             # High-level architecture overview
│   ├── api_reference.md            # REST API endpoints
│   └── strategy_guides/            # Per-strategy implementation notes
│
├── scripts/                        # Helper/automation scripts
│   ├── init_db.py                  # Create schema and seed roles/users
│   ├── run_backtest_all.py         # Batch test all active strategies
│   └── deploy.sh                   # CI/CD shell deployment script
│
├── tests/                          # Unit and integration tests
│   ├── unit/                       # Fast, logic-focused unit tests
│   │   ├── test_config_loader.py
│   │   ├── test_strategy_registry.py
│   │   └── ...
│   └── integration/                # Full system/flow tests
│       ├── test_live_trader.py
│       ├── test_api_endpoints.py
│       └── ...
│
├── .env                            # Live environment variables (excluded from git)
├── .gitignore
├── requirements.txt                # Python dependencies
├── README.md                       # Developer setup and overview
└── pyproject.toml                  # If using Poetry or modern packaging
