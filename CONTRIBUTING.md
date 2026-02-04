# Contributing to Market Event AI

Thank you for your interest in contributing to Market Event AI! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, professional, and constructive. We're all here to learn and build something useful.

## How to Contribute

### 1. Issues

#### Reporting Bugs

Before creating a bug report:
- Check existing issues to avoid duplicates
- Use the latest version
- Provide minimal reproducible example

Include in your bug report:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces
- Relevant configuration

#### Suggesting Features

Feature suggestions are welcome! Include:
- Use case and motivation
- Proposed API or interface
- Alternative solutions considered
- Willingness to implement

### 2. Pull Requests

#### Setup Development Environment

```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR_USERNAME/news-driven-market-reaction.git
cd news-driven-market-reaction
git checkout Trump

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (if available)
pre-commit install
```

#### Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/issue-123
   ```

2. **Make changes**
   - Write code following existing style
   - Add/update tests
   - Update documentation
   - Add type hints
   - Add docstrings

3. **Test your changes**
   ```bash
   # Run tests
   pytest
   
   # Run specific tests
   pytest tests/unit/test_mymodule.py
   
   # Check coverage
   pytest --cov=market_event_ai --cov-report=html
   
   # Run integration tests
   pytest tests/integration/
   ```

4. **Format code**
   ```bash
   # Format with black
   make format
   
   # Or manually
   black src/ tests/
   ruff check src/ tests/ --fix
   ```

5. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # or
   git commit -m "fix: resolve issue #123"
   ```

   Commit message format:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation
   - `test:` - Tests
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvement
   - `chore:` - Maintenance

6. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   ```
   
   Then create a pull request on GitHub.

#### Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: What, why, and how
- **Tests**: All tests must pass
- **Documentation**: Update if needed
- **Code style**: Follow existing conventions
- **Size**: Keep PRs focused and reasonably sized

### 3. Code Style

#### Python Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions small and focused
- Use meaningful variable names

Example:

```python
def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
    
    Returns:
        Annualized Sharpe ratio
    
    Example:
        >>> returns = np.array([0.01, 0.02, -0.01])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"{sharpe:.2f}")
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
```

#### Documentation Style

- Use clear, concise language
- Provide examples
- Explain the "why", not just the "what"
- Keep documentation up-to-date with code

### 4. Testing

#### Writing Tests

- Test one thing per test
- Use descriptive test names
- Include positive and negative cases
- Mock external dependencies
- Test edge cases

Example:

```python
def test_label_generator_classification():
    """Test classification label generation."""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'ticker': 'SPY',
        'close': np.random.randn(100).cumsum() + 100,
        'log_return': np.random.randn(100) * 0.01
    })
    
    gen = LabelGenerator(threshold=0.02, horizon=1)
    result = gen.create_classification_labels(df)
    
    # Check label column exists
    assert 'label' in result.columns
    
    # Check label values are valid
    assert set(result['label'].dropna().unique()).issubset({0, 1})
    
    # Check future_return is calculated
    assert 'future_return' in result.columns
```

#### Test Coverage

- Aim for >80% coverage
- Focus on critical paths
- Don't test external libraries
- Test edge cases and error handling

### 5. Areas for Contribution

#### High Priority

- [ ] Improve documentation
- [ ] Add more unit tests
- [ ] Optimize performance
- [ ] Add more data sources
- [ ] Implement deep learning models

#### Medium Priority

- [ ] Add more technical indicators
- [ ] Improve visualization
- [ ] Add portfolio optimization
- [ ] Implement risk management
- [ ] Add real-time streaming

#### Low Priority

- [ ] Web dashboard
- [ ] API endpoints
- [ ] Database integration
- [ ] Cloud deployment
- [ ] Monitoring and alerts

### 6. Adding New Features

#### Adding a New Data Source

1. Create downloader in `src/market_event_ai/data/downloaders.py`
2. Add schema in `src/market_event_ai/data/schemas.py`
3. Add preprocessing in `src/market_event_ai/preprocess/preprocessors.py`
4. Update CLI command
5. Add tests
6. Update documentation

Example structure:

```python
class NewDataDownloader:
    """Download data from new source."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def download(self, start_date: str, end_date: str) -> Path:
        """Download data for date range."""
        # Implementation
        pass
```

#### Adding a New Model

1. Create trainer in `src/market_event_ai/models/trainers.py`
2. Implement `create_model()` method
3. Add to CLI choices
4. Add tests
5. Update documentation

Example:

```python
def create_model(self):
    """Create model instance."""
    if self.model_type == 'my_new_model':
        return MyNewModel(
            param1=value1,
            random_state=self.random_seed
        )
    # ...
```

#### Adding New Features

1. Extend `FeatureEngineer` in `src/market_event_ai/features/extractors.py`
2. Add feature generation method
3. Test for data leakage
4. Add tests
5. Update documentation

### 7. Documentation

#### Code Documentation

- All public functions need docstrings
- Use Google style docstrings
- Include type hints
- Provide examples

#### User Documentation

- Update README.md for user-facing changes
- Update ARCHITECTURE.md for design changes
- Add examples to QUICKSTART.md
- Update CLI help text

### 8. Review Process

1. **Automated Checks**
   - Tests must pass
   - Code style checks
   - Coverage requirements

2. **Manual Review**
   - Code quality
   - Design decisions
   - Documentation
   - Test coverage

3. **Feedback**
   - Address review comments
   - Update PR based on feedback
   - Re-request review

4. **Merge**
   - Squash commits if needed
   - Update changelog
   - Close related issues

## Development Tips

### Debugging

```python
# Use logging instead of print
import logging
logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.error("Error message")
```

### Testing Locally

```bash
# Run specific test
pytest tests/unit/test_config.py::test_settings_initialization -v

# Run with debugging
pytest --pdb

# Run with coverage
pytest --cov=market_event_ai --cov-report=term-missing
```

### Performance Profiling

```bash
# Profile with cProfile
python -m cProfile -s cumulative -m market_event_ai.cli train

# Use line profiler for specific functions
kernprof -l -v script.py
```

## Questions?

- Open an issue for questions
- Check existing issues and PRs
- Read the documentation thoroughly

## Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort! 🎉
