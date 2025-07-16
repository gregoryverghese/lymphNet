# Contributing to LymphNet

Thank you for your interest in contributing to LymphNet! This document provides guidelines and information for contributors.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/lymphNet.git
   cd lymphNet
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install development dependencies
   ```

## 📝 Development Guidelines

### Code Style

- Follow **PEP 8** style guidelines
- Use **Black** for code formatting
- Keep line length under 88 characters
- Use meaningful variable and function names

### Code Quality

- Write **docstrings** for all functions and classes
- Add **type hints** where appropriate
- Include **error handling** for edge cases
- Write **unit tests** for new functionality

### Git Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit with descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: brief description of changes"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub

### Commit Message Format

Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## 🧪 Testing

Run tests before submitting:
```bash
pytest tests/
```

## 📚 Documentation

- Update **README.md** if adding new features
- Add **docstrings** to new functions/classes
- Update **requirements.txt** if adding dependencies

## 🐛 Reporting Issues

When reporting bugs, please include:
- **Description** of the problem
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Error messages** if applicable

## 💡 Feature Requests

For feature requests:
- Describe the **use case** and **benefit**
- Provide **examples** if possible
- Consider **implementation complexity**

## 📋 Pull Request Checklist

Before submitting a PR, ensure:
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation is updated
- [ ] No hardcoded paths or credentials
- [ ] Error handling is included
- [ ] Commit messages are descriptive

## 🤝 Code Review Process

1. **Automated checks** must pass
2. **At least one maintainer** must approve
3. **Address feedback** from reviewers
4. **Squash commits** if requested

## 📞 Getting Help

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact gregory.verghese@gmail.com

Thank you for contributing to LymphNet! 🎉 