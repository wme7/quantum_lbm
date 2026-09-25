"""CLI smoke tests for classical and quantum subcommands."""

from typer.testing import CliRunner

from lbm.cli import app

runner = CliRunner()


def test_run_classical_help():
    result = runner.invoke(app, ["run", "classical", "--help"])
    assert result.exit_code == 0
    assert "Shan" in result.stdout or "classical" in result.stdout.lower()


def test_run_quantum_help():
    result = runner.invoke(app, ["run", "quantum", "--help"])
    assert result.exit_code == 0
    assert "--statistic" in result.stdout
    assert "--h" in result.stdout


def test_run_classical_tiny():
    result = runner.invoke(
        app,
        [
            "run",
            "classical",
            "--nx",
            "24",
            "--ny",
            "24",
            "--steps",
            "20",
            "--u-inf",
            "0.05",
            "--t-inf",
            "0.5",
            "--diameter",
            "4",
            "--re",
            "20",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "classical" in result.stdout.lower()


def test_run_quantum_mb_tiny():
    result = runner.invoke(
        app,
        [
            "run",
            "quantum",
            "--statistic",
            "MB",
            "--h",
            "1.0",
            "--nx",
            "24",
            "--ny",
            "24",
            "--steps",
            "20",
            "--u-inf",
            "0.05",
            "--t-inf",
            "0.5",
            "--diameter",
            "4",
            "--re",
            "20",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "quantum" in result.stdout.lower()
    assert "z_inf" in result.stdout


def test_run_quantum_fd_tiny():
    result = runner.invoke(
        app,
        [
            "run",
            "quantum",
            "--statistic",
            "FD",
            "--h",
            "1.0",
            "--nx",
            "24",
            "--ny",
            "24",
            "--steps",
            "20",
            "--u-inf",
            "0.05",
            "--t-inf",
            "0.5",
            "--diameter",
            "4",
            "--re",
            "20",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "FD" in result.stdout
