import typer
from src.app import main as run_app


def cli_main(
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        "-r",
        help="Fuerza la reconstrucción del corpus (ignora la caché)",
    ),
):
    """Lanza la interfaz del buscador para El Quijote."""
    run_app(rebuild=rebuild)


def main():
    typer.run(cli_main)


if __name__ == "__main__":
    main()
