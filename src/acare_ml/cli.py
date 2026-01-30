import click
from pathlib import Path


@click.group()
@click.version_option(version="0.1.0")
def main():
    """ACare ML pipeline CLI"""
    pass


@main.command()
@click.option("--config", type=click.Path(exists=True), help="Config YAML path")
@click.option("--output-dir", type=click.Path(), default="data/interim", help="Output directory")
def build_dataset(config, output_dir):
    """Build dataset from raw data"""
    click.echo(f"Building dataset with config: {config}")
    click.echo(f"Output to: {output_dir}")


@main.command()
@click.option("--config", type=click.Path(exists=True), help="Config YAML path")
@click.option("--input-dir", type=click.Path(), default="data/interim", help="Input directory")
@click.option("--output-dir", type=click.Path(), default="data/processed", help="Output directory")
def build_features(config, input_dir, output_dir):
    """Build features from interim data"""
    click.echo(f"Building features with config: {config}")
    click.echo(f"Input from: {input_dir}")
    click.echo(f"Output to: {output_dir}")


@main.command()
@click.option("--config", type=click.Path(exists=True), required=True, help="Training config YAML")
@click.option("--data-dir", type=click.Path(), default="data/processed", help="Processed data dir")
@click.option("--output-dir", type=click.Path(), default="artifacts/models", help="Model output dir")
def train(config, data_dir, output_dir):
    """Train model"""
    click.echo(f"Training model with config: {config}")
    click.echo(f"Data from: {data_dir}")
    click.echo(f"Saving model to: {output_dir}")


@main.command()
@click.option("--model-path", type=click.Path(exists=True), required=True, help="Trained model path")
@click.option("--data-path", type=click.Path(exists=True), required=True, help="Input data path")
@click.option("--output-path", type=click.Path(), help="Predictions output path")
def infer(model_path, data_path, output_path):
    """Run inference with trained model"""
    click.echo(f"Loading model from: {model_path}")
    click.echo(f"Running inference on: {data_path}")
    if output_path:
        click.echo(f"Saving predictions to: {output_path}")


if __name__ == "__main__":
    main()
