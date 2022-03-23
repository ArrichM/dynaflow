import click
import time
import dynaflow.tracking_store as tracking_store
import dynaflow.model_registry as model_registry


@click.group()
def dynaflow():
    pass


def _deploy(
    base_name: str, region: str, read_capacity_units: int, write_capacity_units: int
):

    # Deploy the two tables
    for model, suffix in zip(
        [tracking_store.BaseEntry, model_registry.BaseEntry],
        ["-tracking-store", "-model-registry"],
    ):
        model.Meta.table_name = base_name + suffix
        model.Meta.region = region
        # Create the table if it does not exist yet
        if not model.exists():
            click.echo(f"Deploying table: {base_name}{suffix} ({region})...")
            model.create_table(
                read_capacity_units=read_capacity_units,
                write_capacity_units=write_capacity_units,
                wait=True,
            )
            # Wait until the table is ready for usage
            while not model.exists():
                time.sleep(0.5)
            click.echo(f"Successfully deployed table: {base_name}{suffix} ({region})")
        else:
            click.echo(f"Table {base_name}{suffix} ({region}) already exists")


def _destroy(base_name: str, region: str):

    # Deploy the two tables
    for model, suffix in zip(
        [tracking_store.BaseEntry, model_registry.BaseEntry],
        ["-tracking-store", "-model-registry"],
    ):
        model.Meta.table_name = base_name + suffix
        model.Meta.region = region
        # Create the table if it does not exist yet
        if model.exists():
            model.delete_table()
            click.echo(f"Successfully deleted table: {base_name}{suffix} ({region})")

        else:
            click.echo(f"Table {base_name}{suffix} ({region}) does not exist")


@dynaflow.command()
@click.option(
    "--base-name",
    default="mlflow",
    help="The name of the AWS dynamodb table to create.",
)
@click.option(
    "--region",
    default="eu-central-1",
    help="The region in which to create the AWS Dynamodb table.",
)
@click.option(
    "--read-capacity-units",
    default=1,
    help="The provisioned RCU for the table.",
)
@click.option(
    "--write-capacity-units",
    default=1,
    help="The provisioned WCU for the table.",
)
def deploy(
    base_name: str, region: str, read_capacity_units: int, write_capacity_units: int
):
    _deploy(base_name, region, read_capacity_units, write_capacity_units)


@dynaflow.command()
@click.option(
    "--base-name",
    default="mlflow",
    help="The name of the AWS dynamodb table to create.",
)
@click.option(
    "--region",
    default="eu-central-1",
    help="The region in which to create the AWS Dynamodb table.",
)
def destroy(base_name: str, region: str):

    _destroy(base_name, region)
