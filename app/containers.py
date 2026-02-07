from pathlib import Path
from dependency_injector import containers, providers
import logging.config
from pydantic_ai import Embedder, Agent
from . import gateways, services


class Core(containers.DeclarativeContainer):
    config = providers.Configuration()

    logging = providers.Resource(logging.config.dictConfig, config=config.logging)


class Gateways(containers.DeclarativeContainer):
    config = providers.Configuration()

    neo4j = providers.Resource(
        gateways.Neo4jGateway,
        uri=config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password,
    )
    db = providers.Singleton(
        gateways.DatabaseGateway,
        uri=config.database.uri,
    )

    graph_llm = providers.Singleton(
        Agent,
        model=config.llm.graph.model,
    )
    graph_embedder = providers.Singleton(
        Embedder,
        model=config.embedding.graph.model,
    )


class Services(containers.DeclarativeContainer):
    config = providers.Configuration()
    gateways = providers.DependenciesContainer()

    graph = providers.Factory(
        services.GraphService,
        session_factory=gateways.neo4j.provided.session,
        embedder=gateways.graph_embedder,
    )


class Application(containers.DeclarativeContainer):
    config = providers.Configuration(
        yaml_files=[Path(__file__).parents[1] / "config.yaml"], strict=True
    )

    core = providers.Container(Core, config=config.core)
    gateways = providers.Container(Gateways, config=config.gateways)

    services = providers.Container(Services, config=config, gateways=gateways)
