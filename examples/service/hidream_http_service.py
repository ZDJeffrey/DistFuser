from argparse import ArgumentParser
from omegaconf import OmegaConf

from distfuser import (
    run_api_server,
    Engine,
    SimpleT2XWorkflow,
    HiDreamTextEncoderWorker,
    HiDreamDiTWorker,
    HiDreamVaeWorker,
)


def main(args):
    config = OmegaConf.load(args.config)

    workflow_conf = config.get("workflow", {})
    dit_conf = workflow_conf.get("dit", {})
    encoder_conf = workflow_conf.get("encoder", {})
    vae_conf = workflow_conf.get("vae", {})

    # Initialize the Engine instances
    dit = Engine.from_config(HiDreamDiTWorker, dit_conf)
    encoder = Engine.from_config(HiDreamTextEncoderWorker, encoder_conf)
    vae = Engine.from_config(HiDreamVaeWorker, vae_conf)

    # # Create the API server
    http_server_conf = config.get("http_server", {})
    run_api_server(
        SimpleT2XWorkflow(encoder, dit, vae),
        http_server_conf.get("host", "localhost"),
        http_server_conf.get("port", 8000),
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Run the HiDream HTTP server with a T2X workflow.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hidream/hidream_fast_service.yaml",
        help="Path to the configuration file for the HiDream service.",
    )
    args = parser.parse_args()
    main(args)
